"""The polite HTTP client of ``fetch-alerts``: one rate limit, retries with backoff (task 3.1).

ALeRCE's public services are shared, so every request of a run (API and avro, first
attempts and retries alike) goes through one :class:`RateLimiter`: request starts are at
least ``1 / rate`` seconds apart. :class:`Fetcher` retries what is transient, a timeout or
connection failure, HTTP 5xx and HTTP 429, with exponential backoff (``backoff · 2^(k-1)``
seconds before attempt ``k + 1``, capped at ``backoff_cap``, and never shorter than a
numeric ``Retry-After`` up to :data:`MAX_RETRY_AFTER`). Any other HTTP status (404, 400,
...) is final. No jitter: a run is a
single sequential client, so there is no herd to spread out, and the delays stay testable.

Requests carry a generic ``User-Agent`` (the package name and version, no personal data).
Only ``https://`` URLs are opened, redirects included, and response bodies are read up to
a caller-given size within a total deadline (:data:`READ_DEADLINE`; the socket timeout is
per operation only), so a misbehaving server can neither send an unbounded body nor
stall the run by trickling bytes.

The clock, the sleep and the transport (:data:`Opener`) are injectable, so the tests run
without network and without waiting.
"""

from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from importlib.metadata import version
from typing import IO

__all__ = [
    "MAX_RETRY_AFTER",
    "READ_DEADLINE",
    "USER_AGENT",
    "Fetcher",
    "HttpError",
    "Opener",
    "RateLimiter",
    "https_opener",
    "urllib_opener",
]

#: The ``User-Agent`` of every request: the tool, never a person.
USER_AGENT = f"astro-simclr/{version('astro-simclr')} (research; raw alert retrieval)"

#: Longest ``Retry-After`` honoured, in seconds (a longer one is cut to this).
MAX_RETRY_AFTER = 600.0
#: Longest time one response may take to arrive in full, in seconds.
READ_DEADLINE = 300.0
#: Bytes per read while streaming a body.
_CHUNK = 64 * 1024

#: Transport: ``(url, timeout seconds, max bytes) -> body``. It raises
#: ``urllib.error.HTTPError`` for an HTTP error status, an ``OSError`` or
#: ``http.client.HTTPException`` for a network failure, and :class:`HttpError` for a
#: response it refuses (too large).
type Opener = Callable[[str, float, int], bytes]


class HttpError(Exception):
    """A request failed for good: a final HTTP status, or every retry failed.

    Attributes:
        url: the requested URL.
        status: the last HTTP status, or ``None`` for a network failure.
    """

    def __init__(self, url: str, status: int | None, message: str) -> None:
        self.url = url
        self.status = status
        super().__init__(f"{url}: {message}")


class _HttpsRedirects(urllib.request.HTTPRedirectHandler):
    """Follows redirects only to ``https://`` URLs."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not newurl.startswith("https://"):
            raise HttpError(req.full_url, code, f"redirect to a non-https URL {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsRedirects())


def https_opener() -> urllib.request.OpenerDirector:
    """A urllib opener that follows redirects to ``https://`` URLs only (as :func:`urllib_opener`).

    For streaming downloads too large for :class:`Fetcher` (task 3.5's Zenodo archive).
    """
    return urllib.request.build_opener(_HttpsRedirects())


def urllib_opener(
    url: str,
    timeout: float,
    max_bytes: int,
    *,
    deadline: float = READ_DEADLINE,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """GET ``url`` with :data:`USER_AGENT` and return at most ``max_bytes`` of body.

    Redirects are followed to ``https://`` URLs only. The body is read in chunks, and the
    whole response must arrive within ``deadline`` seconds.

    Raises:
        ValueError: if ``url`` is not ``https://``.
        HttpError: if the body is larger than ``max_bytes``, or a redirect leaves https.
        TimeoutError: if the deadline passes (transient: :class:`Fetcher` retries it).
        urllib.error.HTTPError: for an HTTP error status.
        OSError, http.client.HTTPException: for a network failure.
    """
    if not url.startswith("https://"):
        raise ValueError(f"refusing to request {url!r}: only https:// URLs")
    start = clock()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    chunks: list[bytes] = []
    size = 0
    with _OPENER.open(request, timeout=timeout) as response:
        while chunk := response.read(min(_CHUNK, max_bytes + 1 - size)):
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise HttpError(url, None, f"response larger than {max_bytes} bytes")
            if clock() - start > deadline:
                raise TimeoutError(f"response not complete after {deadline:g} s")
    return b"".join(chunks)


class RateLimiter:
    """Spaces calls of :meth:`wait` at least ``1 / rate`` seconds apart.

    Raises:
        ValueError: if ``rate`` is not positive.
    """

    def __init__(
        self,
        rate: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not rate > 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._next: float | None = None

    def wait(self) -> None:
        """Sleep until the next request may start, and reserve that slot."""
        now = self._clock()
        if self._next is not None and now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self.interval


class Fetcher:
    """GET with one rate limit and retries (module docstring); counts the requests made.

    Attributes:
        requests: the number of requests started so far (retries included).

    Raises:
        ValueError: if ``rate`` is not positive or ``max_attempts`` is below 1.
    """

    def __init__(
        self,
        *,
        rate: float,
        opener: Opener = urllib_opener,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 5,
        backoff: float = 2.0,
        backoff_cap: float = 60.0,
        timeout: float = 60.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self._limiter = RateLimiter(rate, clock=clock, sleep=sleep)
        self._opener = opener
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._backoff = backoff
        self._backoff_cap = backoff_cap
        self._timeout = timeout
        self.requests = 0

    def get(self, url: str, *, max_bytes: int) -> bytes:
        """GET ``url``, retrying transient failures; return the body.

        Raises:
            HttpError: on a final HTTP status (anything but 5xx and 429), a refused
                response, or when all ``max_attempts`` attempts failed.
        """
        status: int | None = None
        reason = ""
        for attempt in range(1, self._max_attempts + 1):
            self._limiter.wait()
            self.requests += 1
            retry_after: float | None = None
            try:
                return self._opener(url, self._timeout, max_bytes)
            except urllib.error.HTTPError as err:
                status, reason = err.code, f"HTTP {err.code}"
                if not (err.code >= 500 or err.code == 429):
                    raise HttpError(url, status, reason) from None
                retry_after = _retry_after(err.headers)
            except (OSError, http.client.HTTPException) as err:
                # URLError, TimeoutError, ConnectionError and SSLError are all OSError.
                status, reason = None, f"{type(err).__name__}: {err}"
            if attempt < self._max_attempts:
                delay = min(self._backoff * 2 ** (attempt - 1), self._backoff_cap)
                if retry_after is not None:
                    delay = max(delay, min(retry_after, MAX_RETRY_AFTER))
                self._sleep(delay)
        raise HttpError(url, status, f"{reason} (after {self._max_attempts} attempts)")


def _retry_after(headers: Message | None) -> float | None:
    """The ``Retry-After`` header in seconds, if it is a number (dates are ignored)."""
    value = None if headers is None else headers.get("Retry-After")
    try:
        return None if value is None else max(0.0, float(value))
    except ValueError:
        return None
