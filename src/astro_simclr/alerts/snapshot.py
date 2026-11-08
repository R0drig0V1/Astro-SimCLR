"""The response snapshot and cache of ``fetch-alerts`` (task 3.1, plan-review amendments 2 and 6).

Every raw response a run receives (cone-search JSON, detection-list JSON, alert avro) is
kept, byte for byte, with its URL, kind, retrieval time (UTC, ISO 8601) and SHA-256, so the
index can be rebuilt offline from the snapshot (``fetch-alerts --from-snapshot``) and any
line of it traced to the bytes it was decided on. The full detection list of every object
is therefore stored too, which the phase-3 reconstruction needs (a random alert, or the
first alert, per object).

:class:`Recorder` sits between :class:`~astro_simclr.alerts.alerce.AlerceClient` and the
network (or a snapshot):

- it caches the cacheable responses (cone searches and detection lists) by URL, in an LRU
  bounded by total bytes (detection lists can be hundreds of KB), because rows share
  objects (about 1,392 rows of ``red`` share a position with another row); a cache hit is
  not a request;
- it keeps each new response until the store has committed it
  (:meth:`Recorder.pending`, then :meth:`Recorder.drop_pending` once the shard and its
  index lines are durable), so a failed commit loses no body;
- it lists the responses the current row used (:meth:`Recorder.start_row`,
  :attr:`Recorder.row_refs`), recorded as the row's ``responses`` column.

The cache lives in memory only: after a restart, a shared object is fetched again (and
stored again), which costs requests but never correctness.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from astro_simclr.alerts.transport import Fetcher

__all__ = ["KINDS", "RawFetch", "Recorder", "Response", "online_fetch", "utc_now"]

#: The kinds of response, as tagged by the client.
KINDS: tuple[str, ...] = ("objects", "detections", "avro")

#: ``(url, max_bytes) -> (body, retrieval time)``: the network, or a snapshot.
type RawFetch = Callable[[str, int], tuple[bytes, str]]


def utc_now() -> str:
    """The current UTC time, ISO 8601 with seconds (``2026-11-08T10:35:17+00:00``)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def online_fetch(fetcher: Fetcher, now: Callable[[], str] = utc_now) -> RawFetch:
    """A :data:`RawFetch` over the network: ``fetcher.get``, stamped with ``now()``."""

    def fetch(url: str, max_bytes: int) -> tuple[bytes, str]:
        body = fetcher.get(url, max_bytes=max_bytes)
        return body, now()

    return fetch


@dataclass(frozen=True, slots=True)
class Response:
    """One raw response as stored in the snapshot."""

    url: str
    kind: str
    retrieved_at: str
    sha256: str
    body: bytes

    @property
    def ref(self) -> str:
        """``kind:sha256``, as listed in an index line's ``responses`` column."""
        return f"{self.kind}:{self.sha256}"


class Recorder:
    """A caching, recording :class:`~astro_simclr.alerts.alerce.Getter` (module docstring).

    ``requests`` reports the network requests of the underlying fetch (0 offline).
    ``cache_bytes`` bounds the cached bodies' total size (0 disables the cache).

    Raises:
        ValueError: if ``cache_bytes`` is negative.
    """

    def __init__(
        self,
        fetch: RawFetch,
        *,
        requests: Callable[[], int] = lambda: 0,
        cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if cache_bytes < 0:
            raise ValueError(f"cache_bytes must be non-negative, got {cache_bytes}")
        self._fetch = fetch
        self._requests = requests
        self._cache_bytes = cache_bytes
        self._cached = 0
        self._cache: OrderedDict[str, Response] = OrderedDict()
        self._pending: list[Response] = []
        self.row_refs: list[str] = []

    @property
    def requests(self) -> int:
        """Network requests so far."""
        return self._requests()

    def get(self, url: str, *, max_bytes: int, kind: str, cache: bool) -> bytes:
        """The body of ``url``, from the cache if ``cache`` and present, else fetched.

        A fetched response is recorded (pending for the store) and, if ``cache``, cached.

        Raises:
            ValueError: if ``kind`` is not one of :data:`KINDS`.
            HttpError: if the fetch fails (failures are neither recorded nor cached).
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if cache and url in self._cache:
            self._cache.move_to_end(url)
            hit = self._cache[url]
            self.row_refs.append(hit.ref)
            return hit.body
        body, retrieved_at = self._fetch(url, max_bytes)
        response = Response(url, kind, retrieved_at, hashlib.sha256(body).hexdigest(), body)
        self._pending.append(response)
        self.row_refs.append(response.ref)
        if cache and len(body) <= self._cache_bytes:
            self._cache[url] = response
            self._cached += len(body)
            while self._cached > self._cache_bytes:
                _, evicted = self._cache.popitem(last=False)
                self._cached -= len(evicted.body)
        return body

    def start_row(self) -> None:
        """Start listing the responses of a new row in :attr:`row_refs`."""
        self.row_refs = []

    def pending(self) -> list[Response]:
        """The responses fetched and not yet committed, oldest first (a copy)."""
        return list(self._pending)

    def drop_pending(self, n: int) -> None:
        """Forget the ``n`` oldest pending responses, once a commit has made them durable."""
        del self._pending[:n]
