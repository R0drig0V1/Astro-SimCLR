"""Tests of the polite HTTP client and the response recorder (task 3.1).

A fake clock and sleep stand in for time, and a fake opener for the network.
"""

from __future__ import annotations

import http.client
import io
import urllib.error
import urllib.request
from collections.abc import Iterator
from email.message import Message
from typing import Any

import pytest

from astro_simclr.alerts import transport
from astro_simclr.alerts.snapshot import Recorder, online_fetch, utc_now
from astro_simclr.alerts.transport import (
    USER_AGENT,
    Fetcher,
    HttpError,
    RateLimiter,
    urllib_opener,
)


class FakeTime:
    """A clock that only moves when slept on."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x", code, "error", headers, None)


def scripted(outcomes: list[Any]) -> Any:
    """An opener that raises or returns the next outcome on each call."""
    it: Iterator[Any] = iter(outcomes)

    def opener(url: str, timeout: float, max_bytes: int) -> bytes:
        outcome = next(it)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, bytes)
        return outcome

    return opener


def test_rate_limiter_spaces_requests() -> None:
    t = FakeTime()
    limiter = RateLimiter(2.0, clock=t.clock, sleep=t.sleep)
    for _ in range(3):
        limiter.wait()
    assert t.sleeps == [0.5, 0.5]
    t.now += 10  # idle time is not banked
    limiter.wait()
    assert t.sleeps == [0.5, 0.5]
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(0)


def test_fetcher_retries_5xx_and_timeouts_with_exponential_backoff() -> None:
    t = FakeTime()
    opener = scripted([http_error(503), TimeoutError("timed out"), http_error(502), b"body"])
    fetcher = Fetcher(rate=1000.0, opener=opener, clock=t.clock, sleep=t.sleep)
    assert fetcher.get("https://x", max_bytes=10) == b"body"
    assert fetcher.requests == 4
    assert [s for s in t.sleeps if s >= 1] == [2.0, 4.0, 8.0]


def test_fetcher_honours_retry_after_beyond_the_backoff_cap() -> None:
    t = FakeTime()
    outcomes = [http_error(429, "30"), http_error(429, "120"), http_error(429, "9999")]
    opener = scripted([*outcomes, http_error(429, "a date"), b"ok"])
    fetcher = Fetcher(rate=1000.0, opener=opener, clock=t.clock, sleep=t.sleep, backoff_cap=60.0)
    assert fetcher.get("https://x", max_bytes=10) == b"ok"
    # Retry-After wins over the 60 s cap, up to MAX_RETRY_AFTER (600 s).
    assert [s for s in t.sleeps if s >= 1] == [30.0, 120.0, 600.0, 16.0]


def test_fetcher_gives_up_and_does_not_retry_client_errors() -> None:
    t = FakeTime()
    fetcher = Fetcher(
        rate=1000.0,
        opener=scripted([ConnectionResetError("reset")] * 3),
        clock=t.clock,
        sleep=t.sleep,
        max_attempts=3,
    )
    with pytest.raises(HttpError, match="after 3 attempts") as caught:
        fetcher.get("https://x", max_bytes=10)
    assert caught.value.status is None and fetcher.requests == 3
    fetcher = Fetcher(rate=1000.0, opener=scripted([http_error(404)]), clock=t.clock, sleep=t.sleep)
    with pytest.raises(HttpError, match="HTTP 404") as caught:
        fetcher.get("https://x", max_bytes=10)
    assert caught.value.status == 404 and fetcher.requests == 1
    with pytest.raises(ValueError, match="max_attempts"):
        Fetcher(rate=1.0, max_attempts=0)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _FakeOpener:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.seen: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, timeout: float) -> _Response:
        self.seen.append(request)
        return _Response(self.body)


def test_urllib_opener_sends_a_generic_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOpener(b"x" * 20)
    monkeypatch.setattr(transport, "_OPENER", fake)
    assert urllib_opener("https://api.example/", 5.0, 100) == b"x" * 20
    assert fake.seen[0].get_header("User-agent") == USER_AGENT
    assert USER_AGENT.startswith("astro-simclr/") and "@" not in USER_AGENT
    with pytest.raises(HttpError, match="larger than 10"):
        urllib_opener("https://api.example/", 5.0, 10)
    with pytest.raises(ValueError, match="https"):
        urllib_opener("http://api.example/", 5.0, 10)


def test_urllib_opener_enforces_a_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transport, "_OPENER", _FakeOpener(b"y" * (3 * 64 * 1024)))
    ticks = iter(range(100))
    with pytest.raises(TimeoutError, match="not complete after 1 s"):
        urllib_opener(
            "https://api.example/", 5.0, 10**6, deadline=1.0, clock=lambda: float(next(ticks))
        )


def test_redirects_must_stay_on_https() -> None:
    handler = transport._HttpsRedirects()
    request = urllib.request.Request("https://api.example/a")
    headers = http.client.HTTPMessage()
    with pytest.raises(HttpError, match="non-https"):
        handler.redirect_request(request, io.BytesIO(), 302, "Found", headers, "http://evil/")
    follow = handler.redirect_request(request, io.BytesIO(), 302, "Found", headers, "https://b/")
    assert follow is not None and follow.full_url == "https://b/"


def test_recorder_caches_records_and_lists_row_responses() -> None:
    bodies = {"https://a": b"A", "https://b": b"B"}
    calls: list[str] = []

    def fetch(url: str, max_bytes: int) -> tuple[bytes, str]:
        calls.append(url)
        return bodies[url], "2026-11-08T00:00:00+00:00"

    recorder = Recorder(fetch, requests=lambda: len(calls), cache_bytes=1)
    recorder.start_row()
    assert recorder.get("https://a", max_bytes=9, kind="objects", cache=True) == b"A"
    assert recorder.get("https://a", max_bytes=9, kind="objects", cache=True) == b"A"
    assert calls == ["https://a"] and recorder.requests == 1
    recorder.get("https://b", max_bytes=9, kind="detections", cache=True)  # evicts a
    recorder.get("https://a", max_bytes=9, kind="objects", cache=True)
    recorder.get("https://b", max_bytes=9, kind="avro", cache=False)
    assert calls == ["https://a", "https://b", "https://a", "https://b"]
    assert len(recorder.row_refs) == 5 and recorder.row_refs[0] == recorder.row_refs[1]
    pending = recorder.pending()
    assert [r.url for r in pending] == calls and recorder.pending() == pending
    assert pending[0].sha256 == "559aead08264d5795d3909718cdd05abd49572e84fe55590eef31a88a08fdffd"
    recorder.drop_pending(3)
    assert [r.url for r in recorder.pending()] == calls[3:]
    recorder.start_row()
    assert recorder.row_refs == []
    with pytest.raises(ValueError, match="kind"):
        recorder.get("https://a", max_bytes=9, kind="other", cache=False)
    with pytest.raises(ValueError, match="cache_bytes"):
        Recorder(fetch, cache_bytes=-1)
    big = Recorder(lambda url, n: (b"z" * 10, "T"), cache_bytes=5)  # over the budget
    big.get("https://z", max_bytes=99, kind="objects", cache=True)
    big.get("https://z", max_bytes=99, kind="objects", cache=True)
    assert len(big.pending()) == 2  # never cached


def test_online_fetch_stamps_the_retrieval_time() -> None:
    fetcher = Fetcher(rate=1000.0, opener=scripted([b"x"]))
    body, when = online_fetch(fetcher, now=lambda: "T")("https://x", 10)
    assert (body, when) == (b"x", "T")
    assert utc_now().endswith("+00:00")
