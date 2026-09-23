"""Shared request spacing and failure cooldown for configured web providers.

The SQLite path is optional for tests, but deployments should use one shared path
for every API worker. Only SHA-256 identities and safe HTTP codes are persisted.
"""

import asyncio
from contextlib import closing
from email.utils import parsedate_to_datetime
import hashlib
import os
from pathlib import Path
import sqlite3
import time


class WebGuardUnavailable(RuntimeError):
    pass


class WebGuardBlocked(RuntimeError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"web_upstream_circuit_open_{status}")


def retry_after_seconds(value: str | None, *, now: float) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return None
    return max(1.0, min(3600.0, seconds))


class WebProviderGuard:
    def __init__(self, path: Path | None, *, min_interval: float = 1.0,
                 auth_cooldown: float = 900.0, rate_cooldown: float = 60.0,
                 lease_seconds: float = 50.0):
        self.path = Path(path) if path else None
        self.min_interval = min_interval
        self.auth_cooldown = auth_cooldown
        self.rate_cooldown = rate_cooldown
        self.lease_seconds = lease_seconds
        self._lock = asyncio.Lock()
        self._memory: dict[str, tuple[float, float, int]] = {}

    @staticmethod
    def identity(provider: str, credential: str) -> str:
        return hashlib.sha256((provider + "\0" + credential).encode()).hexdigest()

    def _connect(self):
        assert self.path is not None
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.execute("CREATE TABLE IF NOT EXISTS provider_guard ("
                           "identity TEXT PRIMARY KEY, next_at REAL NOT NULL, "
                           "blocked_until REAL NOT NULL, status INTEGER NOT NULL)")
        return connection

    def _reserve_disk(self, identity: str, now: float) -> tuple[float, int]:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT next_at, blocked_until, status FROM provider_guard WHERE identity=?",
                    (identity,)).fetchone()
                if row and row[1] > now:
                    return 0.0, row[2]
                if row and row[0] > now:
                    return row[0] - now, 0
                connection.execute(
                    "INSERT INTO provider_guard(identity,next_at,blocked_until,status) "
                    "VALUES(?,?,0,0) ON CONFLICT(identity) DO UPDATE SET "
                    "next_at=excluded.next_at, blocked_until=0, status=0",
                    (identity, now + self.lease_seconds))
                return 0.0, 0

    def _block_disk(self, identity: str, until: float, status: int):
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO provider_guard(identity,next_at,blocked_until,status) "
                    "VALUES(?,0,?,?) ON CONFLICT(identity) DO UPDATE SET "
                    "blocked_until=MAX(blocked_until,excluded.blocked_until),status=excluded.status",
                    (identity, until, status))

    def _complete_disk(self, identity: str, next_at: float):
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("UPDATE provider_guard SET next_at=? WHERE identity=?",
                                   (next_at, identity))

    async def reserve(self, identity: str):
        while True:
            now = time.time()
            try:
                if self.path:
                    wait, status = await asyncio.to_thread(self._reserve_disk, identity, now)
                else:
                    async with self._lock:
                        next_at, blocked_until, status = self._memory.get(identity, (0.0, 0.0, 0))
                        if blocked_until > now:
                            wait = 0.0
                        elif next_at > now:
                            wait, status = next_at - now, 0
                        else:
                            self._memory[identity] = (now + self.lease_seconds, 0.0, 0)
                            wait = status = 0
            except (OSError, sqlite3.Error) as exc:
                raise WebGuardUnavailable("web_guard_unavailable") from exc
            if status:
                raise WebGuardBlocked(status)
            if not wait:
                return
            # Another worker may finish or block the credential before its lease ends.
            await asyncio.sleep(min(wait, 0.25))

    async def complete(self, identity: str):
        next_at = time.time() + self.min_interval
        try:
            if self.path:
                await asyncio.to_thread(self._complete_disk, identity, next_at)
            else:
                async with self._lock:
                    _, blocked_until, status = self._memory.get(identity, (0.0, 0.0, 0))
                    self._memory[identity] = (next_at, blocked_until, status)
        except (OSError, sqlite3.Error) as exc:
            raise WebGuardUnavailable("web_guard_unavailable") from exc

    async def block(self, identity: str, status: int, retry_after: str | None = None):
        now = time.time()
        if status in {401, 402, 403, 432, 433}:
            seconds = self.auth_cooldown
        elif status == 429:
            seconds = retry_after_seconds(retry_after, now=now) or self.rate_cooldown
        elif 500 <= status <= 599:
            seconds = 10.0
        else:
            return
        until = now + seconds
        try:
            if self.path:
                await asyncio.to_thread(self._block_disk, identity, until, status)
            else:
                async with self._lock:
                    next_at, blocked_until, _ = self._memory.get(identity, (0.0, 0.0, 0))
                    self._memory[identity] = (next_at, max(blocked_until, until), status)
        except (OSError, sqlite3.Error) as exc:
            raise WebGuardUnavailable("web_guard_unavailable") from exc
