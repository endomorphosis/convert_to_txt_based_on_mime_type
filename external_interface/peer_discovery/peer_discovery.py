"""
LAN Peer Discovery using UDP broadcast.

Each node periodically broadcasts its presence on a well-known UDP port.
All nodes listen on the same port and maintain a list of discovered peers
(host, port) tuples where ``port`` is the HTTP job-queue endpoint port.

Security note
-------------
The UDP listener binds to all interfaces so that it can receive subnet
broadcast packets.  All incoming announcements are validated against known
private IP ranges (RFC 1918 / RFC 4193) so that only LAN peers are
registered.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
import time
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)

# How often (seconds) a node re-announces itself on the LAN.
_ANNOUNCE_INTERVAL = 10

# Peers not seen within this window (seconds) are removed from the registry.
_PEER_TTL = 60


class PeerDiscovery:
    """
    Discovers other nodes running on the same LAN segment via UDP broadcast.

    Attributes:
        _peers: mapping from ``(host, job_port)`` to the last-seen timestamp.

    Usage::

        discovery = PeerDiscovery(job_port=8765, discovery_port=8766)
        await discovery.start()
        ...
        peers = discovery.get_peers()   # {(host, port), ...}
        await discovery.stop()
    """

    def __init__(self, job_port: int = 8765, discovery_port: int = 8766) -> None:
        """
        Args:
            job_port: The TCP port on which this node's HTTP job-queue
                endpoint is listening. Broadcast in announcements so that
                peers know where to drain jobs from.
            discovery_port: The UDP port used for broadcast announcements
                and reception.
        """
        self._job_port = job_port
        self._discovery_port = discovery_port

        self._peers: Dict[Tuple[str, int], float] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background announcement and listener tasks."""
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._announce_loop()),
            asyncio.create_task(self._listen_loop()),
            asyncio.create_task(self._eviction_loop()),
        ]
        logger.info(
            "PeerDiscovery started (job_port=%d, discovery_port=%d)",
            self._job_port,
            self._discovery_port,
        )

    async def stop(self) -> None:
        """Cancel background tasks and release resources."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("PeerDiscovery stopped")

    def get_peers(self) -> Set[Tuple[str, int]]:
        """Return the set of currently known peers as ``(host, job_port)`` tuples."""
        return set(self._peers.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_announcement(self) -> bytes:
        payload = json.dumps({"job_port": self._job_port})
        return payload.encode()

    def _parse_announcement(self, data: bytes, sender_host: str) -> Tuple[str, int] | None:
        try:
            msg = json.loads(data.decode())
            job_port = int(msg["job_port"])
            return sender_host, job_port
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    async def _announce_loop(self) -> None:
        """Periodically broadcast our presence to the LAN."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)
        try:
            payload = self._build_announcement()
            loop = asyncio.get_running_loop()
            while self._running:
                try:
                    await loop.sock_sendto(
                        sock, payload, ("<broadcast>", self._discovery_port)
                    )
                    logger.debug("Broadcast announcement sent")
                except OSError as exc:
                    logger.warning("Broadcast send failed: %s", exc)
                await asyncio.sleep(_ANNOUNCE_INTERVAL)
        finally:
            sock.close()

    async def _listen_loop(self) -> None:
        """Listen for announcements from peers and register them."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass  # SO_REUSEPORT not available on all platforms (e.g. Windows)
        sock.bind(("", self._discovery_port))
        sock.setblocking(False)
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                try:
                    data, (host, _) = await loop.sock_recvfrom(sock, 1024)
                except OSError as exc:
                    logger.warning("Discovery receive failed: %s", exc)
                    await asyncio.sleep(1)
                    continue

                peer = self._parse_announcement(data, host)
                if peer is None:
                    continue

                # Reject announcements from non-private (non-LAN) addresses.
                if not _is_private_address(host):
                    logger.debug("Ignoring announcement from non-LAN address: %s", host)
                    continue

                # Don't register ourselves.
                if peer[1] == self._job_port and _is_local_address(host):
                    continue

                if peer not in self._peers:
                    logger.info("Discovered new peer: %s:%d", *peer)
                self._peers[peer] = time.monotonic()
        finally:
            sock.close()

    async def _eviction_loop(self) -> None:
        """Remove peers that have not been seen recently."""
        while self._running:
            await asyncio.sleep(_ANNOUNCE_INTERVAL)
            cutoff = time.monotonic() - _PEER_TTL
            stale = [p for p, ts in self._peers.items() if ts < cutoff]
            for peer in stale:
                logger.info("Removing stale peer: %s:%d", *peer)
                del self._peers[peer]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_local_address(host: str) -> bool:
    """Return True if *host* resolves to one of this machine's own addresses."""
    try:
        local_host = socket.gethostname()
        local_addrs = {
            info[4][0]
            for info in socket.getaddrinfo(local_host, None)
        }
        local_addrs.add("127.0.0.1")
        local_addrs.add("::1")
        return host in local_addrs
    except OSError:
        return False


def _is_private_address(host: str) -> bool:
    """Return True if *host* is a private / loopback / link-local address.

    Only announcements from private RFC 1918 / RFC 4193 address space are
    accepted.  This prevents a public-internet peer from registering itself
    with the local node via a crafted UDP packet.
    """
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return False
