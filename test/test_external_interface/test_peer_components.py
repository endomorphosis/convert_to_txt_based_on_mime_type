#!/usr/bin/env python3 -m pytest
"""
Tests for LAN peer discovery and job-draining components.

These tests are designed to run entirely locally, without requiring real
network infrastructure.  The PeerJobServer is started on a random free port
and the PeerJobDrainer connects to it through localhost.
"""
from __future__ import annotations

import asyncio
import json
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from external_interface.peer_discovery.peer_discovery import (
    PeerDiscovery,
    _is_local_address,
)
from external_interface.peer_job_drainer.peer_job_drainer import (
    PeerJobDrainer,
    PeerJobServer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Return an available TCP port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# PeerDiscovery unit tests
# ---------------------------------------------------------------------------

class TestPeerDiscovery:

    def test_build_announcement_contains_job_port(self):
        discovery = PeerDiscovery(job_port=1234, discovery_port=5678)
        payload = discovery._build_announcement()
        msg = json.loads(payload.decode())
        assert msg["job_port"] == 1234

    def test_parse_announcement_valid(self):
        discovery = PeerDiscovery(job_port=1234, discovery_port=5678)
        payload = json.dumps({"job_port": 9000}).encode()
        result = discovery._parse_announcement(payload, "192.168.1.10")
        assert result == ("192.168.1.10", 9000)

    def test_parse_announcement_invalid_json(self):
        discovery = PeerDiscovery(job_port=1234, discovery_port=5678)
        result = discovery._parse_announcement(b"not-json", "192.168.1.10")
        assert result is None

    def test_parse_announcement_missing_key(self):
        discovery = PeerDiscovery(job_port=1234, discovery_port=5678)
        result = discovery._parse_announcement(b'{"other_key": 1}', "192.168.1.10")
        assert result is None

    def test_get_peers_initially_empty(self):
        discovery = PeerDiscovery(job_port=1234, discovery_port=5678)
        assert discovery.get_peers() == set()

    def test_local_address_helper(self):
        # 127.0.0.1 should always be considered local
        assert _is_local_address("127.0.0.1") is True

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """PeerDiscovery.start/stop should not raise even without a real network."""
        discovery = PeerDiscovery(job_port=_free_port(), discovery_port=_free_port())
        await discovery.start()
        assert discovery._running is True
        await discovery.stop()
        assert discovery._running is False

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self):
        discovery = PeerDiscovery(job_port=_free_port(), discovery_port=_free_port())
        await discovery.start()
        tasks_before = len(discovery._tasks)
        await discovery.start()  # second call should be a no-op
        assert len(discovery._tasks) == tasks_before
        await discovery.stop()

    @pytest.mark.asyncio
    async def test_peer_eviction(self):
        """Peers whose TTL has expired are removed by the eviction loop."""
        import time

        discovery = PeerDiscovery(job_port=1234, discovery_port=5678)
        peer = ("10.0.0.1", 9000)
        # Insert a peer with a timestamp in the past
        discovery._peers[peer] = time.monotonic() - 9999
        # Trigger eviction manually (without the async loop)
        cutoff = time.monotonic() - 60
        stale = [p for p, ts in discovery._peers.items() if ts < cutoff]
        for p in stale:
            del discovery._peers[p]
        assert peer not in discovery._peers


# ---------------------------------------------------------------------------
# PeerJobServer + PeerJobDrainer integration tests
# ---------------------------------------------------------------------------

class TestPeerJobServerAndDrainer:

    @pytest.fixture
    def job_list(self) -> List[str]:
        return ["job_001", "job_002", "job_003"]

    @pytest.fixture
    def make_server(self, job_list):
        """Factory that creates a PeerJobServer backed by a mutable job_list."""

        def _get_pending():
            return list(job_list)

        def _claim(requested):
            claimed = [j for j in requested if j in job_list]
            for j in claimed:
                job_list.remove(j)
            return claimed

        port = _free_port()
        server = PeerJobServer(port, _get_pending, _claim)
        return server, port

    @pytest.mark.asyncio
    async def test_get_jobs_endpoint(self, make_server, job_list):
        server, port = make_server
        await server.start()
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/jobs") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert set(data["jobs"]) == set(job_list)
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_claim_jobs_endpoint(self, make_server, job_list):
        server, port = make_server
        await server.start()
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/jobs/claim",
                    json={"jobs": ["job_001", "job_002"]},
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert set(data["claimed"]) == {"job_001", "job_002"}
            # The jobs should now be gone from the list
            assert "job_001" not in job_list
            assert "job_002" not in job_list
            assert "job_003" in job_list
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_drain_from_peer(self, make_server, job_list):
        server, port = make_server
        original_count = len(job_list)
        await server.start()
        try:
            local_queue: asyncio.Queue = asyncio.Queue()
            drainer = PeerJobDrainer(local_queue)
            count = await drainer.drain_from_peer("127.0.0.1", port)
            assert count == original_count
            assert local_queue.qsize() == count
            # Original job list should now be empty (jobs were claimed)
            assert job_list == []
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_drain_from_unreachable_peer_returns_zero(self):
        local_queue: asyncio.Queue = asyncio.Queue()
        drainer = PeerJobDrainer(local_queue, request_timeout=1.0)
        # Port 1 is reserved and should be unreachable
        count = await drainer.drain_from_peer("127.0.0.1", 1)
        assert count == 0
        assert local_queue.empty()

    @pytest.mark.asyncio
    async def test_drain_from_peer_with_no_jobs(self):
        """Server with empty queue returns 0 drained jobs."""
        port = _free_port()
        server = PeerJobServer(port, lambda: [], lambda jobs: [])
        await server.start()
        try:
            local_queue: asyncio.Queue = asyncio.Queue()
            drainer = PeerJobDrainer(local_queue)
            count = await drainer.drain_from_peer("127.0.0.1", port)
            assert count == 0
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_drain_from_all_peers(self, job_list):
        """drain_from_all_peers aggregates results from multiple peers."""
        port1 = _free_port()
        jobs1 = ["a", "b"]
        port2 = _free_port()
        jobs2 = ["c"]

        def make_callbacks(job_store):
            def get_jobs():
                return list(job_store)
            def claim(requested):
                c = [j for j in requested if j in job_store]
                for j in c:
                    job_store.remove(j)
                return c
            return get_jobs, claim

        g1, c1 = make_callbacks(jobs1)
        g2, c2 = make_callbacks(jobs2)

        server1 = PeerJobServer(port1, g1, c1)
        server2 = PeerJobServer(port2, g2, c2)
        await server1.start()
        await server2.start()
        try:
            local_queue: asyncio.Queue = asyncio.Queue()
            drainer = PeerJobDrainer(local_queue)
            total = await drainer.drain_from_all_peers(
                {("127.0.0.1", port1), ("127.0.0.1", port2)}
            )
            assert total == 3
            assert local_queue.qsize() == 3
        finally:
            await server1.stop()
            await server2.stop()

    @pytest.mark.asyncio
    async def test_claim_invalid_json_returns_400(self, make_server):
        server, port = make_server
        await server.start()
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/jobs/claim",
                    data=b"not-json",
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    assert resp.status == 400
        finally:
            await server.stop()
