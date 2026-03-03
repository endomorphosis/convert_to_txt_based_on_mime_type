"""
Peer Job Drainer – pulls pending job entries from a remote peer's HTTP
job-queue endpoint and feeds them into the local processing queue.

Protocol
--------
Each node exposes two HTTP routes on ``lan_peer_port``:

  GET /jobs
      Returns a JSON object ``{"jobs": [<job_id>, ...]}``.  Each entry is a
      string identifier (e.g. a file path or CID) for a pending job.

  POST /jobs/claim
      Body: ``{"jobs": [<job_id>, ...]}``
      The peer atomically removes those job IDs from its own queue and
      responds with the subset it could actually claim:
      ``{"claimed": [<job_id>, ...]}``

The ``PeerJobServer`` class in this module provides the server-side
implementation; ``PeerJobDrainer`` provides the client-side draining logic.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, List, Set, Tuple

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server side – expose this node's job queue over HTTP
# ---------------------------------------------------------------------------

class PeerJobServer:
    """
    Lightweight aiohttp HTTP server that exposes the local job queue so that
    peer nodes can discover and drain pending jobs from it.

    Args:
        job_port: TCP port to listen on.
        get_pending_jobs: Callable that returns the list of pending job IDs
            (e.g. file path strings) currently in the local queue.
        claim_jobs: Callable that removes a set of job IDs from the local
            queue and returns the subset that was successfully removed.
    """

    def __init__(
        self,
        job_port: int,
        get_pending_jobs: Callable[[], List[str]],
        claim_jobs: Callable[[List[str]], List[str]],
    ) -> None:
        self._port = job_port
        self._get_pending_jobs = get_pending_jobs
        self._claim_jobs = claim_jobs
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start the HTTP server in the background."""
        app = web.Application()
        app.router.add_get("/jobs", self._handle_get_jobs)
        app.router.add_post("/jobs/claim", self._handle_claim_jobs)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await site.start()
        logger.info("PeerJobServer listening on port %d", self._port)

    async def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        logger.info("PeerJobServer stopped")

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------

    async def _handle_get_jobs(self, request: web.Request) -> web.Response:
        jobs = self._get_pending_jobs()
        return web.json_response({"jobs": jobs})

    async def _handle_claim_jobs(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            requested: List[str] = body.get("jobs", [])
        except (json.JSONDecodeError, ValueError):
            return web.Response(status=400, text="Invalid JSON body")

        claimed = self._claim_jobs(requested)
        return web.json_response({"claimed": claimed})


# ---------------------------------------------------------------------------
# Client side – drain jobs from a remote peer
# ---------------------------------------------------------------------------

class PeerJobDrainer:
    """
    Queries a remote peer's HTTP job-queue endpoint and transfers pending
    jobs to the local queue.

    Args:
        local_queue: An ``asyncio.Queue`` instance to which drained job IDs
            will be added so the local processor can pick them up.
        request_timeout: Seconds to wait for each HTTP request to a peer.
    """

    def __init__(
        self,
        local_queue: asyncio.Queue,
        request_timeout: float = 5.0,
    ) -> None:
        self._local_queue = local_queue
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)

    async def drain_from_peer(self, host: str, job_port: int) -> int:
        """
        Attempt to drain all available jobs from a single peer.

        Returns the number of jobs successfully transferred to the local
        queue, or 0 if the peer could not be reached or had no jobs.

        Args:
            host: IP address or hostname of the peer.
            job_port: TCP port of the peer's HTTP job-queue server.
        """
        base_url = f"http://{host}:{job_port}"
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                # 1. Ask the peer what jobs it has.
                available = await self._fetch_available_jobs(session, base_url)
                if not available:
                    logger.debug("Peer %s:%d has no jobs available", host, job_port)
                    return 0

                # 2. Claim as many as possible.
                claimed = await self._claim_jobs(session, base_url, available)
                if not claimed:
                    logger.debug("Could not claim any jobs from peer %s:%d", host, job_port)
                    return 0

                # 3. Feed claimed jobs into the local queue.
                for job_id in claimed:
                    await self._local_queue.put(job_id)

                logger.info(
                    "Drained %d job(s) from peer %s:%d",
                    len(claimed), host, job_port,
                )
                return len(claimed)

        except aiohttp.ClientError as exc:
            logger.warning("Could not connect to peer %s:%d – %s", host, job_port, exc)
            return 0
        except asyncio.TimeoutError:
            logger.warning("Timeout connecting to peer %s:%d", host, job_port)
            return 0

    async def drain_from_all_peers(
        self, peers: Set[Tuple[str, int]]
    ) -> int:
        """
        Drain jobs from all given peers concurrently.

        Args:
            peers: Set of ``(host, job_port)`` tuples as returned by
                ``PeerDiscovery.get_peers()``.

        Returns:
            Total number of jobs drained across all peers.
        """
        if not peers:
            return 0

        results = await asyncio.gather(
            *[self.drain_from_peer(host, port) for host, port in peers],
            return_exceptions=True,
        )
        total = 0
        for result in results:
            if isinstance(result, int):
                total += result
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_available_jobs(
        self, session: aiohttp.ClientSession, base_url: str
    ) -> List[str]:
        async with session.get(f"{base_url}/jobs") as resp:
            if resp.status != 200:
                logger.warning(
                    "GET %s/jobs returned status %d", base_url, resp.status
                )
                return []
            data = await resp.json()
            return data.get("jobs", [])

    async def _claim_jobs(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        job_ids: List[str],
    ) -> List[str]:
        async with session.post(
            f"{base_url}/jobs/claim",
            json={"jobs": job_ids},
        ) as resp:
            if resp.status != 200:
                logger.warning(
                    "POST %s/jobs/claim returned status %d", base_url, resp.status
                )
                return []
            data = await resp.json()
            return data.get("claimed", [])
