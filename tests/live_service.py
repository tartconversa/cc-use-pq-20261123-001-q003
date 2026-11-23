from __future__ import annotations

from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from threading import Thread
import time

import httpx
import uvicorn

from fieldproof.api import create_app
from tests.support import ApiCase, CrossoverCase, ROOT


def _wait_for_server(server: uvicorn.Server, thread: Thread) -> None:
    deadline = time.monotonic() + 8
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("FieldProof did not bind the loopback socket")


def run_parallel(client: httpx.Client) -> None:
    scenario = ApiCase()
    scenario.client = client
    state = scenario.setup_trial()
    cut = scenario.deliver("field-signals", scenario.green_events())["current_cut"]
    scenario.close(state["version"], cut)
    frozen = scenario.freeze(cut)
    evidence_id = frozen["evidence_id"]
    scenario.review(evidence_id, "field", "anna")
    scenario.review(evidence_id, "methods", "ben")
    bulletin = scenario.issue(evidence_id)
    assert bulletin["outcome"] == "advance"
    assert scenario.call("GET", ROOT + f"/evidence/{evidence_id}") == frozen
    print(f"[parallel] market bulletin {bulletin['edition']} from cut {cut}: "
          f"{bulletin['outcome']}")


def run_crossover(client: httpx.Client) -> None:
    scenario = CrossoverCase()
    scenario.client = client
    opened = scenario.setup_crossover()
    cut = scenario.cross_deliver("crossover-signals", scenario.balanced_events())["current_cut"]
    preview = scenario.cross_preview()
    market = preview["market_results"][0]
    assert market["recommendation"] == "advance", market["blocked_by"]
    closed = scenario.call("POST", "/trials/frost/close", {
        "expected_version": opened["version"], "expected_cut": cut,
    })
    assert closed["close_schedule_seq"] == preview["schedule_seq"]
    frozen = scenario.cross_freeze(cut)
    evidence_id = frozen["evidence_id"]
    scenario.cross_review(evidence_id, "field", "cleo")
    scenario.cross_review(evidence_id, "methods", "drew")
    bulletin = scenario.cross_issue(evidence_id, "advance")
    assert bulletin["outcome"] == "advance"
    assert bulletin["result"]["recommendation"] == "advance"
    print(f"[crossover] paired bulletin {bulletin['edition']} from cut {cut} and schedule "
          f"{closed['close_schedule_seq']}: {bulletin['outcome']}")


def main():
    with TemporaryDirectory() as directory, socket.socket(
        socket.AF_INET, socket.SOCK_STREAM
    ) as listener:
        listener.bind(("127.0.0.1", 0))
        address, port = listener.getsockname()
        app = create_app(Path(directory) / "pilot.sqlite3")
        config = uvicorn.Config(app, host=address, port=port, log_level="error",
                                lifespan="off", timeout_graceful_shutdown=2)
        server = uvicorn.Server(config)
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            _wait_for_server(server, thread)
            with httpx.Client(base_url=f"http://{address}:{port}", timeout=4,
                              trust_env=False) as client:
                assert client.get("/health").json() == {"status": "ready"}
                run_parallel(client)
                run_crossover(client)
        finally:
            server.should_exit = True
            thread.join(5)
            if thread.is_alive():
                raise RuntimeError("FieldProof did not stop after the acceptance flow")


if __name__ == "__main__":
    main()
