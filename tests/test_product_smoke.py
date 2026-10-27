"""Integration smoke test for the product skeleton (exercise 7): the API container answers
/health with a working database. Starts db + api with docker compose if they are not already
up, and stops them again afterwards. Skipped when docker is unavailable.

    pytest -m integration
"""

import json
import os
import shutil
import subprocess
import time
import urllib.request

import pytest

from zafra.paths import ROOT

PRODUCT = ROOT / "product"
API = f"http://localhost:{os.environ.get('API_PORT', '8000')}"

pytestmark = pytest.mark.integration


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=3) as r:
            return json.load(r)
    except OSError:
        return None


@pytest.fixture(scope="module")
def api_up():
    if not _docker_available():
        pytest.skip("docker is not available")
    started = False
    if _health() is None:
        compose = ["docker", "compose", "--project-directory", str(PRODUCT)]
        subprocess.run([*compose, "up", "-d", "--wait", "db", "api"], check=True, timeout=600)
        started = True
    yield
    if started:
        subprocess.run([*compose, "stop", "db", "api"], check=False, timeout=120)


def test_api_health(api_up):
    deadline = time.time() + 30
    health = _health()
    while health is None and time.time() < deadline:
        time.sleep(1)
        health = _health()
    assert health == {"status": "ok", "db": "ok"}
