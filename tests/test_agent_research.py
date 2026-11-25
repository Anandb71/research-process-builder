"""Offline regression coverage for the research agent's Firecrawl REST client."""

import importlib.util
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import pytest
import requests


URL = "https://example.test/company"
CONTENT = "Company research evidence. " * 20


def response(payload, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode("utf-8")
    return result


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    path = Path(__file__).resolve().parents[1] / "scripts" / "agent_research.py"
    spec = importlib.util.spec_from_file_location("agent_research_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    post = Mock(return_value=response({"success": True, "data": {"markdown": CONTENT}}))
    monkeypatch.setattr(module.requests, "post", post)
    return module, post


@pytest.mark.parametrize("timeout_ms", [None, 60_000])
def test_v2_request_preserves_fresh_content_and_scrape_deadline(agent, timeout_ms):
    module, post = agent
    options = {} if timeout_ms is None else {"timeout_ms": timeout_ms}

    assert module.firecrawl_scrape(URL, **options) == CONTENT

    budget_ms = 30_000 if timeout_ms is None else timeout_ms
    post.assert_called_once_with(
        "https://api.firecrawl.dev/v2/scrape",
        headers={
            "Authorization": "Bearer test-firecrawl-key",
            "Content-Type": "application/json",
        },
        json={
            "url": URL,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "maxAge": 0,
            "timeout": budget_ms,
        },
        timeout=budget_ms / 1000 + 5,
    )


@pytest.mark.parametrize("length", [0, 200, 201, 12_000, 12_001])
def test_content_threshold_and_truncation_are_preserved(agent, length):
    module, post = agent
    content = "x" * length
    post.return_value = response({"success": True, "data": {"markdown": content}})

    assert module.firecrawl_scrape(URL) == (content[:12_000] if length > 200 else None)


@pytest.mark.parametrize("payload", [
    {"success": False, "error": "Scrape failed"},
    {"success": False, "data": {"markdown": CONTENT}},
    {"success": True},
    {"success": True, "data": {}},
    {"success": True, "data": None},
    {"success": True, "data": {"markdown": None}},
])
def test_failed_or_missing_markdown_returns_none(agent, payload):
    module, post = agent
    post.return_value = response(payload)

    assert module.firecrawl_scrape(URL) is None


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_do_not_return_content_or_retry_paid_requests(agent, status):
    module, post = agent
    post.return_value = response({"success": True, "data": {"markdown": CONTENT}}, status)

    assert module.firecrawl_scrape(URL) is None
    post.assert_called_once()


@pytest.mark.parametrize("error", [requests.Timeout, requests.ConnectionError])
def test_transport_failures_return_none_without_retry(agent, error):
    module, post = agent
    post.side_effect = error("unavailable")

    assert module.firecrawl_scrape(URL) is None
    post.assert_called_once()


def test_invalid_json_returns_none(agent):
    module, post = agent
    post.return_value._content = b"not json"

    assert module.firecrawl_scrape(URL) is None


def test_missing_key_skips_the_request(agent, monkeypatch):
    module, post = agent
    monkeypatch.setattr(module, "FIRECRAWL_KEY", "")

    assert module.firecrawl_scrape(URL) is None
    post.assert_not_called()


def test_scrape_over_loopback_http(agent, monkeypatch):
    module, _ = agent
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers["Authorization"], json.loads(body)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": True,
                "data": {"markdown": CONTENT},
            }).encode("utf-8"))

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server, requests.Session() as session:
        session.trust_env = False
        monkeypatch.setattr(module, "FC_BASE", f"http://127.0.0.1:{server.server_port}/v2")
        monkeypatch.setattr(module.requests, "post", session.post)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            assert module.firecrawl_scrape(URL) == CONTENT
        finally:
            server.shutdown()
            thread.join(timeout=2)

    assert received == [("/v2/scrape", "Bearer test-firecrawl-key", {
        "url": URL,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "maxAge": 0,
        "timeout": 30_000,
    })]
