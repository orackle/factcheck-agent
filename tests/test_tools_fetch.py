from __future__ import annotations

import httpx
import pytest

from factcheck.tools.fetch import fetch_doc

HTML_OK = """
<html><head><title>Example Article</title></head>
<body>
<script>var x = 1;</script>
<nav>site nav junk that should be stripped out entirely here</nav>
<p>This is the first real paragraph and it is definitely long enough.</p>
<p>This is the second real paragraph, also long enough to be kept intact.</p>
</body></html>
"""

HTML_EMPTY = "<html><head><title>Empty</title></head><body><p>short</p></body></html>"


def _mock_transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_doc_extracts_paragraphs(monkeypatch):
    def handler(request):
        return httpx.Response(200, text=HTML_OK, headers={"content-type": "text/html"})

    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Client(transport=httpx.MockTransport(handler)).get(url))

    doc = fetch_doc("https://example.com/article", timeout_seconds=5, max_chars=6000)

    assert doc.ok
    assert doc.title == "Example Article"
    assert "first real paragraph" in doc.text
    assert "site nav junk" not in doc.text


def test_fetch_doc_reports_error_on_http_failure(monkeypatch):
    def handler(request):
        return httpx.Response(404, text="not found")

    def fake_get(url, **kw):
        response = httpx.Client(transport=httpx.MockTransport(handler)).get(url)
        response.raise_for_status()
        return response

    monkeypatch.setattr(httpx, "get", fake_get)

    doc = fetch_doc("https://example.com/missing", timeout_seconds=5, max_chars=6000)

    assert not doc.ok
    assert doc.error is not None


def test_fetch_doc_rejects_non_html_content_type(monkeypatch):
    def handler(request):
        return httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})

    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Client(transport=httpx.MockTransport(handler)).get(url))

    doc = fetch_doc("https://example.com/file.pdf", timeout_seconds=5, max_chars=6000)

    assert not doc.ok
    assert "content-type" in doc.error


def test_fetch_doc_flags_pages_with_no_real_body_text(monkeypatch):
    def handler(request):
        return httpx.Response(200, text=HTML_EMPTY, headers={"content-type": "text/html"})

    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Client(transport=httpx.MockTransport(handler)).get(url))

    doc = fetch_doc("https://example.com/thin", timeout_seconds=5, max_chars=6000)

    assert not doc.ok
    assert doc.error == "no extractable body text"


def test_fetch_doc_truncates_to_max_chars(monkeypatch):
    long_paragraph = "This sentence repeats to build a very long paragraph. " * 200
    html = f"<html><head><title>Long</title></head><body><p>{long_paragraph}</p></body></html>"

    def handler(request):
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Client(transport=httpx.MockTransport(handler)).get(url))

    doc = fetch_doc("https://example.com/long", timeout_seconds=5, max_chars=100)

    assert doc.ok
    assert len(doc.text) <= 100
