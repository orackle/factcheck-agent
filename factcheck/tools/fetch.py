"""Page fetching + text extraction.

Kept deliberately simple (httpx + BeautifulSoup, strip boilerplate tags,
join paragraph text) rather than pulling in a heavier extraction library.
Good enough for news/blog/doc pages, which is what fact-checking sources
mostly are; PDFs and JS-rendered pages are out of scope for v1 and are
reported back as a `FetchedDoc.error` rather than silently returning junk.
"""
from __future__ import annotations

import httpx

from factcheck.schemas import FetchedDoc

_USER_AGENT = (
    "Mozilla/5.0 (compatible; FactCheckAgent/0.1; "
    "+https://github.com/orackle/factcheck-agent)"
)

_BLOCK_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript", "form")


def fetch_doc(url: str, timeout_seconds: float, max_chars: int) -> FetchedDoc:
    try:
        response = httpx.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        return FetchedDoc(url=url, error=f"fetch failed: {e}")

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return FetchedDoc(url=url, error=f"unsupported content-type: {content_type or 'unknown'}")

    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        return FetchedDoc(url=url, error=f"beautifulsoup4 is not installed: {e}")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag_name in _BLOCK_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li"])]
    text = "\n".join(p for p in paragraphs if len(p) > 40)

    if not text.strip():
        return FetchedDoc(url=url, title=title, error="no extractable body text")

    return FetchedDoc(url=url, title=title, text=text[:max_chars])
