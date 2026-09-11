"""财联社 (cls.cn) telegraph news fetcher — Playwright + desktop Chrome UA.

财联社主页为纯 JS 渲染，requests 无法获取实质内容，须 Playwright + 桌面
Chrome UA，``goto(url, wait_until="domcontentloaded", timeout=30000)``。

The telegraph feed only exposes the current day's items. For a single fetch we
load the page, parse ``HH:MM:SS`` time markers, and return the latest items.
"""

import re
from dataclasses import dataclass

CHROMIUM_PATH = "/snap/bin/chromium"
TELEGRAPH_URL = "https://www.cls.cn/telegraph"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Noise lines that appear between real telegraph items.
_NOISE = ("桌面通知", "声音提醒", "语音电报", "电报持续更新中", "日期", "加红", "全部")
_TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2})")


@dataclass
class NewsItem:
    time: str  # HH:MM:SS
    content: str


def _get_browser():
    from playwright.sync_api import sync_playwright
    _pw = sync_playwright().start()
    browser = _pw.chromium.launch(
        headless=True, executable_path=CHROMIUM_PATH, args=["--no-sandbox", "--disable-dev-shm-usage"]
    )
    return _pw, browser


def fetch_telegraph(max_items: int = 50, timeout: int = 30) -> list[NewsItem]:
    """Fetch the latest telegraph items from 财联社.

    Returns a list of NewsItem (time + content), newest content first, noise
    filtered out.
    """
    pw, browser = _get_browser()
    try:
        page = browser.new_page(user_agent=UA)
        page.goto(TELEGRAPH_URL, wait_until="domcontentloaded", timeout=timeout * 1000)
        page.wait_for_timeout(2500)
        text = page.inner_text("body")
    finally:
        browser.close()
        pw.stop()

    parts = _TIME_RE.split(text)
    items: list[NewsItem] = []
    for i in range(1, len(parts), 2):
        time_str = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Strip trailing metadata line (labels / read count).
        content = re.split(r"\n(阅[\d.]+[WK]?|评论\(|分享\()", content)[0].strip()
        if not content or len(content) < 8:
            continue
        if any(content.startswith(n) or n == content for n in _NOISE):
            continue
        items.append(NewsItem(time=time_str, content=content))
        if len(items) >= max_items:
            break
    return items


def format_news(date_str: str, items: list[NewsItem]) -> str:
    """Render news items into the daily news txt format."""
    lines = [f"日期: {date_str}", "财经资讯（财联社电报）："]
    if not items:
        lines.append("（无资讯数据）")
        return "\n".join(lines)
    for it in items:
        lines.append(f"[{it.time}] {it.content}")
    return "\n".join(lines)
