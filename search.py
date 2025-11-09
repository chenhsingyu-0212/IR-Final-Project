from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import time
import re


def is_english(text: str, threshold: float = 0.9) -> bool:
    """簡單判斷文字是否主要為英文。"""
    if not text:
        return False
    letters = re.findall(r"[A-Za-z]", text)
    ratio = len(letters) / max(len(text), 1)
    return ratio > threshold


def fetch_full_text(url: str, max_chars: int = 6000) -> str:
    """抓取網頁完整純文字內容（移除 HTML、JS、CSS）。"""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if not is_english(text[:800]):  # 只檢查前段是否為英文
            return ""
        return text[:max_chars]
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return ""


def search_and_retrieve_fulltext(query: str, num_results: int = 5):
    """
    使用 ddgs 搜尋英文內容，抓取前 num_results 筆結果的完整頁面內容。
    回傳 list of dict: [{"title": str, "url": str, "content": str}, ...]
    """
    results = []
    with DDGS() as ddgs:
        search_gen = ddgs.text(query, region="us-en", safesearch="off", max_results=num_results * 3)
        for result in search_gen:
            if len(results) >= num_results:
                break

            title = result.get("title")
            url = result.get("href") or result.get("url")
            if not url or not url.startswith("http"):
                continue

            print(f"🔎 Fetching: {title} ({url})")
            content = fetch_full_text(url)
            if content:
                results.append({
                    "title": title,
                    "url": url,
                    "content": content,
                })
                time.sleep(1.2)  # 防止封鎖

    return results


def combine_documents(docs):
    """把多篇文件合併成 prompt context 字串"""
    combined = ""
    for d in docs:
        combined += f"\n\n### {d['title']}\n{d['content']}\n"
    return combined


if __name__ == "__main__":
    query = "What are the main benefits of reinforcement learning in robotics?"
    docs = search_and_retrieve_fulltext(query)
    print(f"\n✅ Retrieved {len(docs)} documents.")
    for i, d in enumerate(docs, 1):
        print(f"\n--- Doc #{i} ---\nTitle: {d['title']}\nContent:\n{d['content']}...")
