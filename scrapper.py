import os
import re
import time
import requests
import datetime
import urllib.parse
import concurrent.futures
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from vector_store import setup_chromadb, data_load

load_dotenv()


def process_image(img_url, article_url, article_id, images_dir):
    if not img_url:
        return None

    if not img_url.startswith(("http://", "https://")):
        img_url = urllib.parse.urljoin(article_url, img_url)

    img_lower = img_url.lower()
    if not (
        img_lower.endswith(".png")
        or img_lower.endswith(".jpg")
        or img_lower.endswith(".jpeg")
    ):
        return None

    img_filename = re.sub(r"[^\w\-_.]", "_", f"{os.path.basename(img_url)}")
    issue_images_dir = os.path.join(images_dir, article_id)
    os.makedirs(issue_images_dir, exist_ok=True)
    img_filepath = os.path.join(issue_images_dir, img_filename)

    try:
        img_response = requests.get(img_url, timeout=10)
        img_response.raise_for_status()

        with open(img_filepath, "wb") as img_file:
            img_file.write(img_response.content)

        # caption = caption_image(img_filepath)

        return {
            "url": img_url,
            "filename": img_filename,
            "path": os.path.abspath(img_filepath),
            "caption": "",
        }
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        return None


def process_url(url, selector, images_dir):
    try:
        response = requests.get(url, timeout=25)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        article_container = soup.select_one(selector)

        if not article_container:
            return {
                "url": url,
                "status": "no_matches",
                "chapters": [],
            }

        article_id = url.split("/")[-2]

        published_date_element = article_container.select_one(
            "div.mt-1.text-slate-600.text-base.text-sm"
        )
        published_date = (
            published_date_element.get_text(strip=True)
            if published_date_element
            else ""
        )
        try:
            published_date_iso = datetime.datetime.strptime(
                published_date.strip(), "%b %d, %Y"
            ).strftime("%Y-%m-%d")
        except ValueError:
            published_date_iso = ""

        news_tag = article_container.find(["h2", "h1"], id="news")

        if news_tag:
            post_news_elements = []
            current = news_tag.next_sibling
            while current:
                post_news_elements.append(str(current))
                current = current.next_sibling
            html_for_splitting = "".join(post_news_elements)
        else:
            html_for_splitting = str(article_container)

        blocks = re.split(r"<hr\s*/?>", html_for_splitting, flags=re.IGNORECASE)

        chapters = []
        for i, block_html in enumerate(blocks):
            block_soup = BeautifulSoup(block_html, "html.parser")

            header_tag = block_soup.find(["h1", "h2", "strong"])
            title = header_tag.get_text(strip=True) if header_tag else f"Section {i+1}"

            if "deeplearning.ai" in title.lower() or "news" in title.lower():
                continue

            paragraphs = block_soup.find_all("p")
            section_text = " ".join(
                [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
            )

            images = []
            for img in block_soup.find_all("img"):
                img_url = img.get("src")
                if img_url:
                    processed_image = process_image(
                        img_url, url, article_id, images_dir
                    )
                    if processed_image:
                        images.append(processed_image)

            if not section_text and not images:
                continue

            chapters.append(
                {
                    "title": title,
                    "content": section_text,
                    "images": images,
                    "article_id": article_id,
                    "url": url,
                    "published_date": published_date_iso,
                }
            )

        return {
            "url": url,
            "status": "success",
            "article_id": article_id,
            "published_date": published_date_iso,
            "chapters": chapters,
        }

    except Exception as e:
        return {"url": url, "status": "error", "error": str(e), "chapters": []}


def scrape_with_selector_parallel(
    urls,
    selector="#content > article > div > div",
    images_dir="data/images",
    max_workers=10,
):
    os.makedirs(images_dir, exist_ok=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(process_url, url, selector, images_dir): url for url in urls
        }

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                if result["status"] == "success":
                    print(f"Scraped: {url} -> {len(result['chapters'])} chapters")
                elif result["status"] == "no_matches":
                    print(f"No matching elements found for: {url}")
                else:
                    print(
                        f"Error scraping {url}: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                print(f"Error processing {url}: {e}")
                results.append(
                    {"url": url, "status": "error", "error": str(e), "chapters": []}
                )

    return results


if __name__ == "__main__":
    vector_store, chroma_client, chroma_collection = setup_chromadb("data/chroma_db")

    if chroma_collection.count() == 0:
        print("Database is empty. Starting initial data scraping...")

        urls_arabic = [
            f"https://www.deeplearning.ai/the-batch/issue-{i}/" for i in range(200, 302)
        ]

        urls_roman = [
            "https://www.deeplearning.ai/the-batch/issue-i/",
            "https://www.deeplearning.ai/the-batch/issue-ii/",
            "https://www.deeplearning.ai/the-batch/issue-iii/",
            "https://www.deeplearning.ai/the-batch/issue-iv/",
            "https://www.deeplearning.ai/the-batch/issue-v/",
            "https://www.deeplearning.ai/the-batch/issue-vi/",
            "https://www.deeplearning.ai/the-batch/issue-vii/",
            "https://www.deeplearning.ai/the-batch/issue-viii/",
            "https://www.deeplearning.ai/the-batch/issue-ix/",
            "https://www.deeplearning.ai/the-batch/issue-x/",
            "https://www.deeplearning.ai/the-batch/issue-xi/",
            "https://www.deeplearning.ai/the-batch/issue-xii/",
            "https://www.deeplearning.ai/the-batch/issue-xiii/",
            "https://www.deeplearning.ai/the-batch/issue-xiv/",
            "https://www.deeplearning.ai/the-batch/issue-xv/",
            "https://www.deeplearning.ai/the-batch/issue-xvi/",
        ]

        urls_to_scrape = urls_arabic + urls_roman
        css_selector = "#content > article "

        print("Starting parallel scraping...")
        results = scrape_with_selector_parallel(
            urls_to_scrape, css_selector, "data/images", max_workers=10
        )

        successful = sum(1 for r in results if r["status"] == "success")
        print(f"Successfully scraped {successful} out of {len(urls_to_scrape)} URLs")

        print("Loading data into ChromaDB...")
        index = data_load(results, vector_store)
        print("Data loading completed!")
    else:
        print(f"Database already contains {chroma_collection.count()} documents.")
        print("Skipping scraping process.")

    print("\nVerifying database content...")
    try:
        query_engine = index.as_query_engine(similarity_top_k=2)
        response = query_engine.query("What is the latest AI news you have?")
        print(f"Database is operational with {chroma_collection.count()} documents")
    except Exception as e:
        print(f"Error verifying database: {str(e)}")
