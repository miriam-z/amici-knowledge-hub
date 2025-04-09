import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os
import yaml

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Load sources from YAML
with open("sources.yaml", "r") as f:
    sources = yaml.safe_load(f)

if not sources:
    raise ValueError("sources.yaml is empty or invalid!")


# Clean text utility
def clean_text(text):
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


# Smart chunking utility
def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        length_function=len,
    )
    return splitter.split_text(text)


# Q&A extraction for FAQ type sources
def extract_qna(url, output_name):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    def is_valid_question(text):
        text = text.strip()
        if re.match(r"^\d+[\.\)]", text):
            return False
        starters = [
            "What",
            "How",
            "Why",
            "Can",
            "Is",
            "Does",
            "Do",
            "When",
            "Should",
            "Which",
            "Who",
        ]
        return (
            text.endswith("?")
            or any(text.startswith(qw) for qw in starters)
            or "how to" in text.lower()
        )

    questions, answers = [], []
    question_tags = soup.find_all(["h2", "h3"])

    for tag in question_tags:
        potential_question = tag.get_text(" ", strip=True)
        if not is_valid_question(potential_question):
            continue

        answer_tag = tag.find_next_sibling()
        answer_text = ""

        while answer_tag and answer_tag.name not in ["h2", "h3"]:
            if answer_tag.name in ["p", "li"]:
                text = clean_text(answer_tag.get_text(" ", strip=True))
                if len(text) > 20:
                    answer_text += text + "\n"
            elif answer_tag.name in ["ul", "ol"]:
                for li in answer_tag.find_all("li"):
                    text = clean_text(li.get_text(" ", strip=True))
                    if text:
                        answer_text += f"- {text}\n"
            answer_tag = answer_tag.find_next_sibling()

        answer_text = answer_text.strip()

        if answer_text:
            questions.append(potential_question)
            answers.append(answer_text)

    if questions:
        df = pd.DataFrame({"question": questions, "answer": answers, "source": url})
        df.to_csv(f"data/{output_name}.csv", index=False)
        print(f"Extracted {len(df)} Q&A pairs from {url}")
    else:
        print(f"No Q&A pairs found at {url}")


# General page processor with improved extraction
def process_page(url, output_name, source_type):
    print(f"Processing: {url} ({source_type})")

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.find("h1", {"id": "firstHeading"}) or soup.find("title")
    title = title_tag.text.strip() if title_tag else output_name

    if "gov.uk" in url:
        main_content = soup.find("main")
    elif "wikipedia.org" in url:
        main_content = soup.find("div", {"class": "mw-parser-output"})
    else:
        main_content = soup.find("body") or soup

    if not main_content:
        print(f"No main content found at {url}")
        return

    skip_sections = [
        "references",
        "citation",
        "external links",
        "bibliography",
        "see also",
        "further reading",
    ]
    collecting = True
    current_section = ""
    content_blocks = []

    for tag in main_content.find_all(["h2", "h3", "h4", "p", "li"]):
        if tag.name in ["h2", "h3", "h4"]:
            header_text = tag.get_text(" ", strip=True).lower()
            if any(skip in header_text for skip in skip_sections):
                collecting = False
                continue
            current_section = tag.get_text(" ", strip=True)

        if collecting and tag.name in ["p", "li"]:
            text = clean_text(tag.get_text(" ", strip=True))
            if len(text) > 30:
                if current_section:
                    text = f"{current_section}\n{text}"
                content_blocks.append(text)

    if not content_blocks:
        print(f"No meaningful content at {url}")
        return

    full_text = "\n\n".join(content_blocks)
    chunks = chunk_text(full_text)

    all_data = [
        {
            "source": url,
            "title": title,
            "chunk_id": idx,
            "source_type": source_type,
            "content": chunk,
        }
        for idx, chunk in enumerate(chunks)
    ]

    df = pd.DataFrame(all_data)
    df.to_csv(f"data/{output_name}_chunks.csv", index=False)
    print(f"Exported {len(df)} chunks to data/{output_name}_chunks.csv")


# Main pipeline
if __name__ == "__main__":
    print("Starting data preparation pipeline...")

    for source in tqdm(sources, desc="Processing sources"):
        url, source_type, output_name = source["url"], source["type"], source["name"]

        print(f"Processing source: {output_name} ({source_type}) - {url}")

        try:
            if source_type == "faq":
                print(f"Extracting Q&A from {url}...")
                extract_qna(url, output_name)
                print(f"Q&A extraction complete. Now chunking page...")
                process_page(url, output_name, source_type)
            else:
                print(f"Chunking page: {url}")
                process_page(url, output_name, source_type)

            print(f"Finished processing {output_name}")

        except Exception as e:
            print(f"Error processing {url}: {e}")

    print("All sources processed. Data is ready in /data folder.")
