import os
import json
from time import sleep
import requests

BASE_URL = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "Institution-Subfield-Finder (mailto:mihaimelnic@yahoo.com)"}
RESULTS_PER_PAGE = 200
MAX_RESULTS_PER_QUERY = 10000
OUTPUT_DIR = "domain_results_grouped"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAINS = {
    "1": ("AI", "Artificial Intelligence"),
    "2": ("ML", "Machine Learning"),
    "3": ("RL", "Reinforcement Learning"),
    "4": ("DL", "Deep Learning"),
    "5": ("CV", "Computer Vision")
}


def fetch_works(primary, subtopics):
    collected = []
    page = 1
    total = 0
    search_terms = [primary] + subtopics
    search_query = " OR ".join([f'"{term}"' for term in search_terms])
    filter_str = f'title_and_abstract.search:{search_query}'

    while total < MAX_RESULTS_PER_QUERY:
        params = {
            "filter": filter_str,
            "per-page": RESULTS_PER_PAGE,
            "page": page,
            "sort": "cited_by_count:desc"
        }

        response = requests.get(BASE_URL, params=params, headers=HEADERS)
        if response.status_code != 200:
            print(f"❌ Error fetching page {page} for {primary} + subfields: {response.status_code}")
            break

        data = response.json()
        works = data.get("results", [])
        if not works:
            break

        collected.extend(works)
        total += len(works)
        page += 1
        sleep(1)

    return collected[:MAX_RESULTS_PER_QUERY]


def save_works(primary_code, sub_codes, works):
    subfields_str = "_".join(sub_codes)
    filename = f"{primary_code}_with_{subfields_str}.json"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(works, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(works)} works to {filename}")


def main():
    print("\n=== Institution Search ===\n")
    print("Select a concept to search for papers:")
    for k, v in DOMAINS.items():
        print(f"{k}. {v[1]} ({v[0]})")
    print("q. Quit")

    choice = input("\nEnter your choice (1-5 or q): ").strip()
    if choice.lower() == 'q':
        print("Goodbye!")
        return
    if choice not in DOMAINS:
        print("❌ Invalid choice.")
        return

    primary_code, primary_name = DOMAINS[choice]
    subfields = [v[1] for k, v in DOMAINS.items() if k != choice]
    sub_codes = [v[0] for k, v in DOMAINS.items() if k != choice]

    print(f"\n🔍 Searching for: Primary = {primary_name}")
    print(f"    With any of subtopics: {', '.join(subfields)}")

    try:
        works = fetch_works(primary_name, subfields)
        save_works(primary_code, sub_codes, works)
    except Exception as e:
        print(f"⚠️ Failed for {primary_name} with subfields: {e}")

    print("\n🎉 Search completed.")


if __name__ == "__main__":
    main()