import json
import requests
from time import sleep
from collections import defaultdict

MAILTO = "mihaimelnic@yahoo.com"
OUTPUT_FILE = "top_dl_authors_with_papers.json"
TOTAL_PAPERS = 10000
PAPERS_PER_PAGE = 200
SEARCH_TERM = "deep learning"

def fetch_top_dl_papers():
    """Fetch top cited DL papers using title/abstract search"""
    base_url = "https://api.openalex.org/works"
    params = {
        "filter": f"title_and_abstract.search:{SEARCH_TERM}",
        "sort": "cited_by_count:desc",
        "per-page": PAPERS_PER_PAGE,
        "mailto": MAILTO,
        "cursor": "*"
    }
    all_papers = []

    while len(all_papers) < TOTAL_PAPERS:
        try:
            response = requests.get(base_url, params=params).json()
            papers = response.get("results", [])
            all_papers.extend(papers)
            print(f"Fetched {len(all_papers)}/{TOTAL_PAPERS} papers")

            if not response.get("meta", {}).get("next_cursor"):
                break
            params["cursor"] = response["meta"]["next_cursor"]
            sleep(0.5)
        except Exception as e:
            print(f"Error: {e}")
            sleep(2)

    return all_papers[:TOTAL_PAPERS]

def process_authors(papers):
    """Process authors with their top 3 papers and institution/country"""
    author_stats = defaultdict(lambda: {
        "total_citations": 0,
        "all_papers": [],
        "institution": "Unknown",
        "country": "Unknown"
    })

    for paper in papers:
        for authorship in paper.get("authorships", []):
            author = authorship.get("author", {})
            author_id = author.get("id")
            if not author_id:
                continue

            paper_data = {
                "id": paper.get("id"),
                "title": paper.get("title"),
                "year": paper.get("publication_year"),
                "citations": paper.get("cited_by_count", 0),
                "openalex_url": paper.get("id", "").replace("https://openalex.org/", "https://openalex.org/works/"),
                "doi": paper.get("doi", ""),
                "abstract": paper.get("abstract", "")
            }

            author_stats[author_id]["name"] = author.get("display_name", "Unknown")
            author_stats[author_id]["total_citations"] += paper_data["citations"]
            author_stats[author_id]["all_papers"].append(paper_data)

            if author_stats[author_id]["institution"] == "Unknown":
                institutions = authorship.get("institutions", [])
                if institutions:
                    inst = institutions[0]
                    author_stats[author_id]["institution"] = inst.get("display_name", "Unknown")
                    author_stats[author_id]["country"] = inst.get("country_code", "Unknown")

    ranked_authors = []
    for author_id, data in author_stats.items():
        top_papers = sorted(
            data["all_papers"],
            key=lambda x: -x["citations"]
        )[:3]

        ranked_authors.append({
            "id": author_id,
            "name": data["name"],
            "institution": data["institution"],
            "country": data["country"],
            "total_citations": data["total_citations"],
            "paper_count": len(data["all_papers"]),
            "top_3_papers": top_papers
        })

    return sorted(ranked_authors, key=lambda x: -x["total_citations"])

def print_top_authors(authors, count=10):
    """Print top authors summary"""
    print("\n" + "="*80)
    print(f"TOP {count} DEEP LEARNING RESEARCHERS")
    print("="*80)
    print(f"Search Criteria: Title/abstract contains '{SEARCH_TERM}'\n")

    for rank, author in enumerate(authors[:count], 1):
        print(f"{rank}. {author['name']}")
        print(f"   • Institution: {author['institution']} ({author['country']})")
        print(f"   • Total Citations: {author['total_citations']:,}")
        print(f"   • Papers in Dataset: {author['paper_count']}")
        print(f"   • Top 3 Papers:")
        for paper in author["top_3_papers"]:
            print(f"      - {paper['title']} ({paper['year']})")
            print(f"        Citations: {paper['citations']:,}")
            print(f"        DOI: {paper.get('doi', 'Not available')}")
        print("-"*60)

if __name__ == "__main__":
    print(f"Fetching top {SEARCH_TERM} papers...")
    papers = fetch_top_dl_papers()

    print("\nProcessing authors and papers...")
    authors = process_authors(papers)

    output = {
        "metadata": {
            "total_papers": len(papers),
            "total_authors": len(authors),
            "date_collected": "2025-06-06",
            "search_parameters": {
                "text_search": f"{SEARCH_TERM} in title/abstract",
                "sort": "cited_by_count:desc"
            }
        },
        "authors": authors
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print_top_authors(authors)

    print(f"\nSaved {len(authors)} authors with top papers to {OUTPUT_FILE}")