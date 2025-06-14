import requests
import json
from collections import defaultdict
from time import sleep

# Configuration
MAILTO = "mihaimelnic@yahoo.com"
OUTPUT_FILE = "top_ml_authors.json"
TOTAL_PAPERS = 10000
PAPERS_PER_PAGE = 200
ML_CONCEPT_ID = "C119857082"  # Machine Learning concept ID in OpenAlex

def fetch_top_ml_papers():
    """Fetch top 10,000 most cited ML papers"""
    base_url = "https://api.openalex.org/works"
    params = {
        "filter": f"concepts.id:{ML_CONCEPT_ID}",
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

            print(f"Fetched {len(all_papers)}/{TOTAL_PAPERS} ML papers")

            if len(all_papers) >= TOTAL_PAPERS or not response.get("meta", {}).get("next_cursor"):
                break

            params["cursor"] = response["meta"]["next_cursor"]
            sleep(0.5)

        except Exception as e:
            print(f"Error: {e}. Retrying...")
            sleep(2)

    return all_papers[:TOTAL_PAPERS]

def process_ml_authors(papers):
    """Extract authors with their top 3 ML papers and affiliations"""
    author_stats = defaultdict(lambda: {
        "total_citations": 0,
        "paper_count": 0,
        "papers": [],
        "institution": "Unknown",
        "country": "Unknown"
    })

    for paper in papers:
        ml_concept = next(
            (c for c in paper.get("concepts", [])
             if c.get("id") == f"https://openalex.org/{ML_CONCEPT_ID}"
             and c.get("score", 0) > 0.3),
            None
        )
        if not ml_concept:
            continue

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
                "openalex_url": paper.get("id", "").replace("openalex.org/", "openalex.org/works/"),
                "doi": paper.get("doi", ""),
                "concepts": [c["display_name"] for c in paper.get("concepts", [])[:3]]
            }

            author_stats[author_id]["name"] = author.get("display_name", "Unknown")
            author_stats[author_id]["total_citations"] += paper_data["citations"]
            author_stats[author_id]["paper_count"] += 1
            author_stats[author_id]["papers"].append(paper_data)

            # Set institution and country if not already set
            if author_stats[author_id]["institution"] == "Unknown":
                institutions = authorship.get("institutions", [])
                if institutions:
                    inst = institutions[0]
                    author_stats[author_id]["institution"] = inst.get("display_name", "Unknown")
                    author_stats[author_id]["country"] = inst.get("country_code", "Unknown")

    # Compile and sort
    ranked_authors = []
    for author_id, data in author_stats.items():
        top_papers = sorted(data["papers"], key=lambda x: -x["citations"])[:3]

        ranked_authors.append({
            "id": author_id,
            "name": data["name"],
            "institution": data["institution"],
            "country": data["country"],
            "total_citations": data["total_citations"],
            "paper_count": data["paper_count"],
            "top_3_papers": top_papers
        })

    return sorted(ranked_authors, key=lambda x: -x["total_citations"])

if __name__ == "__main__":
    print(f"Fetching top {TOTAL_PAPERS} ML papers by citations...")
    ml_papers = fetch_top_ml_papers()

    print("Processing ML authors...")
    ml_authors = process_ml_authors(ml_papers)

    output = {
        "metadata": {
            "total_papers": len(ml_papers),
            "total_authors": len(ml_authors),
            "date_collected": "2025-06-06",
            "search_parameters": {
                "concept": "Machine Learning (C119857082)",
                "sort": "cited_by_count:desc"
            }
        },
        "authors": ml_authors
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_FILE}")
    print("Top 5 ML researchers by aggregate citations:")
    for author in ml_authors[:5]:
        print(f"- {author['name']} ({author['institution']}, {author['country']}): {author['total_citations']:,} citations")
