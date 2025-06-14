import json
import requests
from time import sleep
from collections import defaultdict

MAILTO = "mihaimelnic@yahoo.com"
OUTPUT_FILE = "top_cv_authors.json"
TOTAL_PAPERS = 10000
PAPERS_PER_PAGE = 200
CV_CONCEPT_ID = "C31972630"

def fetch_top_cv_papers():
    base_url = "https://api.openalex.org/works"
    params = {
        "filter": f"concepts.id:{CV_CONCEPT_ID}",
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
            if not response.get("meta", {}).get("next_cursor"):
                break
            params["cursor"] = response["meta"]["next_cursor"]
            sleep(0.5)
        except Exception as e:
            print(f"Error: {e}")
            sleep(2)
    
    return all_papers[:TOTAL_PAPERS]

def process_cv_authors(papers):
    author_stats = defaultdict(lambda: {
        "total_citations": 0,
        "all_papers": [],
        "raw_authorships": []
    })
    
    for paper in papers:
        cv_score = next(
            (c.get("score", 0) for c in paper.get("concepts", [])
             if c.get("id") == f"https://openalex.org/{CV_CONCEPT_ID}"), 0
        )
        if cv_score < 0.3:
            continue
        for authorship in paper.get("authorships", []):
            author = authorship.get("author", {})
            author_id = author.get("id")
            if author_id:
                paper_data = {
                    "id": paper.get("id"),
                    "title": paper.get("title"),
                    "year": paper.get("publication_year"),
                    "citations": paper.get("cited_by_count", 0),
                    "openalex_url": paper.get("id", "").replace("openalex.org/", "openalex.org/works/"),
                    "doi": paper.get("doi", ""),
                    "concepts": [c["display_name"] for c in paper.get("concepts", [])],
                    "is_cv": cv_score >= 0.5,
                    "authorships": paper.get("authorships", [])
                }
                author_stats[author_id]["name"] = author.get("display_name", "Unknown")
                author_stats[author_id]["total_citations"] += paper_data["citations"]
                author_stats[author_id]["all_papers"].append(paper_data)

    ranked_authors = []
    for author_id, data in author_stats.items():
        sorted_papers = sorted(data["all_papers"], key=lambda x: -x["year"])
        institution = "Unknown"
        for paper in sorted_papers:
            for inst in paper["authorships"]:
                if inst.get("author", {}).get("id") == author_id:
                    institutions = inst.get("institutions", [])
                    if institutions:
                        institution = institutions[0].get("display_name", "Unknown")
                    break
            if institution != "Unknown":
                break
        top_papers = sorted(
            data["all_papers"],
            key=lambda x: (-x["citations"], -x["is_cv"])
        )[:3]
        ranked_authors.append({
            "id": author_id,
            "name": data["name"],
            "institution": institution,
            "total_citations": data["total_citations"],
            "total_papers": len(data["all_papers"]),
            "cv_papers": sum(1 for p in data["all_papers"] if p["is_cv"]),
            "top_3_papers": top_papers
        })
    
    return sorted(ranked_authors, key=lambda x: (-x["total_citations"], -x["cv_papers"]))

def print_top_authors(authors, count=5):
    print("\n" + "="*60)
    print(f"TOP {count} COMPUTER VISION RESEARCHERS")
    print("="*60 + "\n")
    
    for rank, author in enumerate(authors[:count], 1):
        print(f"{rank}. {author['name']}")
        print(f"   • Institution: {author['institution']}")
        print(f"   • Total Citations: {author['total_citations']:,}")
        print(f"   • CV Papers: {author['cv_papers']} (of {author['total_papers']} total)")
        print("   • Top 3 Papers:")
        for paper in author["top_3_papers"]:
            cv_flag = "✓" if paper["is_cv"] else " "
            print(f"     {cv_flag} {paper['title']} ({paper['year']})")
            print(f"       - Citations: {paper['citations']:,}")
            print(f"       - DOI: {paper.get('doi', 'Not available')}")
            print(f"       - Concepts: {', '.join(paper['concepts'][:3])}")
        print("-"*60)

if __name__ == "__main__":
    print(f"Fetching top {TOTAL_PAPERS} Computer Vision papers...")
    cv_papers = fetch_top_cv_papers()
    
    print("\nProcessing authors and papers...")
    cv_authors = process_cv_authors(cv_papers)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_papers": len(cv_papers),
                "total_authors": len(cv_authors),
                "date_collected": "2025-06-06",
                "concept_used": "Computer Vision (C31972630)"
            },
            "authors": cv_authors
        }, f, indent=2, ensure_ascii=False)
    
    print_top_authors(cv_authors)
    
    print(f"\nFull results saved to '{OUTPUT_FILE}'")
