import os
import json
from collections import Counter

def extract_and_save_institutions(concept, filename, save_dir):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    institution_counter = Counter()
    for author in data.get("authors", []):
        institution = author.get("institution", "Unknown")
        citations = author.get("total_citations", 0)
        if institution:
            institution_counter[institution] += citations

    sorted_institutions = institution_counter.most_common()

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{concept.lower().replace(' ', '_')}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_institutions, f, indent=2)

    print(f"Saved institutions for '{concept}' to {output_path}")
    return sorted_institutions


def extract_and_save_papers(concept, filename, save_dir):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    paper_dict = {}
    for author in data.get("authors", []):
        for paper in author.get("top_3_papers", []):
            paper_id = paper.get("id")
            if paper_id not in paper_dict:
                paper_dict[paper_id] = {
                    "id": paper_id,
                    "title": paper.get("title"),
                    "citations": paper.get("citations", 0),
                    "openalex_url": paper.get("openalex_url"),
                    "doi": paper.get("doi"),
                    "year": paper.get("year"),
                    "abstract": paper.get("abstract"),
                }

    deduplicated_papers = sorted(paper_dict.values(), key=lambda x: x["citations"], reverse=True)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{concept.lower().replace(' ', '_')}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduplicated_papers, f, indent=2)

    print(f"Saved papers for '{concept}' to {output_path}")
    return deduplicated_papers


if __name__ == "__main__":
    concept_map = {
        "artificial intelligence": "searching_codes/top_authors_concept/top_ai_authors_with_papers.json",
        "machine learning": "searching_codes/top_authors_concept/top_ml_authors.json",
        "deep learning": "searching_codes/top_authors_concept/top_dl_authors_with_papers.json",
        "reinforcement learning": "searching_codes/top_authors_concept/top_rl_authors_with_papers.json",
        "computer vision": "searching_codes/top_authors_concept/top_cv_authors.json",
    }

    institutions_dir = "searching_codes/institutions_by_domain"
    papers_dir = "searching_codes/papers_by_concept"

    for concept, file_path in concept_map.items():
        extract_and_save_institutions(concept, file_path, institutions_dir)
        extract_and_save_papers(concept, file_path, papers_dir)