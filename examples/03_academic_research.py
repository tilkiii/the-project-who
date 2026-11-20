"""
03_academic_research.py - Autonomous Academic Literature & Grounding
===================================================================
Demonstrates how the autonomous research engine queries peer-reviewed
databases (arXiv, PubMed, Semantic Scholar) to statistically ground behavioral markers.
"""

from who import AcademicResearcher


def main():
    print("=== Initializing Autonomous Academic Researcher ===")
    researcher = AcademicResearcher(target_name="Target X")

    query_topic = "Bayesian game theory mechanism design equilibrium"
    print(f"\nQuerying peer-reviewed literature for: '{query_topic}'...")

    results = researcher.research_topic(
        topic=query_topic,
        sources=["arxiv"],
        limit=3,
        ingest=False,
    )

    print(f"\nStatus: {results.get('status')}")
    print(f"Papers Found: {results.get('total_papers_found', 0)}")

    for i, paper in enumerate(results.get("papers", []), 1):
        print(f"\n[{i}] {paper.get('title')}")
        print(f"    Authors: {', '.join(paper.get('authors', [])[:3])}")
        print(f"    Source: {paper.get('source')} | URL: {paper.get('url')}")
        summary = paper.get('summary', '').strip().replace('\n', ' ')
        print(f"    Abstract: {summary[:200]}...")


if __name__ == "__main__":
    main()
