# Autonomous Academic Literature & Grounding

The `who_learn` autonomous researcher grounds behavioral and psychometric hypotheses against peer-reviewed scientific literature.

## Integrated Repositories

- **arXiv API:** Quantitative finance, game theory, mathematical economics, and computer science.
- **PubMed (E-Utilities):** Cognitive psychology, neurobiology, and chronomics.
- **Semantic Scholar API:** Citation graphs, influence scores, and paper abstracts.
- **OpenAlex:** Cross-disciplinary academic metadata and Open Access publications.

## AI Scientific Referee (0-100 Rigor Index)

The `ScientificReferee` evaluates papers based on empirical methodology:
- Sample size ($N$) and statistical power
- Effect sizes and correlation coefficients ($r, p$-values)
- Meta-analytic weighting via Fisher's $z$-transformation
- Methodological classification (Randomized controlled trial, longitudinal empirical, observational, theoretical)

```python
from who import AcademicResearcher

researcher = AcademicResearcher(target_name="Target X")
results = researcher.research_topic(
    topic="Bayesian game theory mechanism design",
    sources=["arxiv"],
    limit=5
)
for paper in results["papers"]:
    print(paper["title"], paper["url"])
```
