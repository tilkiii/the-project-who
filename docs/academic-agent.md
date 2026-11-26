# Autonomous Academic Literature & Grounding

The `who.learn.autonomous_researcher` (`AcademicResearcher`) module grounds behavioral and psychometric hypotheses against peer-reviewed scientific literature.

## Integrated Repositories

- **arXiv API:** Quantitative finance, game theory, mathematical economics, and computer science.
- **PubMed (E-Utilities):** Cognitive psychology, neurobiology, and chronomics.
- **Semantic Scholar API:** Citation graphs and paper abstracts.
- **Crossref:** Cross-disciplinary bibliographic metadata, DOIs, and publisher abstracts.

## 3-Tier Evidence Separation

- **`QUANTITATIVE_EXTRACTED`**: Retrieved papers reporting explicit empirical correlation (`r`) and sample size (`N >= 10`), eligible for Fisher-z Bayesian weight updates.
- **`QUALITATIVE_ABSTRACT_MATCH`**: Retrieved relevant abstracts lacking explicit `r` or `N`; preserved qualitatively (`r=None`, `N=None`) and excluded from Bayesian Fisher updates.
- **`REFERENCE_BASELINE_PRIOR`**: Curated canonical baseline studies, explicitly tagged as reference priors and never attributed to unrelated retrieved papers.

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
