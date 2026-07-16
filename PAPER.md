# Semantic Agreement for Metadata Extraction

## Motivation

The current evaluation measures agreement by exact string matching over three views:

- `fields`: extracted metadata field names
- `values`: extracted normalized values
- `field_value_pairs`: exact `(field, value)` pairs

This exact-match evaluation is useful because it measures how stable the structured output is. However, it can undercount agreement when models extract the same underlying metadata using slightly different strings.

For example:

```text
Model A field: paper_url
Model B field: arxiv_url
```

These two fields are not exact matches, but they may express the same metadata concept.

Similarly:

```text
Model A value: https://arxiv.org/abs/2503.10614
Model B value: arXiv:2503.10614
```

The strings differ, but the underlying paper reference is effectively the same.

## Why Exact Agreement Is Still Needed

Exact agreement should remain the baseline. It answers:

> Do models produce the same structured metadata strings?

This matters for indexing, database loading, search facets, and reproducible downstream processing. If two models use different field names for the same concept, the metadata may still be semantically correct, but it is less structurally stable.

## Why Semantic Agreement Is Useful

Semantic agreement answers a different question:

> Do models extract equivalent or near-equivalent metadata, even when the strings differ?

This is important for open-schema extraction because models can legitimately vary in wording, granularity, and naming.

Example:

```json
{"field": "project_page_url", "value": "https://ConsisLoRA.github.io/"}
```

versus:

```json
{"field": "website", "value": "https://ConsisLoRA.github.io/"}
```

Exact field agreement: no

Exact value agreement: yes

Semantic field agreement: likely yes

Exact field-value agreement: no

Semantic field-value agreement: likely yes, depending on threshold

## Proposed Semantic Evaluation

For each metadata item, create three text views:

```text
field_text = field
value_text = normalized_value or value
field_value_text = field + ": " + value
```

Then embed each text into a vector space and compare texts by cosine similarity.

A pair of items is considered semantically equivalent if:

```text
cosine_similarity(item_a, item_b) >= tau
```

where `tau` is an empirical threshold, for example:

```text
tau = 0.85
```

We should test multiple thresholds, such as:

```text
0.80, 0.85, 0.90, 0.95
```

## Clustering Instead of Pair Matching

To reuse the current agreement metrics, we can cluster semantically similar strings.

Example field cluster:

```text
cluster_001 = {
  "paper_url",
  "arxiv_url",
  "paper_link"
}
```

After clustering, each raw item is replaced by its cluster ID. The existing metrics can then be applied unchanged:

- Pairwise Jaccard
- Pairwise Cohen's kappa
- Multiway Jaccard
- Krippendorff's alpha

This keeps the evaluation framework simple and makes exact and semantic scores directly comparable.

## Metrics to Report

We keep the metric set focused.

Pairwise model comparison:

- Jaccard mean / median
- Cohen's kappa mean / median

Multi-model agreement:

- Multiway Jaccard mean / median
- Krippendorff's alpha mean / median

We report both exact and semantic versions:

```text
Exact agreement: strict string-level consistency
Semantic agreement: concept-level consistency after thresholded clustering
```

## Expected Interpretation

If semantic agreement is much higher than exact agreement, then models are extracting similar concepts but using different strings.

That would indicate the prompt is semantically effective but structurally unstable.

If exact and semantic agreement are both low, then models are extracting genuinely different metadata.

If exact agreement is already high, a preferred field schema may be less urgent.

## Example Outcome

Suppose exact field-value Jaccard is low:

```text
Exact field-value Jaccard = 0.12
```

but semantic field-value Jaccard is higher:

```text
Semantic field-value Jaccard = 0.28
```

This suggests many outputs differ by wording or field naming rather than by extracted meaning.

In that case, the next improvement should be a preferred field-name schema or a post-processing canonicalization step.

