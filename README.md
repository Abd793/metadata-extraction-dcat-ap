# Metadata Extraction Agreement Across Models

## Setup

- Models evaluated:
  - DeepSeek V4 Flash
  - Qwen 3.5 Flash
  - Qwen 3.6 Flash
- Evaluation uses only files available for all models.
- Common README files evaluated: **30**

## Metric Definitions

| Metric                       | Definition                                                | What It Implies Here                                                        |
|------------------------------|-----------------------------------------------------------|-----------------------------------------------------------------------------|
| Jaccard                      | Intersection / union of extracted items                   | How much two models overlap relative to everything either model extracted   |
| Cohen's Kappa                | Pairwise agreement corrected for chance                   | Whether two models agree more than expected by random field/value selection |
| Positive Agreement           | Shared positives / total positives                        | How often extracted items from two models are mutually shared               |
| Multiway Jaccard             | Intersection across all models / union across all models  | Strict overlap among all three models                                       |
| Fleiss' Kappa                | Multi-rater agreement corrected for chance                | Whether all three models agree more than expected by chance                 |
| Unanimous Positive Agreement | Items extracted by all models / total extracted positives | How often all three models extract the same item                            |

## Metric Definitions With Tiny Examples

### Jaccard: Intersection divided by union.

```text
Model A fields = {license, paper_url, install_command}
Model B fields = {license, paper_url, dataset_url}

Intersection = {license, paper_url} = 2
Union = {license, paper_url, install_command, dataset_url} = 4

Jaccard = 2 / 4 = 0.50
```

### Cohen's Kappa: Pairwise agreement corrected for chance.

Tiny binary example over four possible fields:

```text
Fields:   license  paper_url  dataset_url  install_command
Model A:     1        1           0              1
Model B:     1        1           1              0
```

They agree on:

```text
license = both yes
paper_url = both yes
```

They disagree on:

```text
dataset_url
install_command
```

Raw agreement is `2 / 4 = 0.50`, but Cohen's kappa adjusts this by asking:
“How much agreement would we expect by chance?”


### Positive Agreement: Agreement only over positive extracted items.

```text
Model A fields = {license, paper_url, install_command}
Model B fields = {license, paper_url, dataset_url}

Shared positives = {license, paper_url} = 2
Total positives = 3 + 3 = 6

Positive agreement = 2 * 2 / 6 = 0.67
```

### Multiway Jaccard: Intersection across all models divided by union across all models.

```text
DeepSeek = {license, paper_url, install_command}
Qwen 3.5 = {license, paper_url, dataset_url}
Qwen 3.6 = {license, paper_url, citation}

Shared by all = {license, paper_url} = 2
Union = {license, paper_url, install_command, dataset_url, citation} = 5

Multiway Jaccard = 2 / 5 = 0.40
```

### Fleiss' Kappa: Multi-rater agreement corrected for chance.

Tiny example: three models rate whether each field is present.

```text
Field            DeepSeek   Qwen 3.5   Qwen 3.6
license             1          1          1
paper_url           1          1          1
dataset_url         0          1          0
install_command     1          0          0
```

Fleiss' kappa asks:
“Across all fields, do the three models agree more than expected by chance?”

### Unanimous Positive Agreement: How much of the extracted positive mass is shared by all models.

```text
DeepSeek = {license, paper_url, install_command}
Qwen 3.5 = {license, paper_url, dataset_url}
Qwen 3.6 = {license, paper_url, citation}

Shared by all = {license, paper_url} = 2
Total positives = 3 + 3 + 3 = 9
Models = 3

Unanimous positive agreement = 3 * 2 / 9 = 0.67
```

## Agreement Views

- **Fields**: whether models extract the same metadata field names.
- **Values**: whether models extract the same normalized values.
- **Field + Value pairs**: whether both field name and value match exactly.

## Overall Pairwise Agreement

| View                | Jaccard | Cohen's Kappa | Positive Agreement |
|---------------------|--------:|--------------:|-------------------:|
| Fields              |  0.2279 |        0.3406 |             0.3570 |
| Values              |  0.3090 |        0.4453 |             0.4540 |
| Field + Value Pairs |  0.1489 |        0.2365 |             0.2464 |

## Three-Model Agreement

| View                | Multiway Jaccard | Fleiss' Kappa | Unanimous Positive Agreement |
|---------------------|-----------------:|--------------:|-----------------------------:|
| Fields              |           0.0958 |        0.3274 |                       0.2005 |
| Values              |           0.1828 |        0.4377 |                       0.3396 |
| Field + Value Pairs |           0.0564 |        0.2242 |                       0.1277 |

## Pairwise Agreement

| Model Pair           | View          | Jaccard | Cohen's Kappa | Positive Agreement |
|----------------------|---------------|--------:|--------------:|-------------------:|
| DeepSeek vs Qwen 3.5 | Fields        |  0.1698 |        0.2656 |             0.2847 |
| DeepSeek vs Qwen 3.5 | Values        |  0.3029 |        0.4376 |             0.4471 |
| DeepSeek vs Qwen 3.5 | Field + Value |  0.1144 |        0.1894 |             0.2005 |
| DeepSeek vs Qwen 3.6 | Fields        |  0.1856 |        0.2836 |             0.3026 |
| DeepSeek vs Qwen 3.6 | Values        |  0.2759 |        0.4071 |             0.4173 |
| DeepSeek vs Qwen 3.6 | Field + Value |  0.1136 |        0.1832 |             0.1946 |
| Qwen 3.5 vs Qwen 3.6 | Fields        |  0.3282 |        0.4726 |             0.4837 |
| Qwen 3.5 vs Qwen 3.6 | Values        |  0.3482 |        0.4911 |             0.4976 |
| Qwen 3.5 vs Qwen 3.6 | Field + Value |  0.2188 |        0.3370 |             0.3440 |


## Key Observations

- **Qwen 3.5 and Qwen 3.6 agree the most**, especially on values and field names.
- **Values have higher agreement than fields**, suggesting models often find similar facts but label them differently.
- **Field + value agreement is lowest**, showing that exact structured metadata consistency is still weak.

<!-- - A preferred field-name schema should likely improve agreement, especially for field and field-value metrics. -->

<!-- Models agree most on **extracted values**, less on **field names**, and least on exact **field-value pairs**. -->

<!-- This suggests the extraction prompt is finding similar evidence, but the open-schema field naming introduces inconsistency. A preferred field-name schema should improve structured agreement. -->
