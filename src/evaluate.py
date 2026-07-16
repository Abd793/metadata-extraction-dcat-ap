import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

from sklearn.metrics import cohen_kappa_score


DEFAULT_MODELS = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.5-flash-02-23",
    "qwen/qwen3.6-flash",
]

AGREEMENT_VIEWS = ("fields", "values", "field_value_pairs")


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def extract_metadata_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("metadata"), list):
        return data["metadata"]
    result = data.get("result")
    if isinstance(result, dict) and isinstance(result.get("metadata"), list):
        return result["metadata"]
    return []


def item_sets_from_file(path: Path) -> dict[str, set[str]]:
    fields = set()
    values = set()
    field_value_pairs = set()

    for item in extract_metadata_items(load_json(path)):
        if not isinstance(item, dict):
            continue

        field = normalize_text(item.get("field", ""))
        value = normalize_text(item.get("normalized_value") or item.get("value", ""))
        if field:
            fields.add(field)
        if value:
            values.add(value)
        if field and value:
            field_value_pairs.add(f"{field}\t{value}")

    return {
        "fields": fields,
        "values": values,
        "field_value_pairs": field_value_pairs,
    }


def discover_model_dirs(results_dir: Path, models: list[str] | None) -> dict[str, Path]:
    selected_models = models or DEFAULT_MODELS
    return {model: results_dir / safe_model_name(model) for model in selected_models}


def load_results(
    results_dir: Path,
    models: list[str] | None,
    limit: int | None,
) -> dict[str, dict[str, dict[str, set[str]]]]:
    model_dirs = discover_model_dirs(results_dir, models)
    results: dict[str, dict[str, dict[str, set[str]]]] = {}

    for model, model_dir in model_dirs.items():
        if not model_dir.exists():
            raise FileNotFoundError(f"Model result directory does not exist: {model_dir}")
        paths = sorted(model_dir.glob("*.json"))
        if limit is not None:
            paths = paths[:limit]
        results[model] = {path.stem: item_sets_from_file(path) for path in paths}

    return results


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def positive_agreement(vec_a: list[int], vec_b: list[int]) -> float:
    both_positive = sum(a == 1 and b == 1 for a, b in zip(vec_a, vec_b))
    positives = sum(vec_a) + sum(vec_b)
    if positives == 0:
        return 1.0
    return 2 * both_positive / positives


def multiway_jaccard(item_sets: list[set[str]]) -> float:
    if not item_sets:
        return 0.0
    union = set.union(*item_sets) if item_sets else set()
    if not union:
        return 1.0
    intersection = set.intersection(*item_sets)
    return len(intersection) / len(union)


def multiway_positive_agreement(item_sets: list[set[str]]) -> float:
    if not item_sets:
        return 0.0
    total_positive = sum(len(items) for items in item_sets)
    if total_positive == 0:
        return 1.0
    intersection = set.intersection(*item_sets)
    return len(item_sets) * len(intersection) / total_positive


def cohen_kappa(vec_a: list[int], vec_b: list[int]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Cohen kappa vectors must have the same length.")
    if not vec_a:
        return 0.0

    score = float(cohen_kappa_score(vec_a, vec_b))
    if math.isnan(score):
        return 1.0 if vec_a == vec_b else 0.0
    return score


def fleiss_kappa(item_sets: list[set[str]], universe: list[str]) -> float:
    if not item_sets or not universe:
        return 0.0

    n_raters = len(item_sets)
    if n_raters < 2:
        return 0.0

    category_totals = [0, 0]
    item_agreements = []
    for item in universe:
        present = sum(item in items for items in item_sets)
        absent = n_raters - present
        category_totals[0] += absent
        category_totals[1] += present
        item_agreements.append(
            (present * (present - 1) + absent * (absent - 1)) / (n_raters * (n_raters - 1))
        )

    observed = safe_mean(item_agreements)
    total_assignments = len(universe) * n_raters
    expected = sum((count / total_assignments) ** 2 for count in category_totals)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def krippendorff_alpha(item_sets: list[set[str]], universe: list[str]) -> float:
    if not item_sets or not universe:
        return 0.0

    n_raters = len(item_sets)
    if n_raters < 2:
        return 0.0

    observed_disagreements = []
    category_totals = [0, 0]
    for item in universe:
        present = sum(item in items for items in item_sets)
        absent = n_raters - present
        category_totals[0] += absent
        category_totals[1] += present
        observed_disagreements.append((present * absent * 2) / (n_raters * (n_raters - 1)))

    observed = safe_mean(observed_disagreements)
    total_assignments = len(universe) * n_raters
    expected = (category_totals[0] * category_totals[1] * 2) / (total_assignments * (total_assignments - 1))
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected


def binary_vector(items: set[str], universe: list[str]) -> list[int]:
    return [1 if item in items else 0 for item in universe]


def view_metrics(items_a: set[str], items_b: set[str], universe: list[str]) -> dict[str, Any]:
    vec_a = binary_vector(items_a, universe)
    vec_b = binary_vector(items_b, universe)
    return {
        "jaccard": jaccard(items_a, items_b),
        "cohen_kappa": cohen_kappa(vec_a, vec_b),
        "positive_agreement": positive_agreement(vec_a, vec_b),
        "count_a": len(items_a),
        "count_b": len(items_b),
        "shared_count": len(items_a & items_b),
    }


def summarize_rows(rows: list[dict[str, Any]], view: str) -> dict[str, float]:
    jaccards = [row[view]["jaccard"] for row in rows]
    kappas = [row[view]["cohen_kappa"] for row in rows]
    positive_agreements = [row[view]["positive_agreement"] for row in rows]
    return {
        "average_jaccard": safe_mean(jaccards),
        "median_jaccard": safe_median(jaccards),
        "average_cohen_kappa": safe_mean(kappas),
        "median_cohen_kappa": safe_median(kappas),
        "average_positive_agreement": safe_mean(positive_agreements),
        "median_positive_agreement": safe_median(positive_agreements),
    }


def multimodel_view_metrics(item_sets: list[set[str]], universe: list[str]) -> dict[str, Any]:
    intersection = set.intersection(*item_sets) if item_sets else set()
    union = set.union(*item_sets) if item_sets else set()
    return {
        "multiway_jaccard": multiway_jaccard(item_sets),
        "fleiss_kappa": fleiss_kappa(item_sets, universe),
        "krippendorff_alpha": krippendorff_alpha(item_sets, universe),
        "unanimous_positive_agreement": multiway_positive_agreement(item_sets),
        "min_count": min((len(items) for items in item_sets), default=0),
        "max_count": max((len(items) for items in item_sets), default=0),
        "shared_by_all_count": len(intersection),
        "union_count": len(union),
    }


def summarize_multimodel_rows(rows: list[dict[str, Any]], view: str) -> dict[str, float]:
    multiway_jaccards = [row[view]["multiway_jaccard"] for row in rows]
    fleiss_kappas = [row[view]["fleiss_kappa"] for row in rows]
    krippendorff_alphas = [row[view]["krippendorff_alpha"] for row in rows]
    unanimous_positive_agreements = [row[view]["unanimous_positive_agreement"] for row in rows]
    return {
        "average_multiway_jaccard": safe_mean(multiway_jaccards),
        "median_multiway_jaccard": safe_median(multiway_jaccards),
        "average_fleiss_kappa": safe_mean(fleiss_kappas),
        "median_fleiss_kappa": safe_median(fleiss_kappas),
        "average_krippendorff_alpha": safe_mean(krippendorff_alphas),
        "median_krippendorff_alpha": safe_median(krippendorff_alphas),
        "average_unanimous_positive_agreement": safe_mean(unanimous_positive_agreements),
        "median_unanimous_positive_agreement": safe_median(unanimous_positive_agreements),
    }


def evaluate(results: dict[str, dict[str, dict[str, set[str]]]]) -> dict[str, Any]:
    model_names = sorted(results)
    common_docs = sorted(set.intersection(*(set(results[model]) for model in model_names))) if model_names else []
    excluded_doc_counts = {
        model: len(set(docs) - set(common_docs))
        for model, docs in results.items()
    }
    universes = {
        view: sorted(
            {
                item
                for model in model_names
                for doc in common_docs
                for item in results[model][doc][view]
            }
        )
        for view in AGREEMENT_VIEWS
    }

    # Restrict every model to the same document set before computing pairwise metrics.
    common_results = {
        model: {doc: results[model][doc] for doc in common_docs}
        for model in model_names
    }

    pairwise = []
    for model_a, model_b in combinations(model_names, 2):
        docs_a = common_results[model_a]
        docs_b = common_results[model_b]

        doc_rows = []
        for doc in common_docs:
            row: dict[str, Any] = {"doc": doc}
            for view in AGREEMENT_VIEWS:
                row[view] = view_metrics(docs_a[doc][view], docs_b[doc][view], universes[view])
            doc_rows.append(row)

        pairwise.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "shared_doc_count": len(common_docs),
                "averages": {view: summarize_rows(doc_rows, view) for view in AGREEMENT_VIEWS},
                "docs": doc_rows,
            }
        )

    all_doc_rows = [row for pair in pairwise for row in pair["docs"]]
    multimodel_doc_rows = []
    for doc in common_docs:
        row: dict[str, Any] = {"doc": doc}
        for view in AGREEMENT_VIEWS:
            row[view] = multimodel_view_metrics(
                [common_results[model][doc][view] for model in model_names],
                universes[view],
            )
        multimodel_doc_rows.append(row)

    return {
        "model_count": len(model_names),
        "models": model_names,
        "doc_counts": {model: len(docs) for model, docs in results.items()},
        "common_doc_count": len(common_docs),
        "common_docs": common_docs,
        "excluded_doc_counts": excluded_doc_counts,
        "universe_sizes": {view: len(universe) for view, universe in universes.items()},
        "pairwise": pairwise,
        "overall": {
            "pair_doc_count": len(all_doc_rows),
            "averages": {view: summarize_rows(all_doc_rows, view) for view in AGREEMENT_VIEWS},
        },
        "overall_multimodel": {
            "doc_count": len(multimodel_doc_rows),
            "averages": {
                view: summarize_multimodel_rows(multimodel_doc_rows, view)
                for view in AGREEMENT_VIEWS
            },
            "docs": multimodel_doc_rows,
        },
    }


def print_view(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: "
        f"Jaccard mean/median={metrics['average_jaccard']:.4f}/{metrics['median_jaccard']:.4f}, "
        f"Cohen's kappa mean/median={metrics['average_cohen_kappa']:.4f}/{metrics['median_cohen_kappa']:.4f}, "
        f"Positive agreement mean/median="
        f"{metrics['average_positive_agreement']:.4f}/{metrics['median_positive_agreement']:.4f}"
    )


def print_multimodel_view(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: "
        f"Multiway Jaccard mean/median="
        f"{metrics['average_multiway_jaccard']:.4f}/{metrics['median_multiway_jaccard']:.4f}, "
        f"Fleiss' kappa mean/median="
        f"{metrics['average_fleiss_kappa']:.4f}/{metrics['median_fleiss_kappa']:.4f}, "
        f"Krippendorff's alpha mean/median="
        f"{metrics['average_krippendorff_alpha']:.4f}/{metrics['median_krippendorff_alpha']:.4f}, "
        f"Unanimous positive agreement mean/median="
        f"{metrics['average_unanimous_positive_agreement']:.4f}/"
        f"{metrics['median_unanimous_positive_agreement']:.4f}"
    )


def print_report(report: dict[str, Any]) -> None:
    print(f"Models: {', '.join(report['models'])}")
    print("Document counts:")
    for model, count in report["doc_counts"].items():
        print(f"  {model}: {count}")
    print(f"Common documents used for evaluation: {report['common_doc_count']}")
    print("Excluded documents:")
    for model, count in report["excluded_doc_counts"].items():
        print(f"  {model}: {count}")
    print("Universe sizes:")
    for view, count in report["universe_sizes"].items():
        print(f"  {view}: {count}")

    print("\nPairwise metrics:")
    for pair in report["pairwise"]:
        print(f"  {pair['model_a']} vs {pair['model_b']} (shared docs={pair['shared_doc_count']})")
        for view in AGREEMENT_VIEWS:
            print_view(f"    {view}", pair["averages"][view])

    print("\nOverall:")
    print(f"  Pair-doc count: {report['overall']['pair_doc_count']}")
    for view in AGREEMENT_VIEWS:
        print_view(f"  {view}", report["overall"]["averages"][view])

    print("\nOverall multimodel agreement:")
    print(f"  Document count: {report['overall_multimodel']['doc_count']}")
    for view in AGREEMENT_VIEWS:
        print_multimodel_view(f"  {view}", report["overall_multimodel"]["averages"][view])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate metadata extraction agreement across model outputs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/llm"))
    parser.add_argument("--models", nargs="*", default=None, help="Optional model ids.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum JSON files per model.")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation_report.json"), help="JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = load_results(args.results_dir, args.models, args.limit)
    report = evaluate(results)
    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
