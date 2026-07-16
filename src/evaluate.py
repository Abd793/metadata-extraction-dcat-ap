import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score


DEFAULT_MODELS = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.5-flash-02-23",
    "qwen/qwen3.6-flash",
]

AGREEMENT_VIEWS = ("fields", "values", "field_value_pairs")
SEMANTIC_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_SIMILARITY_CHUNK_SIZE = 512


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def item_text_for_embedding(item: str, view: str) -> str:
    if view == "field_value_pairs":
        return item.replace("\t", ": ")
    return item


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


def multiway_jaccard(item_sets: list[set[str]]) -> float:
    if not item_sets:
        return 0.0
    union = set.union(*item_sets) if item_sets else set()
    if not union:
        return 1.0
    intersection = set.intersection(*item_sets)
    return len(intersection) / len(union)


def cohen_kappa(vec_a: list[int], vec_b: list[int]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Cohen kappa vectors must have the same length.")
    if not vec_a:
        return 0.0

    score = float(cohen_kappa_score(vec_a, vec_b))
    if math.isnan(score):
        return 1.0 if vec_a == vec_b else 0.0
    return score


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
        "count_a": len(items_a),
        "count_b": len(items_b),
        "shared_count": len(items_a & items_b),
    }


def summarize_rows(rows: list[dict[str, Any]], view: str) -> dict[str, float]:
    jaccards = [row[view]["jaccard"] for row in rows]
    kappas = [row[view]["cohen_kappa"] for row in rows]
    return {
        "average_jaccard": safe_mean(jaccards),
        "median_jaccard": safe_median(jaccards),
        "average_cohen_kappa": safe_mean(kappas),
        "median_cohen_kappa": safe_median(kappas),
    }


def multimodel_view_metrics(item_sets: list[set[str]], universe: list[str]) -> dict[str, Any]:
    intersection = set.intersection(*item_sets) if item_sets else set()
    union = set.union(*item_sets) if item_sets else set()
    return {
        "multiway_jaccard": multiway_jaccard(item_sets),
        "krippendorff_alpha": krippendorff_alpha(item_sets, universe),
        "min_count": min((len(items) for items in item_sets), default=0),
        "max_count": max((len(items) for items in item_sets), default=0),
        "shared_by_all_count": len(intersection),
        "union_count": len(union),
    }


def summarize_multimodel_rows(rows: list[dict[str, Any]], view: str) -> dict[str, float]:
    multiway_jaccards = [row[view]["multiway_jaccard"] for row in rows]
    krippendorff_alphas = [row[view]["krippendorff_alpha"] for row in rows]
    return {
        "average_multiway_jaccard": safe_mean(multiway_jaccards),
        "median_multiway_jaccard": safe_median(multiway_jaccards),
        "average_krippendorff_alpha": safe_mean(krippendorff_alphas),
        "median_krippendorff_alpha": safe_median(krippendorff_alphas),
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


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def common_docs_for_results(results: dict[str, dict[str, dict[str, set[str]]]]) -> list[str]:
    model_names = sorted(results)
    if not model_names:
        return []
    return sorted(set.intersection(*(set(results[model]) for model in model_names)))


def load_embedding_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SEMANTIC_EMBEDDING_MODEL, local_files_only=True)


def semantic_cluster_map(
    items: list[str],
    view: str,
    threshold: float,
    embedding_model: Any,
) -> tuple[dict[str, str], list[list[str]]]:
    if not items:
        return {}, []

    texts = [item_text_for_embedding(item, view) for item in items]
    embeddings = embedding_model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    union_find = UnionFind(len(items))

    for start in range(0, len(items), SEMANTIC_SIMILARITY_CHUNK_SIZE):
        end = min(start + SEMANTIC_SIMILARITY_CHUNK_SIZE, len(items))
        similarities = np.matmul(embeddings[start:end], embeddings.T)
        row_indices, col_indices = np.where(similarities >= threshold)
        for local_row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
            row_index = start + local_row_index
            if col_index <= row_index:
                continue
            union_find.union(row_index, col_index)

    clusters_by_root: dict[int, list[str]] = {}
    for index, item in enumerate(items):
        clusters_by_root.setdefault(union_find.find(index), []).append(item)

    clusters = [sorted(cluster) for cluster in clusters_by_root.values()]
    clusters.sort(key=lambda cluster: (cluster[0], len(cluster)))

    item_to_cluster = {}
    for cluster_index, cluster in enumerate(clusters):
        cluster_id = f"{view}_cluster_{cluster_index:05d}"
        for item in cluster:
            item_to_cluster[item] = cluster_id

    return item_to_cluster, clusters


def build_semantic_cluster_maps(
    results: dict[str, dict[str, dict[str, set[str]]]],
    threshold: float,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    model_names = sorted(results)
    common_docs = common_docs_for_results(results)
    cluster_maps = {}
    cluster_report = {}
    embedding_model = load_embedding_model()

    for view in AGREEMENT_VIEWS:
        items = sorted(
            {
                item
                for model in model_names
                for doc in common_docs
                for item in results[model][doc][view]
            }
        )
        item_to_cluster, clusters = semantic_cluster_map(items, view, threshold, embedding_model)
        cluster_maps[view] = item_to_cluster
        cluster_report[view] = {
            "raw_item_count": len(items),
            "cluster_count": len(clusters),
            "merged_cluster_count": sum(1 for cluster in clusters if len(cluster) > 1),
            "examples": [cluster for cluster in clusters if len(cluster) > 1][:10],
        }

    return cluster_maps, cluster_report


def apply_semantic_clusters(
    results: dict[str, dict[str, dict[str, set[str]]]],
    cluster_maps: dict[str, dict[str, str]],
) -> dict[str, dict[str, dict[str, set[str]]]]:
    transformed: dict[str, dict[str, dict[str, set[str]]]] = {}
    for model, docs in results.items():
        transformed[model] = {}
        for doc, item_sets in docs.items():
            transformed[model][doc] = {
                view: {cluster_maps[view].get(item, item) for item in items}
                for view, items in item_sets.items()
            }
    return transformed


def evaluate_semantic(
    results: dict[str, dict[str, dict[str, set[str]]]],
    threshold: float,
) -> dict[str, Any]:
    cluster_maps, cluster_report = build_semantic_cluster_maps(results, threshold)
    semantic_results = apply_semantic_clusters(results, cluster_maps)
    report = evaluate(semantic_results)
    report["semantic"] = {
        "backend": SEMANTIC_EMBEDDING_MODEL,
        "threshold": threshold,
        "clusters": cluster_report,
    }
    return report


def print_view(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: "
        f"Jaccard mean/median={metrics['average_jaccard']:.4f}/{metrics['median_jaccard']:.4f}, "
        f"Cohen's kappa mean/median={metrics['average_cohen_kappa']:.4f}/{metrics['median_cohen_kappa']:.4f}"
    )


def print_multimodel_view(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: "
        f"Multiway Jaccard mean/median="
        f"{metrics['average_multiway_jaccard']:.4f}/{metrics['median_multiway_jaccard']:.4f}, "
        f"Krippendorff's alpha mean/median="
        f"{metrics['average_krippendorff_alpha']:.4f}/{metrics['median_krippendorff_alpha']:.4f}"
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


def print_semantic_summary(report: dict[str, Any]) -> None:
    semantic = report["semantic"]
    print(f"\nSemantic clustering: backend={semantic['backend']}, threshold={semantic['threshold']}")
    for view in AGREEMENT_VIEWS:
        clusters = semantic["clusters"][view]
        print(
            f"  {view}: raw items={clusters['raw_item_count']}, "
            f"clusters={clusters['cluster_count']}, merged clusters={clusters['merged_cluster_count']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate metadata extraction agreement across model outputs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/llm"))
    parser.add_argument("--models", nargs="*", default=None, help="Optional model ids.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum JSON files per model.")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation_report.json"), help="JSON report path.")
    parser.add_argument("--semantic", action="store_true", help="Also evaluate thresholded semantic agreement.")
    parser.add_argument("--semantic-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = load_results(args.results_dir, args.models, args.limit)
    exact_report = evaluate(results)

    if args.semantic:
        semantic_report = evaluate_semantic(results, args.semantic_threshold)
        report = {
            "exact": exact_report,
            "semantic": semantic_report,
        }
        print("Exact agreement")
        print_report(exact_report)
        print("\nSemantic agreement")
        print_report(semantic_report)
        print_semantic_summary(semantic_report)
    else:
        report = exact_report
        print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
