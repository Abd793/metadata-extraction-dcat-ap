#!/usr/bin/env python3
"""Run OpenRouter metadata extraction over README files with a durable cache."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError


DEFAULT_MODELS = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.5-flash-02-23",
    "openai/gpt-oss-120b",
]

MODEL_REASONING_HANDLERS = [
    {
        "name": "openai_gpt_oss_120b",
        "model_pattern": r"^openai/gpt-oss-120b$",
        "request_reasoning": False,
        "message_reasoning_fields": ["reasoning", "reasoning_content"],
        "message_reasoning_detail_fields": ["reasoning_details"],
        "strip_inline_think": True,
    },
    {
        "name": "qwen_qwen3_5_flash_02_23",
        "model_pattern": r"^qwen/qwen3\.5-flash-02-23$",
        "request_reasoning": True,
        "message_reasoning_fields": ["reasoning_content", "reasoning"],
        "message_reasoning_detail_fields": ["reasoning_details"],
        "strip_inline_think": True,
    },
    {
        "name": "deepseek_v4_pro",
        "model_pattern": r"^(deepseek/)?deepseek-v4-pro$",
        "request_reasoning": True,
        "message_reasoning_fields": ["reasoning_content", "reasoning"],
        "message_reasoning_detail_fields": ["reasoning_details"],
        "strip_inline_think": True,
    },
]

GENERIC_REASONING_HANDLER = {
    "name": "generic",
    "model_pattern": r".*",
    "request_reasoning": False,
    "message_reasoning_fields": ["reasoning", "reasoning_content"],
    "message_reasoning_detail_fields": ["reasoning_details"],
    "strip_inline_think": True,
}

METADATA_DELIMITER = "===========metadata extract========="

SYSTEM_PROMPT_OPEN_SCHEMA = """You are a metadata extraction system for artifacts documented on GitHub.

Your task is to read the provided artifact document and extract all useful metadata that would help someone understand, index, search, reproduce, or reuse the artifact.

The artifact document is a GitHub README file in Markdown format. Use its structure (headers, code blocks, badges, lists) to help you locate and delimit evidence spans and populate evidence_location -- for example, citing a Markdown header as the location, or treating a fenced code block as a single coherent span. This is a structural note only: it does not imply a fixed list of expected fields, and you should still identify whatever metadata is available in this specific document, not a generic README template.

Use only the information explicitly stated in the document. Do not use outside knowledge. Do not infer facts that are not directly supported by the text.

You may decide by yourself which metadata fields are relevant and how to name them, based entirely on what this specific document contains. Do not restrict yourself to any fixed list of field types, and do not assume the artifact belongs to any particular kind of project unless the document itself establishes that. Name each field using whatever term best describes what you found. Prefer short, snake_case field names.

For each metadata item, return:
- field: a short snake_case name for the metadata type you identified
- value: the extracted value, as close to verbatim as possible
- normalized_value: apply ONLY these normalizations, nothing else:
  - trim surrounding whitespace/punctuation
  - collapse internal whitespace
  - do not shorten, expand, or abbreviate names
  If no normalization applies, repeat value exactly.
- evidence: a text span copied VERBATIM from the artifact text, including original Markdown syntax. Do not paraphrase, reflow, merge non-adjacent sentences, or fix typos.
- evidence_location: line number, Markdown section/header title, or "unknown" if not available
- confidence: one of "high", "medium", "low"
- reason: a short explanation of why this value is useful metadata

Important rules:
1. Extract only metadata supported by evidence in the artifact text.
2. If a possible metadata value is ambiguous, include it with confidence "low" and explain the ambiguity in "reason".
3. Do not output generic concepts unless the document names a concrete value.
4. Do not invent missing metadata.
5. If the same underlying fact could be tagged with more than one field name, choose the single field name you judge most precise, and do not emit duplicate items for the same evidence span under multiple field names.
6. If no useful metadata is found, return an empty list.
7. Output ONLY the JSON object below. No markdown code fences, no preamble, no explanation outside the JSON structure.

Return only valid JSON in the following format:

{
  "artifact_id": "<ARTIFACT_ID>",
  "metadata": [
    {
      "field": "...",
      "value": "...",
      "normalized_value": "...",
      "evidence": "...",
      "evidence_location": "...",
      "confidence": "high | medium | low",
      "reason": "..."
    }
  ]
}"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract README metadata through OpenRouter with threaded requests and cache-safe outputs."
    )
    parser.add_argument("--input-dir", default="readme_files", help="Directory containing README markdown files.")
    parser.add_argument("--output-dir", default="results", help="Root directory for cached raw and parsed responses.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="OpenRouter model ids.")
    parser.add_argument("--threads", type=int, default=4, help="Worker threads. Use --threads 1 for debugging.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of README files to process.")
    parser.add_argument("--doc", action="append", default=None, help="Process only this doc basename. Can be repeated.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--reasoning-effort", default="high", help="Reasoning effort for model handlers that request reasoning.")
    parser.add_argument("--force", action="store_true", help="Ignore successful cache files and request again.")
    parser.add_argument("--dry-run", action="store_true", help="Print scheduled work without calling OpenRouter.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def read_docs(input_dir: Path, only_docs: set[str] | None, limit: int | None) -> list[Path]:
    docs = sorted(input_dir.glob("*.md"))
    if only_docs:
        docs = [path for path in docs if path.name in only_docs or path.stem in only_docs]
    if limit is not None:
        docs = docs[:limit]
    return docs


def output_paths(output_dir: Path, doc_path: Path, model: str) -> tuple[Path, Path]:
    model_name = safe_name(model.replace("/", "__"))
    model_dir = output_dir / model_name
    return model_dir / f"{doc_path.stem}.txt", model_dir / f"{doc_path.stem}.json"


def raw_cache_exists(path: Path) -> bool:
    return path.exists()


def parsed_cache_exists(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("success"))


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def extract_json_object(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def reasoning_handler_for_model(model: str) -> dict[str, Any]:
    for handler in MODEL_REASONING_HANDLERS:
        if re.search(handler["model_pattern"], model, flags=re.IGNORECASE):
            return handler
    return GENERIC_REASONING_HANDLER


def collect_reasoning(message: dict[str, Any], handler: dict[str, Any]) -> tuple[Any | None, Any | None]:
    reasoning = None
    for field in handler["message_reasoning_fields"]:
        if message.get(field):
            reasoning = message[field]
            break

    reasoning_details = None
    for field in handler["message_reasoning_detail_fields"]:
        if message.get(field):
            reasoning_details = message[field]
            break

    if reasoning is None and isinstance(reasoning_details, list):
        text_parts = []
        for item in reasoning_details:
            if isinstance(item, dict):
                for key in ("text", "content", "summary"):
                    if item.get(key):
                        text_parts.append(str(item[key]))
        reasoning = "\n".join(text_parts) if text_parts else None
    return reasoning, reasoning_details


def split_inline_reasoning(text: str, handler: dict[str, Any]) -> tuple[str, str | None]:
    if not handler["strip_inline_think"]:
        return text, None
    pattern = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
    parts = [match.group(1).strip() for match in pattern.finditer(text)]
    cleaned = pattern.sub("", text).strip()
    return cleaned, "\n\n".join(parts) if parts else None


def combine_reasoning(*values: Any | None) -> str | None:
    parts = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        if text:
            parts.append(text)
    return "\n\n".join(parts) if parts else None


def compose_raw_cache_text(response_text: str, reasoning_text: str | None) -> str:
    response_text = response_text.strip()
    if not reasoning_text:
        return response_text
    return f"{reasoning_text.strip()}\n{METADATA_DELIMITER}\n{response_text}"


def split_raw_cache_text(raw_text: str) -> tuple[str | None, str]:
    if METADATA_DELIMITER not in raw_text:
        return None, raw_text.strip()
    reasoning_text, response_text = raw_text.split(METADATA_DELIMITER, 1)
    return reasoning_text.strip() or None, response_text.strip()


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return json.loads(response.model_dump_json())


def build_messages(doc_content: str, doc_name: str) -> list[dict[str, str]]:
    user_prompt = (
        "Please extract metadata from the following GitHub README document. "
        "The artifact_id should be inferred from the project name (use the repository name if available, or the title).\n\n"
        f"Document name: {doc_name}\n\n---\n{doc_content}\n---"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT_OPEN_SCHEMA}, {"role": "user", "content": user_prompt}]


def parsed_record_from_raw(
    *,
    raw_text: str,
    model: str,
    doc_name: str,
    json_path: Path,
    raw_path: Path,
    prompt_hash: str | None = None,
    request: dict[str, Any] | None = None,
    raw_response: dict[str, Any] | None = None,
    reasoning_response: Any | None = None,
    reasoning_details: Any | None = None,
    reasoning_handler: str | None = None,
    usage: dict[str, Any] | None = None,
    latency_seconds: float | None = None,
) -> dict[str, Any]:
    cached_reasoning, response_text = split_raw_cache_text(raw_text)
    reasoning_response = combine_reasoning(reasoning_response, cached_reasoning)
    record: dict[str, Any] = {
        "model": model,
        "doc": doc_name,
        "raw_response_path": str(raw_path),
        "final_response": response_text,
    }
    if raw_response is not None:
        record["raw_response"] = raw_response
    if reasoning_response is not None:
        record["reasoning_response"] = reasoning_response
    if reasoning_details is not None:
        record["reasoning_details"] = reasoning_details
    if reasoning_handler is not None:
        record["reasoning_handler"] = reasoning_handler
    if usage is not None:
        record["usage"] = usage
    if request is not None:
        record["request"] = request
    elif prompt_hash is not None:
        record["request"] = {"prompt_hash": prompt_hash}
    if latency_seconds is not None:
        record["latency_seconds"] = latency_seconds

    try:
        record["result"] = extract_json_object(response_text)
        record["success"] = True
    except Exception as exc:
        record["success"] = False
        record["error"] = f"Could not parse LLM response as JSON: {exc!r}"

    atomic_write_json(json_path, record)
    return record


def parse_cached_raw(job: dict[str, Any]) -> dict[str, Any]:
    handler = reasoning_handler_for_model(job["model"])
    raw_text = job["raw_path"].read_text(encoding="utf-8")
    existing_reasoning, existing_response = split_raw_cache_text(raw_text)
    existing_response, inline_reasoning = split_inline_reasoning(existing_response, handler)
    reasoning_response = combine_reasoning(existing_reasoning, inline_reasoning)
    if inline_reasoning is not None:
        raw_text = compose_raw_cache_text(existing_response, reasoning_response)
        atomic_write_text(job["raw_path"], raw_text)
    return parsed_record_from_raw(
        raw_text=raw_text,
        model=job["model"],
        doc_name=job["doc_path"].name,
        json_path=job["json_path"],
        raw_path=job["raw_path"],
        reasoning_handler=handler["name"],
    )


def request_one(job: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    doc_path = job["doc_path"]
    model = job["model"]
    raw_path = job["raw_path"]
    json_path = job["json_path"]
    doc_content = doc_path.read_text(encoding="utf-8")
    messages = build_messages(doc_content, doc_path.name)
    prompt_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()
    handler = reasoning_handler_for_model(model)

    extra_body: dict[str, Any] = {}
    if handler["request_reasoning"]:
        extra_body["reasoning"] = {"effort": job["reasoning_effort"]}

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=job["api_key"],
            timeout=job["timeout"],
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=job["temperature"],
            extra_body=extra_body or None,
            extra_headers={
                "HTTP-Referer": "https://github.com/metadata-extraction",
                "X-Title": "metadata-extraction",
            },
        )
        raw_response = response_to_dict(response)
        choice = raw_response.get("choices", [{}])[0]
        message = choice.get("message") or {}
        final_response = message.get("content") or ""
        final_response, inline_reasoning = split_inline_reasoning(final_response, handler)
        provider_reasoning, reasoning_details = collect_reasoning(message, handler)
        reasoning_response = combine_reasoning(provider_reasoning, inline_reasoning)
        raw_cache_text = compose_raw_cache_text(final_response, reasoning_response)
        atomic_write_text(raw_path, raw_cache_text)
        usage = raw_response.get("usage")
        record = parsed_record_from_raw(
            raw_text=raw_cache_text,
            model=model,
            doc_name=doc_path.name,
            json_path=json_path,
            raw_path=raw_path,
            raw_response=raw_response,
            reasoning_details=reasoning_details,
            reasoning_handler=handler["name"],
            usage=usage,
            request={
                "model": model,
                "temperature": job["temperature"],
                "extra_body": extra_body,
                "reasoning_handler": handler["name"],
                "prompt_hash": prompt_hash,
            },
            latency_seconds=round(time.time() - start, 3),
        )
    except OpenAIError as exc:
        record = {
            "model": model,
            "doc": doc_path.name,
            "success": False,
            "error": repr(exc),
            "request": {
                "model": model,
                "temperature": job["temperature"],
                "extra_body": extra_body,
                "reasoning_handler": handler["name"],
                "prompt_hash": prompt_hash,
            },
            "latency_seconds": round(time.time() - start, 3),
        }
    except Exception as exc:
        record = {
            "model": model,
            "doc": doc_path.name,
            "success": False,
            "error": repr(exc),
            "request": {
                "model": model,
                "temperature": job["temperature"],
                "extra_body": extra_body,
                "reasoning_handler": handler["name"],
                "prompt_hash": prompt_hash,
            },
            "latency_seconds": round(time.time() - start, 3),
        }

    atomic_write_json(json_path, record)
    return record


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if args.threads < 1:
        print("--threads must be at least 1.", file=sys.stderr)
        return 2

    docs = read_docs(input_dir, set(args.doc) if args.doc else None, args.limit)
    jobs = []
    skipped = 0
    parsed_from_cache = 0
    for doc_path in docs:
        for model in args.models:
            raw_path, json_path = output_paths(output_dir, doc_path, model)
            if not args.force and raw_cache_exists(raw_path):
                skipped += 1
                if not args.dry_run and not parsed_cache_exists(json_path):
                    parse_cached_raw(
                        {
                            "doc_path": doc_path,
                            "model": model,
                            "raw_path": raw_path,
                            "json_path": json_path,
                        }
                    )
                    parsed_from_cache += 1
                continue
            jobs.append(
                {
                    "doc_path": doc_path,
                    "model": model,
                    "raw_path": raw_path,
                    "json_path": json_path,
                    "temperature": args.temperature,
                    "timeout": args.timeout,
                    "api_key": api_key,
                    "reasoning_effort": args.reasoning_effort,
                }
            )

    print(f"Found {len(docs)} docs, {len(args.models)} models.")
    print(f"Scheduled {len(jobs)} requests; skipped {skipped} existing raw response files.")
    if parsed_from_cache:
        print(f"Parsed {parsed_from_cache} cached raw response files into JSON.")

    if args.dry_run:
        for job in jobs:
            print(f"DRY RUN {job['doc_path'].name} / {job['model']} -> {job['raw_path']}")
        return 0
    if not jobs:
        return 0
    if not api_key:
        print("OPENROUTER_API_KEY is required for uncached requests.", file=sys.stderr)
        return 2

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_job = {executor.submit(request_one, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                print(f"FAILED {job['doc_path'].name} / {job['model']}: {exc}")
                continue
            status = "OK" if result.get("success") else "FAILED"
            if not result.get("success"):
                failures += 1
            print(f"{status} {result['doc']} / {result['model']} -> {job['raw_path']}")

    if failures:
        print(f"Completed with {failures} failed request(s).")
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
