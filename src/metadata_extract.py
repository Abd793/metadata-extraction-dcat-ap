# Standard library
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Dict

# Third-party
# import numpy as np
# import pandas as pd
# from termcolor import colored
# from tqdm import tqdm


# import os
# import json
# import asyncio
# import glob
from openai import OpenAI
# from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# configurations
# OPENROUTER_API_KEY = "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"
# MODELS = [
#     "deepseek/deepseek-v4-flash",           # think mode default, temperature=0
#     "qwen/qwen3.5-flash",                  # think mode default, temperature=0
#     "openai/gpt-oss-120b"           # temperature=0 (if supported)
# ]

# FOLDER_PATH = r"E:\TUB\WS25-26\fiz\metadata_extraction\readme_files"  # folder containing README markdown files
# MAX_CONCURRENT = 20                       # request concurrency limit (adjust based on your system and API limits)
# OUTPUT_DIR = "."                          # output directory for JSON results

# Prompt
PROMPT = """You are a metadata extraction system for artifacts documented on GitHub.

Your task is to read the provided artifact document and extract useful metadata that would help someone understand, index, search, reproduce, or reuse the artifact.

The artifact document is a GitHub README file in Markdown format. The input is shown with line numbers for reference only. Use the line numbers to populate evidence_location, but do not include line number prefixes in the evidence field. Use the README structure (headers, code blocks, badges, lists) to help you locate and delimit evidence spans and populate evidence_location — for example, citing a Markdown header as the location, or treating a fenced code block as a single coherent span. This is a structural note only: it does not imply a fixed list of expected fields, and you should still identify whatever metadata is available in this specific document, not a generic README template.

Use only the information explicitly stated in the document. Do not use outside knowledge. Do not infer facts that are not directly supported by the text.

You may decide by yourself which metadata fields are relevant and how to name them, based entirely on what this specific document contains. Do not restrict yourself to any fixed list of field types, and do not assume the artifact belongs to any particular kind of project (e.g. software library, machine learning model, dataset, research paper, hardware project, documentation site, etc.) unless the document itself establishes that. Name each field using whatever term best describes what you found. Prefer short, snake_case field names (e.g. "programming_language", not "the language used").

Prioritize metadata that helps a user identify, understand, install, run, reproduce, evaluate, cite, or reuse the artifact.

High-priority metadata includes:
- artifact identity: name, purpose, description
- access links: repository, paper, project page, model, dataset, demo
- reproducibility: installation, dependencies, environment, hardware, checkpoints, commands
- usage: training, inference, evaluation, configuration
- scientific context: task, method, benchmark, metrics, results, citation
- legal/reuse information: license, terms, attribution

Avoid extracting low-value metadata such as generic section headings, vague benefits, repeated examples, marketing claims without concrete values, or ordinary prose that does not name a reusable fact. Extract commands only when they define installation, setup, training, evaluation, inference, reproduction, or deployment steps. Do not extract duplicate or near-duplicate commands.

For each metadata item, return:
- field: a short snake_case name for the metadata type you identified
- value: the extracted value, as close to verbatim as possible
- normalized_value: apply ONLY these normalizations, nothing else:
  - trim surrounding whitespace/punctuation
  - collapse internal whitespace
  - do not shorten, expand, or abbreviate names (e.g. keep "MIT License" as "MIT License", not "MIT"; keep "Node.js 18" as-is, not "Node 18")
  If no normalization applies, repeat value exactly.
- evidence: a text span copied VERBATIM (character-for-character) from the artifact text, including original Markdown syntax (e.g. keep "## Installation" as-is, keep code fence contents as-is). Do not include the input line number prefixes. Do not paraphrase, reflow, merge non-adjacent sentences, or fix typos in the evidence field. This field will be programmatically checked as a substring of the source document after JSON string decoding, so JSON escaping of newlines and quotes is acceptable.
- evidence_location: line number or line range from the numbered input, optionally plus Markdown section/header title; use "unknown" only if not available
- confidence: one of "high", "medium", "low", using these criteria:
  - high: the value is named explicitly and unambiguously, with no plausible alternative reading
  - medium: the value is supported but requires minor interpretation (e.g. resolving an acronym used elsewhere in the doc, or a value split across adjacent sentences)
  - low: the text is genuinely ambiguous between two or more readings, or the evidence is indirect/implicit
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
}

Input text:
{input_text}
"""


REASONING_DELIMITER = "\n\n===== FINAL RESPONSE =====\n\n"
LOG_LOCK = threading.Lock()


def number_lines(text: str) -> str:
    return "\n".join(f"{line_no}: {line}" for line_no, line in enumerate(text.splitlines(), start=1))


def reasoning_extra_body_for_model(model: str) -> dict[str, Any]:
    model_name = model.lower()

    if "deepseek" in model_name:
        return {
            "include_reasoning": True,
            "reasoning": {"enabled": True, "effort": "xhigh", "exclude": False},
        }

    if "gpt-oss-120b" in model_name:
        return {
            "include_reasoning": True,
            "reasoning": {"enabled": True, "effort": "high", "exclude": False},
        }

    if 'qwen' in model_name:
        return {
            "include_reasoning": True,
            "reasoning": {"enabled": True, "exclude": False},
        }

    return {
        "include_reasoning": True,
        "reasoning": {"enabled": True, "exclude": False, "effort": "high", "mode": "pro"},
    }


def object_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "model_dump_json"):
        return json.loads(value.model_dump_json())
    if isinstance(value, dict):
        return value
    return {}

# def collect_reasoning_text(message: Any) -> tuple[str | None, Any | None]:
#     message_dict = object_to_dict(message)
#     reasoning = message_dict.get("reasoning") or message_dict.get("reasoning_content")
#     reasoning_details = message_dict.get("reasoning_details")

#     if reasoning is None and isinstance(reasoning_details, list):
#         text_parts = []
#         for item in reasoning_details:
#             if not isinstance(item, dict):
#                 continue
#             for key in ("text", "content", "summary"):
#                 value = item.get(key)
#                 if isinstance(value, str):
#                     text_parts.append(value)
#                 elif isinstance(value, list):
#                     text_parts.extend(
#                         entry.get("text")
#                         for entry in value
#                         if isinstance(entry, dict) and isinstance(entry.get("text"), str)
#                     )
#         reasoning = "\n".join(text_parts) if text_parts else None

#     return reasoning, reasoning_details

# def compose_raw_response(response_text: str, reasoning_text: str | None) -> str:
#     if not reasoning_text:
#         return response_text
#     return f"{reasoning_text.strip()}{REASONING_DELIMITER}{response_text.strip()}"


def normalize_json(content: str) -> dict[str, Any]:
    if not content:
        raise ValueError("Cannot normalize empty response content.")

    text = content.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        result = json.loads(text[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result).__name__}.")
    if "metadata" not in result:
        raise ValueError("Expected normalized JSON object to contain a 'metadata' key.")
    if not isinstance(result["metadata"], list):
        raise ValueError("Expected normalized JSON object's 'metadata' value to be a list.")

    return result


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def log_event(output_dir: Path, event: str, model: str, file: Path, **kwargs: Any) -> None:
    append_jsonl(
        output_dir / "_logs" / "metadata_extract_events.jsonl",
        {
            "event": event,
            "model": model,
            "file": str(file),
            "readme_filename": file.name,
            **kwargs,
        },
    )


def log_failure(output_dir: Path, model: str, file: Path, stage: str, exc: BaseException, **kwargs: Any) -> None:
    log_event(
        output_dir,
        "failure",
        model,
        file,
        stage=stage,
        error_type=type(exc).__name__,
        error=str(exc),
        traceback=traceback.format_exc(),
        **kwargs,
    )


def normalize_and_write_json(model: str, file: Path, output_dir: Path, cache_file: Path, json_file: Path) -> dict[str, Any] | None:
    try:
        message_dict = json.loads(cache_file.read_text(encoding="utf-8"))
        content = message_dict.get("content") or ""
        result = normalize_json(content)
        json_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log_event(output_dir, "json_normalized", model, file, cache_file=str(cache_file), json_file=str(json_file))
        return result
    except Exception as exc:
        log_failure(output_dir, model, file, "normalize_json", exc, cache_file=str(cache_file), json_file=str(json_file))
        return None


# metadata extraction function
def extract_metadata(client: OpenAI, model: str, file: Path, output_dir: Path):
    cache_file = output_dir / file.with_suffix('.txt').name
    json_file = output_dir / file.with_suffix('.json').name
    output_dir.mkdir(parents=True, exist_ok=True)

    if json_file.exists():
        logging.info(f'[Cache Hit] {str(file)}')
        log_event(output_dir, "cache_hit_json", model, file, json_file=str(json_file))
        return json.loads(json_file.read_text(encoding="utf-8"))

    if cache_file.exists():
        logging.info(f'[Raw Cache Hit] {str(file)}')
        log_event(output_dir, "cache_hit_raw", model, file, cache_file=str(cache_file))
        return normalize_and_write_json(model, file, output_dir, cache_file, json_file)

    logging.info(f'[Cache Miss] {str(file)}')
    log_event(output_dir, "cache_miss", model, file, cache_file=str(cache_file), json_file=str(json_file))
    doc_content = file.read_text(encoding="utf-8")
    numbered_doc_content = number_lines(doc_content)
    messages = [
        {"role": "user", "content": PROMPT.replace('{input_text}', numbered_doc_content)}
    ]
    extra_body = reasoning_extra_body_for_model(model)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            extra_body=extra_body if extra_body else None
        )
    except Exception as exc:
        log_failure(output_dir, model, file, "request", exc)
        raise
    message = completion.choices[0].message
    message_dict = object_to_dict(message)
    cache_file.write_text(json.dumps(message_dict, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    log_event(output_dir, "raw_response_saved", model, file, cache_file=str(cache_file))
    return normalize_and_write_json(model, file, output_dir, cache_file, json_file)


# main async function
def main(args):
    safe_model_name = args.model.replace('/', '_').replace(':', '_')
    result_dir = Path(f'./results/llm/{safe_model_name}')
    # 1. read all markdown files in the folder
    files = sorted([x for x in Path("results/readme_files").glob("*.md")])
    files += sorted([x for x in Path("results/readme_files").glob("*.rst")])

    if args.limit:
        files = files[:args.limit]

    print(f"Found {len(files)} markdown files")

    # 2. initialize AsyncOpenAI client and semaphore for concurrency control
    api_key = os.environ['OPENROUTER_API_KEY']
    if not api_key:
        raise Exception("Please ensure the correctness of api key")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=120,
    )
    if args.parallel == 1:
        for file in files:
            result = extract_metadata(client, args.model, file, result_dir)
            if result is None:
                print(f"FAILED_NORMALIZE {file.name}", file=sys.stderr)
            else:
                print(f"OK {file.name}")
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_file = {
                executor.submit(extract_metadata, client, args.model, file, result_dir): file
                for file in files
            }
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"FAILED {file.name}: {exc}", file=sys.stderr)
                    log_failure(result_dir, args.model, file, "worker", exc)
                else:
                    if result is None:
                        print(f"FAILED_NORMALIZE {file.name}", file=sys.stderr)
                    else:
                        print(f"OK {file.name}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    model_choices = [
        "qwen/qwen3.5-flash-02-23",
        "qwen/qwen3.6-flash",
        "qwen/qwen3.7-flash",

        "deepseek/deepseek-v4-flash",
        'deepseek/deepseek-v3.2',
        'deepseek/deepseek-chat-v3.1',

        'moonshotai/kimi-k3',
        # 'openai/gpt-oss-120b',
        # 'openai/gpt-oss-20b',

        # 'qwen/qwen3.5-9b',
        # 'qwen/qwen3.5-27b',

        # 'anthropic/claude-3-haiku',
        'anthropic/claude-sonnet-5',
        'openai/gpt-5.6-luna-pro',
        'google/gemini-3.6-flash',
        # 'google/gemini-3.5-flash',
        # 'google/gemini-2.5-flash',
    ]
    # Qwen, DeepSeek, Kimi, Claude, GPT, Gemini
    # open source: Qwen, DeepSeek, Kimi,
    # close source: Claude, GPT, Gemini

    # within family: (Qwen 3.5, Qwen 3.6),  (Qwen 3.5, Qwen 3.7),  (Qwen 3.6, Qwen 3.7), (Deepseek x, ..), ..
    # cross family:   (Qwen 3.5, DeepSeek 3.1),

    parser.add_argument("--model", choices=model_choices, required=False, default="deepseek/deepseek-v4-flash")
    parser.add_argument("--parallel", type=int, required=False, default=1)
    parser.add_argument("--limit", type=int, required=False, default=None)

    args = parser.parse_args()

    main(args)
