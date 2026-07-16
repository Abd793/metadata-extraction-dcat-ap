# #!/usr/bin/env bash
# set -euo pipefail

# cd "$(dirname "$0")"

# if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
#   echo "OPENROUTER_API_KEY is required for uncached requests." >&2
#   exit 2
# fi

# THREADS="${THREADS:-4}"

# python3 src/openrouter_metadata_extract.py \
#   --threads "$THREADS" \
#   --models \
#     openai/gpt-oss-120b \
#     qwen/qwen3.5-flash-02-23 \
#     deepseek/deepseek-v4-pro

python3 src/metadata_extract.py --threads 1 --models deepseek/deepseek-v4-flash
