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

# python3 src/metadata_extract.py --parallel 1 --model deepseek/deepseek-v4-flash
# python3 src/metadata_extract.py --parallel 1 --model qwen/qwen3.5-flash-02-23
# python3 src/metadata_extract.py --parallel 1 --model openai/gpt-oss-120b
# python3 src/metadata_extract.py --parallel 1 --model qwen/qwen3.5-flash-02-23 &
# python3 src/metadata_extract.py --parallel 1 --model qwen/qwen3.6-flash &
# python3 src/metadata_extract.py --parallel 1 --model qwen/qwen3.7-flash &

# python3 src/metadata_extract.py --parallel 1 --model deepseek/deepseek-v4-flash &
# python3 src/metadata_extract.py --parallel 1 --model deepseek/deepseek-v3.2 &
# python3 src/metadata_extract.py --parallel 1 --model deepseek/deepseek-chat-v3.1 &

# python3 src/metadata_extract.py --parallel 1 --model moonshotai/kimi-k3 &

# python3 src/metadata_extract.py --parallel 1 --model anthropic/claude-sonnet-5 &
# python3 src/metadata_extract.py --parallel 1 --model openai/gpt-5.6-luna-pro &
# python3 src/metadata_extract.py --parallel 1 --model google/gemini-3.6-flash &
