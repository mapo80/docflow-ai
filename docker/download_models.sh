#!/usr/bin/env bash
set -euo pipefail

LLM_URL="${1:-}"
EMB_URL="${2:-}"
LLM_SHA="${3:-}"
EMB_SHA="${4:-}"
OUT_DIR="${5:-/models}"

mkdir -p "${OUT_DIR}"

download() {
  local url="$1"
  local out="$2"
  local sha="$3"

  if [ -z "${url}" ]; then
    echo "Skip download for ${out} (no URL provided)"
    return 0
  fi

  echo "Downloading ${url} -> ${out}"
  curl -L --fail --retry 5 --retry-delay 2 -o "${out}.partial" "${url}"
  mv "${out}.partial" "${out}"

  if [ -n "${sha}" ]; then
    echo "${sha}  ${out}" | sha256sum -c -
  fi
}

download "${LLM_URL}" "${OUT_DIR}/llm.gguf" "${LLM_SHA}"
download "${EMB_URL}" "${OUT_DIR}/embeddings.gguf" "${EMB_SHA}"

echo "Model download step complete."
