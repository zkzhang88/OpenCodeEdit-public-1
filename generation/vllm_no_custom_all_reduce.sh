#!/usr/bin/env bash
set -euo pipefail

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_no_custom_all_reduce_cache}"

exec vllm "$@" --disable-custom-all-reduce --enforce-eager
