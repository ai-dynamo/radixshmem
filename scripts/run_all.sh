#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
BIN_DIR="$BUILD_DIR/bin"

# Per-target timeout (seconds). Override with RUN_ALL_TIMEOUT env var.
TIMEOUT="${RUN_ALL_TIMEOUT:-300}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

PASSED=0
FAILED=0
FAILURES=()

print_header() {
    echo ""
    echo -e "${CYAN}===== $1 =====${RESET}"
}

run_target() {
    local name="$1"
    local bin="$BIN_DIR/$name"
    if [[ ! -x "$bin" ]]; then
        echo -e "  ${YELLOW}[SKIP]${RESET} $name (binary not found)"
        return
    fi
    echo -e "  ${BOLD}[RUN]${RESET}  $name"
    local tmpout rc start end ms
    tmpout=$(mktemp)
    start=$(date +%s%N)
    rc=0
    timeout "$TIMEOUT" "$bin" > "$tmpout" 2>&1 || rc=$?
    end=$(date +%s%N)
    ms=$(( (end - start) / 1000000 ))
    sed 's/^/    /' "$tmpout"
    rm -f "$tmpout"
    if [[ $rc -eq 0 ]]; then
        echo -e "  ${GREEN}[PASS]${RESET} $name  (${ms}ms)"
        PASSED=$((PASSED + 1))
    elif [[ $rc -eq 124 ]]; then
        echo -e "  ${RED}[TIMEOUT]${RESET} $name  (exceeded ${TIMEOUT}s)"
        FAILED=$((FAILED + 1))
        FAILURES+=("$name")
    else
        echo -e "  ${RED}[FAIL]${RESET} $name  (${ms}ms, exit $rc)"
        FAILED=$((FAILED + 1))
        FAILURES+=("$name")
    fi
}

usage() {
    echo "Usage: $0 [--tests-only | --bench-only] [--timeout SECONDS]"
    exit 0
}

RUN_TESTS=1
RUN_BENCH=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tests-only)  RUN_BENCH=0; shift ;;
        --bench-only)  RUN_TESTS=0; shift ;;
        --timeout)     TIMEOUT="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *)             echo "Unknown option: $1"; usage ;;
    esac
done

# ── Build ──
print_header "Building (Release)"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev 2>&1 | tail -1
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -3
echo -e "${GREEN}Build succeeded.${RESET}"

# ── Collect targets ──
TESTS=()
BENCHES=()
for bin in "$BIN_DIR"/test_* "$BIN_DIR"/bench_*; do
    [[ -x "$bin" ]] || continue
    name="$(basename "$bin")"
    case "$name" in
        test_*)  TESTS+=("$name") ;;
        bench_*) BENCHES+=("$name") ;;
    esac
done

# ── Tests ──
if [[ $RUN_TESTS -eq 1 ]]; then
    print_header "Tests (${#TESTS[@]})"
    for t in "${TESTS[@]}"; do
        run_target "$t"
    done
fi

# ── Benchmarks ──
if [[ $RUN_BENCH -eq 1 ]]; then
    print_header "Benchmarks (${#BENCHES[@]})"
    for b in "${BENCHES[@]}"; do
        run_target "$b"
    done
fi

# ── Summary ──
TOTAL=$((PASSED + FAILED))
print_header "Summary"
echo -e "  Total: $TOTAL   ${GREEN}Passed: $PASSED${RESET}   ${RED}Failed: $FAILED${RESET}"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo -e "  ${RED}Failed targets:${RESET}"
    for f in "${FAILURES[@]}"; do
        echo "    - $f"
    done
    exit 1
fi
echo -e "  ${GREEN}All targets passed.${RESET}"
