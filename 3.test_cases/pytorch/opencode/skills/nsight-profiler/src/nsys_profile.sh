#!/usr/bin/env bash
# nsys_profile.sh — Modern Nsight Systems profiling wrapper for distributed PyTorch training
#
# Usage: nsys_profile.sh [options] -- <training_command>
#
# This script wraps a training command with nsys profile, using 2025/2026 best practices:
#   - PyTorch autograd NVTX annotations (--pytorch=autograd-shapes-nvtx)
#   - Python call stack sampling (--python-sampling)
#   - GPU hardware metrics (--gpu-metrics-devices)
#   - CUDA memory tracking (--cuda-memory-usage)
#   - Proper NCCL/CUDA/NVTX/OSRT tracing
#   - Smart output naming with rank and hostname
#
# Environment variables (set by PyTorchJob or torchrun):
#   RANK          — Global rank of this worker
#   LOCAL_RANK    — Local rank on this node
#   WORLD_SIZE    — Total number of workers
#   HOSTNAME      — Node hostname
#
# Configuration via environment variables:
#   NSYS_DELAY         — Seconds to wait before collecting (default: 30)
#   NSYS_DURATION      — Seconds to collect for (default: 90)
#   NSYS_OUTPUT_DIR    — Directory for .nsys-rep files (default: /local/nsight-reports)
#   NSYS_TRACE         — Comma-separated trace APIs (default: cuda,nvtx,osrt)
#   NSYS_PYTORCH_MODE  — PyTorch annotation mode (default: autograd-shapes-nvtx)
#   NSYS_PYTHON_SAMPLE — Enable Python sampling (default: true)
#   NSYS_GPU_METRICS   — Collect GPU HW metrics (default: none; set "all" for A100/H100/H200)
#   NSYS_CUDA_MEMORY   — Track CUDA memory usage (default: true)
#   NSYS_SAMPLE        — CPU sampling mode (default: none for low overhead)
#   NSYS_RANKS_TO_PROFILE — Comma-separated ranks to profile, or "all" (default: all)
#   NSYS_BIN           — Path to nsys binary (auto-detected)
#   NSYS_EXTRA_ARGS    — Additional nsys arguments
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
NSYS_DELAY="${NSYS_DELAY:-30}"
NSYS_DURATION="${NSYS_DURATION:-90}"
NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-/local/nsight-reports}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_PYTORCH_MODE="${NSYS_PYTORCH_MODE:-autograd-shapes-nvtx}"
NSYS_PYTHON_SAMPLE="${NSYS_PYTHON_SAMPLE:-true}"
NSYS_GPU_METRICS="${NSYS_GPU_METRICS:-none}"
NSYS_CUDA_MEMORY="${NSYS_CUDA_MEMORY:-true}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
NSYS_RANKS_TO_PROFILE="${NSYS_RANKS_TO_PROFILE:-all}"
NSYS_BIN="${NSYS_BIN:-}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"

# ── Parse arguments ───────────────────────────────────────────────────────────
TRAINING_CMD=()
PARSING_OPTS=true

while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        PARSING_OPTS=false
        shift
        continue
    fi
    if $PARSING_OPTS; then
        case "$1" in
            --delay=*) NSYS_DELAY="${1#*=}"; shift ;;
            --duration=*) NSYS_DURATION="${1#*=}"; shift ;;
            --output-dir=*) NSYS_OUTPUT_DIR="${1#*=}"; shift ;;
            --ranks=*) NSYS_RANKS_TO_PROFILE="${1#*=}"; shift ;;
            --help|-h)
                echo "Usage: nsys_profile.sh [--delay=N] [--duration=N] [--output-dir=DIR] [--ranks=0,1|all] -- <command>"
                exit 0 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    else
        TRAINING_CMD+=("$1")
        shift
    fi
done

if [[ ${#TRAINING_CMD[@]} -eq 0 ]]; then
    echo "ERROR: No training command specified. Use: nsys_profile.sh [options] -- <command>"
    exit 1
fi

# ── Auto-detect nsys binary ──────────────────────────────────────────────────
find_nsys() {
    # Check explicit path first
    if [[ -n "$NSYS_BIN" ]] && [[ -x "$NSYS_BIN" ]]; then
        echo "$NSYS_BIN"
        return
    fi
    # Check common HyperPod / DLAMI paths (newest first)
    local search_paths=(
        "/opt/nvidia/nsight-systems/2025.6.1/bin/nsys"
        "/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys"
        "/nsight/2025.6.1/bin/nsys"
    )
    # Also search via glob for any version
    for p in /opt/nvidia/nsight-systems/*/bin/nsys /opt/nvidia/nsight-systems/*/target-linux-x64/nsys; do
        search_paths+=("$p")
    done
    # Check mounted paths (from hostPath volume)
    for p in /nsight/*/bin/nsys /nsight/*/target-linux-x64/nsys; do
        search_paths+=("$p")
    done
    for p in "${search_paths[@]}"; do
        if [[ -x "$p" ]]; then
            echo "$p"
            return
        fi
    done
    # Fall back to PATH
    if command -v nsys &>/dev/null; then
        command -v nsys
        return
    fi
    echo ""
}

NSYS_BIN=$(find_nsys)
if [[ -z "$NSYS_BIN" ]]; then
    echo "ERROR: nsys binary not found. Set NSYS_BIN or mount nsight-systems volume."
    echo "  HyperPod nodes have it at /opt/nvidia/nsight-systems/"
    echo "  Mount as hostPath volume and set NSYS_BIN=/nsight/<version>/bin/nsys"
    exit 1
fi

echo "=== Nsight Systems Profiler Wrapper ==="
echo "nsys binary: $NSYS_BIN"
$NSYS_BIN --version
echo ""

# ── Determine if this rank should be profiled ────────────────────────────────
RANK="${RANK:-${SLURM_PROCID:-0}}"
LOCAL_RANK="${LOCAL_RANK:-${SLURM_LOCALID:-0}}"

should_profile() {
    if [[ "$NSYS_RANKS_TO_PROFILE" == "all" ]]; then
        return 0
    fi
    IFS=',' read -ra PROFILE_RANKS <<< "$NSYS_RANKS_TO_PROFILE"
    for r in "${PROFILE_RANKS[@]}"; do
        if [[ "$RANK" == "$r" ]]; then
            return 0
        fi
    done
    return 1
}

if ! should_profile; then
    echo "Rank $RANK not in profile list ($NSYS_RANKS_TO_PROFILE). Running without profiling."
    exec "${TRAINING_CMD[@]}"
fi

# ── Setup output directory ───────────────────────────────────────────────────
mkdir -p "$NSYS_OUTPUT_DIR"

# Build output filename: report_rank<RANK>_<hostname>_<timestamp>
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${NSYS_OUTPUT_DIR}/report_rank${RANK}_$(hostname)_${TIMESTAMP}"

# ── Build nsys command ───────────────────────────────────────────────────────
NSYS_CMD=(
    "$NSYS_BIN" profile
    --trace="$NSYS_TRACE"
    --sample="$NSYS_SAMPLE"
    --delay="$NSYS_DELAY"
    --duration="$NSYS_DURATION"
    --output="$OUTPUT_FILE"
    --force-overwrite=true
    --kill=none               # Don't kill the process when duration expires
    --stop-on-exit=true       # Stop collection when process exits
)

# PyTorch autograd NVTX annotations (requires nsys >= 2024.5)
if [[ -n "$NSYS_PYTORCH_MODE" ]] && [[ "$NSYS_PYTORCH_MODE" != "none" ]]; then
    NSYS_CMD+=(--pytorch="$NSYS_PYTORCH_MODE")
fi

# Python call stack sampling
if [[ "$NSYS_PYTHON_SAMPLE" == "true" ]]; then
    NSYS_CMD+=(--python-sampling=true --python-sampling-frequency=1000)
fi

# GPU hardware metrics (SM utilization, memory bandwidth, etc.)
if [[ -n "$NSYS_GPU_METRICS" ]] && [[ "$NSYS_GPU_METRICS" != "none" ]]; then
    NSYS_CMD+=(--gpu-metrics-devices="$NSYS_GPU_METRICS")
fi

# CUDA memory usage tracking
if [[ "$NSYS_CUDA_MEMORY" == "true" ]]; then
    NSYS_CMD+=(--cuda-memory-usage=true)
fi

# Export stats after collection
NSYS_CMD+=(--stats=true --export=sqlite)

# Extra arguments
if [[ -n "$NSYS_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    NSYS_CMD+=($NSYS_EXTRA_ARGS)
fi

# Append training command
NSYS_CMD+=("${TRAINING_CMD[@]}")

# ── Print configuration ─────────────────────────────────────────────────────
echo "Profiling Configuration:"
echo "  Rank:          $RANK (Local: $LOCAL_RANK)"
echo "  Delay:         ${NSYS_DELAY}s"
echo "  Duration:      ${NSYS_DURATION}s"
echo "  Output:        ${OUTPUT_FILE}.nsys-rep"
echo "  Traces:        $NSYS_TRACE"
echo "  PyTorch NVTX:  $NSYS_PYTORCH_MODE"
echo "  Python sample: $NSYS_PYTHON_SAMPLE"
echo "  GPU metrics:   $NSYS_GPU_METRICS"
echo "  CUDA memory:   $NSYS_CUDA_MEMORY"
echo ""
echo "Command: ${NSYS_CMD[*]}"
echo "========================================="
echo ""

# ── Execute ──────────────────────────────────────────────────────────────────
exec "${NSYS_CMD[@]}"
