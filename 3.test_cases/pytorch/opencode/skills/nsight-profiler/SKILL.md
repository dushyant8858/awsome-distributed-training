# Nsight Profiler Skill

Profile distributed PyTorch training jobs with NVIDIA Nsight Systems and generate automated bottleneck analysis reports.

## Overview

This skill provides two scripts for GPU profiling of distributed training:

1. **`nsys_profile.sh`** — Wraps a training command with `nsys profile` using modern (2025/2026) best practices
2. **`nsys_analyze.py`** — Parses `.nsys-rep` reports and generates categorized bottleneck analysis

## Architecture

```
┌──────────────────────────────────────────────────┐
│  PyTorchJob Pod                                   │
│  ┌────────────────────────────────────────────┐   │
│  │ nsys_profile.sh                             │   │
│  │  ├── Auto-detects nsys binary               │   │
│  │  ├── Checks rank → profile or skip          │   │
│  │  ├── Configures traces, metrics, NVTX       │   │
│  │  └── exec nsys profile ... torchrun ...     │   │
│  └────────────────────────────────────────────┘   │
│                    ↓ produces                      │
│  /local/nsight-reports/report_rank0_*.nsys-rep    │
└──────────────────────────────────────────────────┘
                    ↓ kubectl cp
┌──────────────────────────────────────────────────┐
│  Analysis (local or on-cluster)                   │
│  ┌────────────────────────────────────────────┐   │
│  │ nsys_analyze.py                             │   │
│  │  ├── Runs nsys stats (6 report types)       │   │
│  │  ├── Classifies kernels into categories     │   │
│  │  ├── Identifies bottleneck type             │   │
│  │  ├── Generates recommendations              │   │
│  │  └── Outputs Markdown or JSON report        │   │
│  └────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

## nsys_profile.sh — Profiling Wrapper

### Modern Features (2025/2026 Best Practices)

| Feature | Flag | Purpose |
|---------|------|---------|
| PyTorch NVTX | `--pytorch=autograd-shapes-nvtx` | Auto-annotates all PyTorch ops (forward, backward, optimizer) with NVTX ranges including tensor shapes. No code changes needed. |
| Python Sampling | `--python-sampling=true` | Samples Python call stacks at 1kHz to identify CPU-side bottlenecks (e.g., `.item()` sync, data loading) |
| GPU HW Metrics | `--gpu-metrics-devices=all` | Collects SM utilization, memory bandwidth, tensor core activity at hardware level |
| CUDA Memory | `--cuda-memory-usage=true` | Tracks memory allocations/deallocations for leak detection |
| `--kill=none` | Keeps training alive | Unlike `--duration` default behavior, training continues after profiling window ends |
| SQLite Export | `--export=sqlite` | Auto-generates .sqlite for post-analysis |
| Stats | `--stats=true` | Auto-generates summary statistics |

### Configuration

All config via environment variables (suitable for K8s env injection):

```yaml
env:
  - name: NSYS_DELAY
    value: "30"          # Skip startup/warmup
  - name: NSYS_DURATION
    value: "90"          # Capture 90s of steady-state
  - name: NSYS_RANKS_TO_PROFILE
    value: "0"           # Only profile rank 0 (reduce overhead)
  - name: NSYS_GPU_METRICS
    value: "none"        # "all" for A100/H100/H200; "none" for A10G (g5)
  - name: NSYS_OUTPUT_DIR
    value: "/local/nsight-reports"
```

### Usage in PyTorchJob

```yaml
command:
  - /bin/bash
  - /scripts/nsys-profile.sh
  - --
  - /opt/conda/bin/torchrun
  - --nproc_per_node=1
  - --nnodes=2
  - /fsdp/train.py
  - --max_steps=50
  - ... training args ...
```

### Selective Rank Profiling

For large-scale runs, profile only specific ranks to reduce overhead:

```bash
# Profile only rank 0 and rank 1
NSYS_RANKS_TO_PROFILE=0,1

# Profile all ranks (default, fine for small scale)
NSYS_RANKS_TO_PROFILE=all
```

Non-profiled ranks execute the training command directly (zero overhead).

## nsys_analyze.py — Automated Analysis

### Kernel Classification

Kernels are auto-classified into categories:

| Category | Pattern Match | What It Means |
|----------|--------------|---------------|
| NCCL AllGather | `ncclDevKernel_AllGather` | FSDP parameter gathering |
| NCCL ReduceScatter | `ncclDevKernel_ReduceScatter` | FSDP gradient reduction |
| NCCL AllReduce | `ncclDevKernel_AllReduce` | DDP gradient sync / loss reduction |
| GEMM (Compute) | `gemm`, `cutlass`, `ampere` | Matrix multiplications (forward/backward) |
| Flash Attention | `flash_fwd`, `flash_bwd` | Attention computation |
| Optimizer | `multi_tensor_apply`, `adam` | Parameter updates |
| Memory Copy | `memcpy`, `CatArray` | Data movement |
| Activation | `silu`, `gelu`, `relu` | Activation functions |

### Bottleneck Classification

Based on kernel time distribution:

| Classification | Criteria | Typical Cause |
|---------------|----------|---------------|
| **Communication Bound** | NCCL > 40% of GPU time | Slow network (TCP vs EFA), small batch size |
| **Synchronization Bound** | cudaSync > 60% of CPU time | `.item()` calls, sync data transfers |
| **Compute Bound** | GEMM + Attn > 70% | Good! GPU is fully utilized |
| **Memory Transfer Bound** | H2D/D2H > 20% | Activation offloading, CPU data loading |
| **Mixed / Balanced** | None dominant | Multiple small bottlenecks |

### Usage

```bash
# Analyze single report
python nsys_analyze.py --reports report_rank0.nsys-rep

# Analyze directory of reports
python nsys_analyze.py --reports /tmp/nsight-reports/ --output /tmp/analysis.md

# JSON output for programmatic consumption
python nsys_analyze.py --reports /tmp/nsight-reports/ --format json --output /tmp/analysis.json

# Verbose mode (shows nsys commands)
python nsys_analyze.py --reports /tmp/nsight-reports/ -v

# Specify nsys binary
python nsys_analyze.py --reports report.nsys-rep --nsys-bin /opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys
```

### Output Format

The Markdown report includes:
1. **GPU Kernel Time Breakdown** — Category-level summary with top kernel names
2. **CUDA API Time** — Top 5 API calls by CPU time (identifies sync bottlenecks)
3. **Memory Transfers** — H2D/D2H/D2D volume and counts
4. **PyTorch Operation Breakdown** — NVTX ranges (if `--pytorch` was used)
5. **OS Runtime** — Thread scheduling, semaphores, I/O
6. **Cross-Worker Comparison** — Side-by-side metrics across ranks
7. **Recommendations** — Actionable optimization suggestions

## HyperPod EKS Deployment

### Prerequisites

- `nsys` is pre-installed on HyperPod nodes at `/opt/nvidia/nsight-systems/`
- Mount as hostPath volume — no container image changes needed

### Volume Mounts

```yaml
volumes:
  - name: nsight
    hostPath:
      path: /opt/nvidia/nsight-systems
      type: Directory
  - name: local
    hostPath:
      path: /mnt/k8s-disks/0
  - name: scripts
    configMap:
      name: nsight-scripts
      defaultMode: 0755

volumeMounts:
  - name: nsight
    mountPath: /nsight
    readOnly: true
  - name: local
    mountPath: /local
  - name: scripts
    mountPath: /scripts
```

### Complete PyTorchJob Example

See the YAML generator in this skill or use:

```bash
# 1. Create ConfigMap with profiling script
kubectl create configmap nsight-scripts \
  --from-file=nsys-profile.sh=src/nsys_profile.sh

# 2. Apply PyTorchJob (use restartPolicy: Never to avoid restart loops)
kubectl apply -f profiled-training-job.yaml

# 3. Wait for completion, copy reports
# If pods are still running:
kubectl cp <pod>:/local/nsight-reports/ /tmp/nsight-reports/

# If pods have completed (can't kubectl cp into completed pods):
# Create a helper busybox pod with hostPath mounted to the same node:
kubectl get pods -o wide | grep <job-name>  # find node name
kubectl run nsys-copy --image=busybox --restart=Never --overrides='
{
  "spec": {
    "nodeSelector": {"kubernetes.io/hostname": "<node-name>"},
    "volumes": [{"name": "local", "hostPath": {"path": "/mnt/k8s-disks/0"}}],
    "containers": [{"name": "copy", "image": "busybox", "command": ["sleep", "300"],
      "volumeMounts": [{"name": "local", "mountPath": "/local"}]}]
  }
}'
kubectl cp nsys-copy:/local/nsight-reports/ /tmp/nsight-reports/
kubectl delete pod nsys-copy

# 4. Analyze (on-cluster or locally)
python nsys_analyze.py --reports /tmp/nsight-reports/ --output /tmp/analysis.md
```

## Key Discoveries & Gotchas

1. **`--kill=none` is critical** — Without it, nsys terminates the training process when `--duration` expires. With `--kill=none`, training continues and nsys writes the report file.

2. **`--pytorch=autograd-shapes-nvtx` requires nsys >= 2024.5** — Earlier versions don't support this flag. The 2025.6.1 version on HyperPod supports it.

3. **`--gpu-metrics-devices` NOT supported on A10G (g5)** — The A10G GPU (Ampere GA102) requires elevated privileges for GPU metrics collection. You get: `Insufficient privilege, see https://developer.nvidia.com/ERR_NVGPUCTRPERM`. Set `NSYS_GPU_METRICS=none` for g5 instances. Works on A100/H100/H200.

4. **`--python-sampling` + `--sample=none`** is the right combo — Python sampling gives call stacks without the overhead of native CPU IP sampling.

5. **Report file size** — With PyTorch NVTX + Python sampling + CUDA memory tracking, expect ~80-90 MB per rank per 120s window. Without these features, ~25-40 MB. Plan storage accordingly for large-scale profiling.

6. **nsys on HyperPod nodes** — Pre-installed at `/opt/nvidia/nsight-systems/<version>/`. The `target-linux-x64/nsys` binary is the CLI profiler. The `bin/nsys` is a wrapper that also works.

7. **Selective rank profiling** — For > 8 GPUs, profile only 1-2 ranks. The wrapper script's `NSYS_RANKS_TO_PROFILE` env var handles this with zero overhead on non-profiled ranks.

8. **`restartPolicy: Never`** is recommended for profiling jobs. With `--kill=none` training continues after profiling, but if the job is configured with `OnFailure` restart, Kubernetes may restart pods unnecessarily. `Never` ensures a clean single run.

## nsys recipe Integration

For deeper analysis, the `.nsys-rep` files can be processed with `nsys recipe`:

```bash
NSYS=/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys

# NCCL communication summary
$NSYS recipe nccl_sum --input report.nsys-rep

# NCCL + GPU overlap analysis (compute-communication overlap)
$NSYS recipe nccl_gpu_overlap_trace --input report.nsys-rep

# GPU idle gaps
$NSYS recipe gpu_gaps --input report.nsys-rep

# CUDA kernel pacing (identifies stragglers)
$NSYS recipe cuda_gpu_kern_pace --input report.nsys-rep

# GPU time utilization heatmap
$NSYS recipe cuda_gpu_time_util_map --input report.nsys-rep
```

## Files

```
nsight-profiler/
├── SKILL.md              # This file
├── skill.yaml            # Skill metadata
└── src/
    ├── nsys_profile.sh   # Profiling wrapper script
    └── nsys_analyze.py   # Automated analysis script
```
