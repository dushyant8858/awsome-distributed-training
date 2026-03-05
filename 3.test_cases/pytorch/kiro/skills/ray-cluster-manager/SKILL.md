# Ray Cluster Manager

A skill for managing Ray and KubeRay clusters on EKS for distributed training workloads.

## Overview

This skill provides complete management of Ray clusters on Kubernetes using KubeRay. It handles installation, configuration, and lifecycle management of Ray clusters with support for EFA (Elastic Fabric Adapter) for high-performance networking.

## Features

- **KubeRay Operator Management**: Install and verify KubeRay operator
- **RayCluster Lifecycle**: Create, scale, and delete Ray clusters
- **EFA Support**: Automatic EFA configuration for high-performance networking
- **YAML Generation**: Generate production-ready RayCluster configurations
- **Status Monitoring**: Check cluster readiness and health

## Usage

### Check if KubeRay is Installed

```python
from ray_cluster_manager.src.ray_manager import check_kuberay_installed

if check_kuberay_installed():
    print("KubeRay is ready")
else:
    print("KubeRay needs to be installed")
```

### Install KubeRay Operator

```python
from ray_cluster_manager.src.ray_manager import install_kuberay

# Install KubeRay on your EKS cluster
install_kuberay(
    cluster_name='my-eks-cluster',
    region='us-west-2'
)
```

### Create a RayCluster

```python
from ray_cluster_manager.src.ray_manager import create_raycluster

# Create a basic RayCluster
config = {
    'job_name': 'distributed-training',
    'image_uri': '123456789012.dkr.ecr.us-west-2.amazonaws.com/ray-training:latest',
    'num_nodes': 4,
    'use_efa': False,
    'checkpoint_pvc': None
}

create_raycluster(config)
```

### Create RayCluster with EFA

```python
from ray_cluster_manager.src.ray_manager import create_raycluster

# Create RayCluster with EFA for high-performance networking
config = {
    'job_name': 'distributed-training-efa',
    'image_uri': '123456789012.dkr.ecr.us-west-2.amazonaws.com/ray-training:latest',
    'num_nodes': 4,
    'use_efa': True,
    'checkpoint_pvc': 'fsx-claim'
}

create_raycluster(config)
```

### Check Cluster Status

```python
from ray_cluster_manager.src.ray_manager import get_raycluster_status

status = get_raycluster_status('distributed-training')
print(f"Cluster status: {status}")
```

### Delete a RayCluster

```python
from ray_cluster_manager.src.ray_manager import delete_raycluster

delete_raycluster('distributed-training')
```

### Generate YAML Only

```python
from ray_cluster_manager.src.ray_manager import generate_raycluster_yaml

yaml_content = generate_raycluster_yaml(
    job_name='my-training',
    image_uri='rayproject/ray:latest',
    num_nodes=2,
    use_efa=True,
    checkpoint_pvc='fsx-pvc'
)

with open('raycluster.yaml', 'w') as f:
    f.write(yaml_content)
```

## Configuration Options

### RayCluster Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_name` | str | Yes | Name of the RayCluster |
| `image_uri` | str | Yes | Container image URI |
| `num_nodes` | int | Yes | Total nodes (1 head + N-1 workers) |
| `use_efa` | bool | No | Enable EFA networking |
| `checkpoint_pvc` | str | No | PVC name for checkpoint storage |
| `head_cpu` | int | No | Head node CPU (default: 4) |
| `head_memory` | str | No | Head node memory (default: 16Gi) |
| `worker_cpu` | int | No | Worker CPU per node (default: 16) |
| `worker_memory` | str | No | Worker memory per node (default: 64Gi) |
| `worker_gpu` | int | No | GPUs per worker (default: 8) |
| `namespace` | str | No | Kubernetes namespace (default: default) |

### EFA Configuration

When `use_efa=True`, the skill automatically:
- Adds EFA security context (IPC_LOCK, SYS_RESOURCE)
- Configures NCCL to use the OFI (EFA) plugin via `NCCL_NET=ofi`
- Sets `LD_LIBRARY_PATH` to include the aws-ofi-nccl plugin
- Sets up EFA device mounts
- Adds appropriate resource limits
- Configures `FI_EFA_USE_DEVICE_RDMA=0` for g5 instances (change to `1` for p4d/p5)

## Environment Variables

The skill automatically configures these environment variables for EFA:

```bash
# CRITICAL: Force NCCL to use OFI plugin (without this, NCCL silently falls back to TCP sockets)
NCCL_NET=ofi
LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# EFA libfabric settings
FI_PROVIDER=efa
FI_EFA_USE_DEVICE_RDMA=0        # 0 for g5 (no GPUDirect), 1 for p4d/p5
FI_EFA_FORK_SAFE=1
FI_EFA_ENABLE_SHM_TRANSFER=1

# NCCL settings
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=^docker,lo,veth
NCCL_TIMEOUT=1800
NCCL_PROTO=simple               # Required when FI_EFA_USE_DEVICE_RDMA=0
```

### ⚠️ Common EFA Pitfall

Without `NCCL_NET=ofi`, NCCL will silently fall back to TCP sockets even when:
- EFA device is present and ACTIVE
- `FI_PROVIDER=efa` is set
- `aws-ofi-nccl` plugin is installed

**Symptom**: EFA counters show 0 bytes (`/sys/class/infiniband/*/ports/*/counters/port_rcv_data`), all traffic goes through eth0.

**Verification**:
```bash
# Check EFA counters (should be non-zero if EFA is working)
cat /sys/class/infiniband/rdmap*/ports/1/counters/port_rcv_data

# Check NCCL init logs for "OFI" or "NET/OFI"
grep -i "ofi\|efa\|net/" /tmp/ray/session_*/logs/worker*.err
```

## Requirements

- Python 3.7+
- kubectl configured with EKS cluster access
- AWS CLI (for EKS cluster operations)
- Helm 3 (for KubeRay installation)

## Architecture

```
RayCluster
├── Head Node
│   ├── Ray Dashboard (port 8265)
│   ├── Ray GCS Server
│   └── Autoscaler
└── Worker Nodes (N-1)
    ├── Ray Workers
    └── GPU/EFA Resources
```

## Error Handling

All functions raise `RayClusterError` on failure with descriptive messages:

```python
from ray_cluster_manager.src.ray_manager import RayClusterError, create_raycluster

try:
    create_raycluster(config)
except RayClusterError as e:
    print(f"Failed to create cluster: {e}")
```

## Examples

### Complete Workflow

```python
from ray_cluster_manager.src.ray_manager import (
    check_kuberay_installed,
    install_kuberay,
    create_raycluster,
    get_raycluster_status,
    delete_raycluster
)

# 1. Check/install KubeRay
if not check_kuberay_installed():
    install_kuberay('my-cluster', 'us-west-2')

# 2. Create cluster
config = {
    'job_name': 'training-job',
    'image_uri': 'my-registry/ray-ml:latest',
    'num_nodes': 4,
    'use_efa': True,
    'checkpoint_pvc': 'fsx-claim',
    'worker_gpu': 8
}

create_raycluster(config)

# 3. Wait for readiness
import time
while True:
    status = get_raycluster_status('training-job')
    if status == 'ready':
        break
    time.sleep(10)

# 4. Use cluster...

# 5. Cleanup
delete_raycluster('training-job')
```

## Common Issues

### ⚠️ CRITICAL: Do NOT use `ray job submit` for Multi-GPU Training

**Problem**: Using `ray job submit` to launch training jobs isolates them from the Ray cluster's GPU resources. The job will fail with "0 GPUs available" even though the cluster has GPUs.

**Symptoms**:
- Training fails with `ValueError: Total available GPUs 0 is less than total desired GPUs N`
- `ray status` shows GPUs available but job can't see them
- Job runs but doesn't utilize GPUs

**Root Cause**: Ray jobs run in isolated processes that don't inherit the cluster's resource pool.

**Solution**: Run training directly in the head pod using `kubectl exec`:

```bash
# ❌ WRONG - Don't use this for multi-GPU training
ray job submit --working-dir /workspace -- python3 train.py

# ✅ CORRECT - Use kubectl exec instead
kubectl exec <head-pod> -- bash -c 'cd /workspace && python3 train.py'
```

**Verification**: After starting training, verify GPUs are being used:
```bash
# Check Ray resource allocation
kubectl exec <head-pod> -- ray status

# Check GPU utilization
kubectl exec <head-pod> -- nvidia-smi
```

## Troubleshooting

### KubeRay Not Found
```bash
# Verify CRD exists
kubectl get crd rayclusters.ray.io
```

### Cluster Not Ready
```bash
# Check pod status
kubectl get pods -l ray.io/cluster=<job-name>

# Check events
kubectl describe raycluster <job-name>
```

### EFA Issues
```bash
# Verify EFA devices
kubectl exec -it <pod> -- ibstat

# Check NCCL settings
kubectl exec -it <pod> -- env | grep NCCL
```
