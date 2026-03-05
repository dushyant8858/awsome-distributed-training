# K8s Cluster Manager Skill

A comprehensive Kubernetes cluster health checker and validator for training workloads on EKS.

## Overview

This skill provides tools to validate and monitor Kubernetes clusters, specifically designed for distributed training workloads. It supports GPU and EFA (Elastic Fabric Adapter) validation, addon status checks, and comprehensive resource reporting.

## Features

- **Cluster Health Checks**: Verify EKS cluster accessibility, API server connectivity, and node health
- **GPU Validation**: Check NVIDIA device plugin installation and GPU node capacity
- **EFA Validation**: Verify EFA device availability for high-performance networking
- **Addon Status**: Monitor Kubernetes addons like Kubeflow, KubeRay, CoreDNS
- **Resource Reporting**: Detailed CPU, GPU, memory, and EFA capacity per node

## ⚠️ EFA Validation: Device Present != NCCL Using It

### Verifying NCCL Is Actually Using EFA

An EFA device being present and `ACTIVE` does **NOT** mean NCCL is using it. NCCL can silently fall back to TCP sockets. To verify:

**Check NCCL logs** (the definitive test):
```bash
# Look for NET/OFI = EFA is active, NET/Socket = TCP fallback
kubectl exec <head-pod> -- grep -i 'NET/OFI\|NET/Socket' /tmp/ray/session_latest/logs/worker*.out
```

- `NCCL INFO NET/OFI Selected provider is efa` = EFA working
- `NCCL INFO NET/Socket` = Fallen back to TCP (bad!)

### Common EFA Misconfiguration

EFA device is present and ACTIVE but NCCL falls back to TCP sockets because the aws-ofi-nccl plugin isn't loaded.

**Required environment variables:**

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_NET` | `ofi` | **CRITICAL**: Forces NCCL to use OFI plugin |
| `LD_LIBRARY_PATH` | `/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:...` | **CRITICAL**: NCCL must find the plugin |
| `FI_PROVIDER` | `efa` | Tells libfabric to use EFA |
| `FI_EFA_USE_DEVICE_RDMA` | `0` (g5) or `1` (p4d/p5) | Enable/disable GPUDirect RDMA |
| `FI_EFA_FORK_SAFE` | `1` | Required for multi-process |
| `NCCL_PROTO` | `simple` | Required when RDMA=0 |

### Instance Type Differences

| Instance | GPU | GPUDirect RDMA | `FI_EFA_USE_DEVICE_RDMA` | Notes |
|----------|-----|----------------|--------------------------|-------|
| g5 (A10G) | 1 per node | No | `0` | CPU bounce buffer, simple protocol |
| p4d (A100) | 8 per node | Yes | `1` | Full RDMA, all protocols |
| p5 (H100) | 8 per node | Yes | `1` | Full RDMA, all protocols |

## Prerequisites

- Python 3.7+
- `kubectl` installed and in PATH
- `aws` CLI installed and configured
- Access to target EKS cluster

## Installation

The skill is self-contained and requires no external dependencies beyond Python's standard library.

```bash
# Clone or copy the skill to your opencode skills directory
cp -r k8s-cluster-manager ~/.config/opencode/skills/
```

## Quick Start

### Python API

```python
from k8s_cluster_manager.src.check_cluster import (
    check_cluster_health,
    check_gpu_availability,
    check_efa_availability,
    check_addon_status,
    get_node_resources,
    run_full_cluster_check
)

# Check overall cluster health
status = check_cluster_health('my-cluster', region='us-west-2')
print(f"Cluster status: {status['status']}")

# Check GPU availability
gpu_info = check_gpu_availability()
print(f"GPUs available: {gpu_info['total_gpu_capacity']}")

# Check EFA availability
efa_info = check_efa_availability()
print(f"EFA devices: {efa_info['total_efa_capacity']}")

# Check addon status
kubeflow_status = check_addon_status('kubeflow')
print(f"Kubeflow running: {kubeflow_status['running']}")

# Get detailed node resources
resources = get_node_resources()
for node in resources['nodes']:
    print(f"{node['name']}: {node['capacity']['cpu']} CPU, {node['capacity']['gpu']} GPU")

# Run comprehensive check
full_report = run_full_cluster_check('my-cluster', region='us-west-2')
```

### Command Line Interface

```bash
# Check cluster health
python -m k8s_cluster_manager.src.check_cluster my-cluster --region us-west-2

# Run full comprehensive check
python -m k8s_cluster_manager.src.check_cluster my-cluster --region us-west-2 --full

# Check specific components
python -m k8s_cluster_manager.src.check_cluster my-cluster --check gpu
python -m k8s_cluster_manager.src.check_cluster my-cluster --check efa
python -m k8s_cluster_manager.src.check_cluster my-cluster --check resources

# Check addon status
python -m k8s_cluster_manager.src.check_cluster my-cluster \
    --check addons --addon-name kubeflow

# Output formats
python -m k8s_cluster_manager.src.check_cluster my-cluster --output json
python -m k8s_cluster_manager.src.check_cluster my-cluster --output yaml
python -m k8s_cluster_manager.src.check_cluster my-cluster --output text

# Verbose logging
python -m k8s_cluster_manager.src.check_cluster my-cluster -v
```

## API Reference

### `check_cluster_health(cluster_name, region=None)`

Performs comprehensive health check on an EKS cluster.

**Parameters:**
- `cluster_name` (str): Name of the EKS cluster
- `region` (str, optional): AWS region

**Returns:**
```python
{
    'cluster_name': str,
    'status': str,  # 'healthy', 'degraded', 'unhealthy'
    'accessible': bool,
    'api_server': bool,
    'node_count': int,
    'ready_nodes': int,
    'issues': List[str],
    'timestamp': str,
    'region': str
}
```

**Example:**
```python
from k8s_cluster_manager.src.check_cluster import check_cluster_health

status = check_cluster_health('training-cluster', region='us-east-1')
if status['status'] == 'healthy':
    print(f"Cluster is healthy with {status['ready_nodes']}/{status['node_count']} nodes ready")
else:
    print("Issues found:")
    for issue in status['issues']:
        print(f"  - {issue}")
```

### `check_gpu_availability()`

Checks GPU availability and capacity in the cluster.

**Returns:**
```python
{
    'gpu_available': bool,
    'device_plugin_installed': bool,
    'gpu_nodes': List[{
        'name': str,
        'capacity': int,
        'allocatable': int,
        'instance_type': str
    }],
    'total_gpu_capacity': int,
    'total_gpu_allocatable': int,
    'gpu_pods_running': int,
    'timestamp': str,
    'issues': List[str]
}
```

**Example:**
```python
from k8s_cluster_manager.src.check_cluster import check_gpu_availability

gpu_info = check_gpu_availability()
if gpu_info['gpu_available']:
    print(f"Found {len(gpu_info['gpu_nodes'])} GPU nodes")
    print(f"Total GPU capacity: {gpu_info['total_gpu_capacity']}")
    print(f"GPUs currently allocatable: {gpu_info['total_gpu_allocatable']}")
else:
    print("GPU not available:")
    for issue in gpu_info['issues']:
        print(f"  - {issue}")
```

### `check_efa_availability()`

Checks EFA (Elastic Fabric Adapter) availability on nodes.

**Returns:**
```python
{
    'efa_available': bool,
    'device_plugin_installed': bool,
    'efa_nodes': List[{
        'name': str,
        'capacity': int,
        'instance_type': str
    }],
    'total_efa_capacity': int,
    'timestamp': str,
    'issues': List[str]
}
```

**Example:**
```python
from k8s_cluster_manager.src.check_cluster import check_efa_availability

efa_info = check_efa_availability()
if efa_info['efa_available']:
    print(f"EFA is available on {len(efa_info['efa_nodes'])} nodes")
    for node in efa_info['efa_nodes']:
        print(f"  {node['name']}: {node['capacity']} EFA devices")
```

### `check_addon_status(addon_name, namespace=None)`

Checks the status of a Kubernetes addon or component.

**Parameters:**
- `addon_name` (str): Name of the addon (e.g., 'kubeflow', 'kuberay', 'nvidia-device-plugin')
- `namespace` (str, optional): Namespace where addon is installed

**Supported Addons:**
- `kubeflow` - Kubeflow operator
- `kuberay` - KubeRay operator
- `nvidia-device-plugin` - NVIDIA GPU device plugin
- `aws-efa` - AWS EFA device plugin
- `coredns` - CoreDNS
- `kube-proxy` - kube-proxy

**Returns:**
```python
{
    'addon_name': str,
    'installed': bool,
    'running': bool,
    'pods': List[{
        'name': str,
        'namespace': str,
        'status': str,
        'ready': bool
    }],
    'services': List[{
        'name': str,
        'namespace': str,
        'type': str,
        'cluster_ip': str
    }],
    'deployments': List[{
        'name': str,
        'namespace': str,
        'replicas': int,
        'ready_replicas': int
    }],
    'timestamp': str,
    'issues': List[str]
}
```

**Example:**
```python
from k8s_cluster_manager.src.check_cluster import check_addon_status

# Check Kubeflow
kubeflow = check_addon_status('kubeflow')
print(f"Kubeflow installed: {kubeflow['installed']}")
print(f"Kubeflow running: {kubeflow['running']}")

# Check KubeRay
kuberay = check_addon_status('kuberay')
if kuberay['installed']:
    print(f"KubeRay has {len(kuberay['pods'])} pods")
```

### `get_node_resources()`

Gets detailed resource information for all nodes.

**Returns:**
```python
{
    'nodes': List[{
        'name': str,
        'instance_type': str,
        'role': str,
        'zone': str,
        'region': str,
        'capacity': {
            'cpu': str,
            'memory': str,
            'gpu': int,
            'efa': int,
            'pods': int
        },
        'allocatable': {
            'cpu': str,
            'memory': str,
            'gpu': int,
            'pods': int
        },
        'conditions': Dict[str, str],
        'ready': bool
    }],
    'total_cpu_capacity': int,
    'total_cpu_allocatable': int,
    'total_memory_capacity': str,
    'total_memory_allocatable': str,
    'total_gpu_capacity': int,
    'total_gpu_allocatable': int,
    'timestamp': str,
    'issues': List[str]
}
```

**Example:**
```python
from k8s_cluster_manager.src.check_cluster import get_node_resources

resources = get_node_resources()
print(f"Total nodes: {len(resources['nodes'])}")
print(f"Total CPU capacity: {resources['total_cpu_capacity']}")
print(f"Total GPU capacity: {resources['total_gpu_capacity']}")

for node in resources['nodes']:
    status = "Ready" if node['ready'] else "Not Ready"
    print(f"{node['name']} ({node['instance_type']}): {status}")
    print(f"  CPU: {node['allocatable']['cpu']}/{node['capacity']['cpu']}")
    print(f"  GPU: {node['allocatable']['gpu']}/{node['capacity']['gpu']}")
```

### `run_full_cluster_check(cluster_name, region=None)`

Runs a comprehensive check of the cluster including all components.

**Parameters:**
- `cluster_name` (str): Name of the EKS cluster
- `region` (str, optional): AWS region

**Returns:**
```python
{
    'cluster_name': str,
    'cluster_health': Dict,
    'gpu_availability': Dict,
    'efa_availability': Dict,
    'node_resources': Dict,
    'addons': Dict[str, Dict],
    'overall_status': str,
    'timestamp': str
}
```

**Example:**
```python
from k8s_cluster_manager.src.check_cluster import run_full_cluster_check
import json

report = run_full_cluster_check('my-cluster', region='us-west-2')

# Save report to file
with open('cluster-report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Print summary
print(f"Overall Status: {report['overall_status']}")
print(f"Nodes: {report['node_resources']['total_cpu_capacity']} CPU cores")
print(f"GPUs: {report['gpu_availability']['total_gpu_capacity']}")
print(f"EFAs: {report['efa_availability']['total_efa_capacity']}")
```

## Error Handling

All functions include comprehensive error handling and return structured results even when checks fail:

```python
from k8s_cluster_manager.src.check_cluster import (
    check_cluster_health,
    K8sClusterError,
    ClusterNotFoundError,
    KubectlError
)

try:
    status = check_cluster_health('my-cluster')
except ClusterNotFoundError as e:
    print(f"Cluster not found: {e}")
except KubectlError as e:
    print(f"Kubectl error: {e}")
except K8sClusterError as e:
    print(f"Cluster error: {e}")
```

Results always include an `issues` list that describes any problems encountered:

```python
status = check_cluster_health('my-cluster')
if status['issues']:
    print("Warnings/Issues:")
    for issue in status['issues']:
        print(f"  ⚠️  {issue}")
```

## Logging

The skill uses Python's standard logging module. Enable debug logging for detailed output:

```python
import logging

logging.getLogger('k8s_cluster_manager').setLevel(logging.DEBUG)

# Or for all modules
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Pre-Training Validation

```python
from k8s_cluster_manager.src.check_cluster import run_full_cluster_check
import sys

def validate_cluster_for_training(cluster_name, min_gpus=8):
    """Validate cluster before starting training job."""
    report = run_full_cluster_check(cluster_name)
    
    # Check overall health
    if report['overall_status'] != 'healthy':
        print("❌ Cluster is not healthy!")
        for issue in report['cluster_health'].get('issues', []):
            print(f"   - {issue}")
        return False
    
    # Check GPU availability
    gpu_info = report['gpu_availability']
    if not gpu_info['gpu_available']:
        print("❌ GPUs not available!")
        return False
    
    if gpu_info['total_gpu_capacity'] < min_gpus:
        print(f"❌ Insufficient GPUs: {gpu_info['total_gpu_capacity']}/{min_gpus}")
        return False
    
    # Check EFA for distributed training
    efa_info = report['efa_availability']
    if not efa_info['efa_available']:
        print("⚠️  EFA not available (high-performance networking disabled)")
    
    print("✅ Cluster validation passed!")
    print(f"   Nodes: {report['cluster_health']['node_count']}")
    print(f"   GPUs: {gpu_info['total_gpu_capacity']}")
    print(f"   EFAs: {efa_info['total_efa_capacity']}")
    return True

if __name__ == '__main__':
    if not validate_cluster_for_training('training-cluster', min_gpus=8):
        sys.exit(1)
```

### Monitoring Script

```python
from k8s_cluster_manager.src.check_cluster import (
    check_cluster_health,
    check_gpu_availability,
    get_node_resources
)
import time
import json

def monitor_cluster(cluster_name, interval=60):
    """Continuously monitor cluster health."""
    while True:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        health = check_cluster_health(cluster_name)
        gpu = check_gpu_availability()
        resources = get_node_resources()
        
        status = {
            'timestamp': timestamp,
            'healthy': health['status'] == 'healthy',
            'nodes_ready': f"{health['ready_nodes']}/{health['node_count']}",
            'gpus_available': gpu['total_gpu_allocatable'],
            'gpus_total': gpu['total_gpu_capacity']
        }
        
        print(json.dumps(status))
        
        if health['status'] != 'healthy':
            print(f"⚠️  ALERT: Cluster status is {health['status']}")
            for issue in health['issues']:
                print(f"   - {issue}")
        
        time.sleep(interval)

if __name__ == '__main__':
    monitor_cluster('production-cluster', interval=300)
```

### Resource Planning

```python
from k8s_cluster_manager.src.check_cluster import get_node_resources

def analyze_capacity():
    """Analyze cluster capacity for resource planning."""
    resources = get_node_resources()
    
    print("Cluster Capacity Analysis")
    print("=" * 50)
    
    # Group by instance type
    by_instance = {}
    for node in resources['nodes']:
        instance_type = node['instance_type']
        if instance_type not in by_instance:
            by_instance[instance_type] = {
                'count': 0,
                'total_cpu': 0,
                'total_gpu': 0,
                'total_efa': 0
            }
        
        by_instance[instance_type]['count'] += 1
        by_instance[instance_type]['total_cpu'] += int(node['capacity']['cpu'])
        by_instance[instance_type]['total_gpu'] += node['capacity']['gpu']
        by_instance[instance_type]['total_efa'] += node['capacity']['efa']
    
    for instance_type, stats in by_instance.items():
        print(f"\n{instance_type}:")
        print(f"  Count: {stats['count']}")
        print(f"  CPU: {stats['total_cpu']} cores")
        print(f"  GPU: {stats['total_gpu']}")
        print(f"  EFA: {stats['total_efa']}")
    
    print(f"\nTotal Capacity:")
    print(f"  CPU: {resources['total_cpu_capacity']} cores")
    print(f"  GPU: {resources['total_gpu_capacity']}")
    print(f"  EFA: {sum(n['capacity']['efa'] for n in resources['nodes'])}")

if __name__ == '__main__':
    analyze_capacity()
```

## Troubleshooting

### Common Issues

**"kubectl is not installed or not in PATH"**
- Install kubectl: https://kubernetes.io/docs/tasks/tools/
- Ensure it's in your PATH: `which kubectl`

**"AWS CLI is not installed or not in PATH"**
- Install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html
- Configure credentials: `aws configure`

**"Failed to update kubeconfig"**
- Verify cluster name is correct
- Check AWS credentials have EKS access
- Ensure cluster exists in specified region

**"No GPU nodes found"**
- Verify nodes have GPU instance types (p3, p4, g4dn, etc.)
- Check NVIDIA device plugin is installed
- Look for node labels: `kubectl get nodes -l nvidia.com/gpu.present=true`

**"NVIDIA device plugin not found"**
- Install device plugin:
  ```bash
  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
  ```

**"EFA device plugin not found"**
- Install EFA device plugin:
  ```bash
  kubectl apply -f https://raw.githubusercontent.com/aws/aws-efa-k8s-device-plugin/main/aws-efa-k8s-device-plugin.yaml
  ```

### Debug Mode

Enable debug logging to see detailed command execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from k8s_cluster_manager.src.check_cluster import check_cluster_health
status = check_cluster_health('my-cluster')
```

## Contributing

This skill is designed to be self-contained and standalone. When contributing:

1. Use only Python standard library + subprocess
2. Maintain independence from other skills
3. Include comprehensive error handling
4. Add logging with timestamps
5. Update this documentation

## License

This skill is provided as-is for Kubernetes cluster management and validation.
