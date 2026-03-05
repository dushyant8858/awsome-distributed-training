#!/usr/bin/env python3
"""
Ray Cluster Manager - Manage Ray and KubeRay clusters on EKS

This module provides complete management of Ray clusters on Kubernetes
using KubeRay, with support for EFA (Elastic Fabric Adapter) for
high-performance distributed training.
"""

import subprocess
import json
import time
import re
from typing import Dict, Any, Optional, List


class RayClusterError(Exception):
    """Exception raised for Ray cluster management errors."""
    pass


def run_command(cmd: List[str], check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the result.
    
    Args:
        cmd: Command and arguments as a list
        check: Whether to raise exception on non-zero exit
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        CompletedProcess instance with results
        
    Raises:
        RayClusterError: If command fails and check=True
    """
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {' '.join(cmd)}\n"
        error_msg += f"Exit code: {e.returncode}\n"
        if e.stdout:
            error_msg += f"Stdout: {e.stdout}\n"
        if e.stderr:
            error_msg += f"Stderr: {e.stderr}"
        raise RayClusterError(error_msg)


def check_kuberay_installed(namespace: str = "kuberay") -> bool:
    """
    Check if KubeRay operator is installed by verifying the CRD exists.
    
    Args:
        namespace: Namespace where KubeRay might be installed
        
    Returns:
        True if KubeRay CRD exists, False otherwise
    """
    try:
        result = run_command(
            ['kubectl', 'get', 'crd', 'rayclusters.ray.io'],
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def install_kuberay(cluster_name: str, region: str, namespace: str = "kuberay") -> None:
    """
    Install KubeRay operator on the EKS cluster.
    
    Args:
        cluster_name: Name of the EKS cluster
        region: AWS region of the cluster
        namespace: Namespace to install KubeRay (default: kuberay)
        
    Raises:
        RayClusterError: If installation fails
    """
    print(f"Installing KubeRay operator on cluster {cluster_name}...")
    
    # Update kubeconfig
    run_command([
        'aws', 'eks', 'update-kubeconfig',
        '--name', cluster_name,
        '--region', region
    ])
    
    # Create namespace if it doesn't exist
    run_command([
        'kubectl', 'create', 'namespace', namespace
    ], check=False)
    
    # Add KubeRay Helm repo
    run_command([
        'helm', 'repo', 'add', 'kuberay',
        'https://ray-project.github.io/kuberay-helm/'
    ], check=False)
    
    # Update Helm repos
    run_command(['helm', 'repo', 'update'])
    
    # Install KubeRay operator
    run_command([
        'helm', 'install', 'kuberay-operator',
        'kuberay/kuberay-operator',
        '--namespace', namespace,
        '--set', 'image.tag=latest'
    ])
    
    # Wait for operator to be ready
    print("Waiting for KubeRay operator to be ready...")
    run_command([
        'kubectl', 'wait', '--for=condition=ready',
        'pod', '-l', 'app.kubernetes.io/name=kuberay-operator',
        '-n', namespace,
        '--timeout=300s'
    ])
    
    print("KubeRay operator installed successfully!")


def generate_raycluster_yaml(
    job_name: str,
    image_uri: str,
    num_nodes: int,
    use_efa: bool = False,
    checkpoint_pvc: Optional[str] = None,
    head_cpu: int = 4,
    head_memory: str = "16Gi",
    worker_cpu: int = 16,
    worker_memory: str = "64Gi",
    worker_gpu: int = 8,
    namespace: str = "default"
) -> str:
    """
    Generate RayCluster YAML configuration.
    
    Args:
        job_name: Name of the RayCluster
        image_uri: Container image URI
        num_nodes: Total number of nodes (1 head + N-1 workers)
        use_efa: Whether to enable EFA networking
        checkpoint_pvc: PVC name for checkpoint storage
        head_cpu: CPU for head node
        head_memory: Memory for head node
        worker_cpu: CPU per worker
        worker_memory: Memory per worker
        worker_gpu: GPUs per worker
        namespace: Kubernetes namespace
        
    Returns:
        YAML string for RayCluster resource
    """
    
    # Calculate worker replicas
    worker_replicas = max(0, num_nodes - 1)
    
    # Base environment variables
    env_vars = [
        {"name": "RAY_DISABLE_DOCKER_CPU_WARNING", "value": "1"},
        {"name": "RAY_PORT", "value": "6379"},
        {"name": "RAY_ADDRESS", "value": "localhost:6379"},
        {"name": "RAY_USAGE_STATS_ENABLED", "value": "0"}
    ]
    
    # Add EFA-specific environment variables
    if use_efa:
        efa_env_vars = [
            # NCCL configuration
            {"name": "NCCL_DEBUG", "value": "INFO"},
            {"name": "NCCL_SOCKET_IFNAME", "value": "^docker,lo,veth"},
            {"name": "NCCL_TIMEOUT", "value": "1800"},
            {"name": "TORCH_NCCL_TRACE_BUFFER_SIZE", "value": "4096"},
            # CRITICAL: Force NCCL to use OFI (EFA) plugin instead of falling back to sockets
            {"name": "NCCL_NET", "value": "ofi"},
            # EFA libfabric configuration
            {"name": "FI_PROVIDER", "value": "efa"},
            {"name": "FI_EFA_FORK_SAFE", "value": "1"},
            {"name": "FI_EFA_ENABLE_SHM_TRANSFER", "value": "1"},
            # RDMA: set to 0 for g5 instances (no GPUDirect RDMA), 1 for p4d/p5
            {"name": "FI_EFA_USE_DEVICE_RDMA", "value": "0"},
            # Ensure OFI NCCL plugin library is found
            {"name": "LD_LIBRARY_PATH", "value": "/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"},
            # Use simple protocol for EFA without GPUDirect RDMA
            {"name": "NCCL_PROTO", "value": "simple"},
        ]
        env_vars.extend(efa_env_vars)
    
    # Build volumes and volume mounts
    volumes = []
    volume_mounts = []
    
    if checkpoint_pvc:
        volumes.append({
            "name": "checkpoint-volume",
            "persistentVolumeClaim": {"claimName": checkpoint_pvc}
        })
        volume_mounts.append({
            "name": "checkpoint-volume",
            "mountPath": "/checkpoints"
        })
    
    # EFA volumes
    if use_efa:
        volumes.extend([
            {
                "name": "efa-devices",
                "hostPath": {"path": "/dev/infiniband"}
            },
            {
                "name": "efa-lib",
                "hostPath": {"path": "/opt/amazon/efa"}
            }
        ])
        volume_mounts.extend([
            {
                "name": "efa-devices",
                "mountPath": "/dev/infiniband"
            },
            {
                "name": "efa-lib",
                "mountPath": "/opt/amazon/efa"
            }
        ])
    
    # Security context for EFA
    security_context = {}
    if use_efa:
        security_context = {
            "capabilities": {
                "add": ["IPC_LOCK", "SYS_RESOURCE"]
            }
        }
    
    # Build head node spec
    head_spec = {
        "rayStartParams": {
            "dashboard-host": "0.0.0.0",
            "metrics-export-port": "8080"
        },
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "ray-head",
                        "image": image_uri,
                        "ports": [
                            {"containerPort": 6379, "name": "gcs"},
                            {"containerPort": 8265, "name": "dashboard"},
                            {"containerPort": 10001, "name": "client"},
                            {"containerPort": 8000, "name": "serve"},
                            {"containerPort": 8080, "name": "metrics"}
                        ],
                        "resources": {
                            "limits": {
                                "cpu": str(head_cpu),
                                "memory": head_memory
                            },
                            "requests": {
                                "cpu": str(head_cpu),
                                "memory": head_memory
                            }
                        },
                        "env": env_vars,
                        "volumeMounts": volume_mounts
                    }
                ],
                "volumes": volumes
            }
        }
    }
    
    if security_context:
        head_spec["template"]["spec"]["containers"][0]["securityContext"] = security_context
    
    # Build worker node spec
    worker_spec = {
        "replicas": worker_replicas,
        "minReplicas": worker_replicas,
        "maxReplicas": worker_replicas,
        "groupName": "worker-group",
        "rayStartParams": {},
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "ray-worker",
                        "image": image_uri,
                        "resources": {
                            "limits": {
                                "cpu": str(worker_cpu),
                                "memory": worker_memory,
                                "nvidia.com/gpu": str(worker_gpu)
                            },
                            "requests": {
                                "cpu": str(worker_cpu),
                                "memory": worker_memory,
                                "nvidia.com/gpu": str(worker_gpu)
                            }
                        },
                        "env": env_vars,
                        "volumeMounts": volume_mounts
                    }
                ],
                "volumes": volumes
            }
        }
    }
    
    if security_context:
        worker_spec["template"]["spec"]["containers"][0]["securityContext"] = security_context
    
    # Build full RayCluster spec
    raycluster = {
        "apiVersion": "ray.io/v1alpha1",
        "kind": "RayCluster",
        "metadata": {
            "name": job_name,
            "namespace": namespace
        },
        "spec": {
            "headGroupSpec": head_spec,
            "workerGroupSpecs": [worker_spec] if worker_replicas > 0 else []
        }
    }
    
    # Convert to YAML
    return _dict_to_yaml(raycluster)


def _dict_to_yaml(d: Any, indent: int = 0) -> str:
    """Convert a dictionary to YAML string."""
    yaml_str = []
    prefix = "  " * indent
    
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                yaml_str.append(f"{prefix}{key}:")
                yaml_str.append(_dict_to_yaml(value, indent + 1))
            elif isinstance(value, list):
                yaml_str.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        yaml_str.append(f"{prefix}  -")
                        for k, v in item.items():
                            if isinstance(v, (dict, list)):
                                yaml_str.append(f"{prefix}    {k}:")
                                yaml_str.append(_dict_to_yaml(v, indent + 3))
                            else:
                                yaml_str.append(f"{prefix}    {k}: {_format_yaml_value(v)}")
                    else:
                        yaml_str.append(f"{prefix}  - {_format_yaml_value(item)}")
            else:
                yaml_str.append(f"{prefix}{key}: {_format_yaml_value(value)}")
    
    return "\n".join(yaml_str)


def _format_yaml_value(value: Any) -> str:
    """Format a value for YAML output."""
    if isinstance(value, str):
        if any(c in value for c in [':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`', '"', "'"]):
            return f'"{value}"'
        return value
    return str(value)


def create_raycluster(config: Dict[str, Any]) -> None:
    """
    Create a RayCluster from configuration.
    
    Args:
        config: Dictionary containing cluster configuration
            Required keys:
                - job_name: Name of the RayCluster
                - image_uri: Container image URI
                - num_nodes: Total number of nodes
            Optional keys:
                - use_efa: Enable EFA (default: False)
                - checkpoint_pvc: PVC for checkpoints
                - head_cpu: Head node CPU (default: 4)
                - head_memory: Head node memory (default: 16Gi)
                - worker_cpu: Worker CPU (default: 16)
                - worker_memory: Worker memory (default: 64Gi)
                - worker_gpu: GPUs per worker (default: 8)
                - namespace: Kubernetes namespace (default: default)
                
    Raises:
        RayClusterError: If creation fails
    """
    required_keys = ['job_name', 'image_uri', 'num_nodes']
    for key in required_keys:
        if key not in config:
            raise RayClusterError(f"Missing required configuration key: {key}")
    
    # Generate YAML
    yaml_content = generate_raycluster_yaml(
        job_name=config['job_name'],
        image_uri=config['image_uri'],
        num_nodes=config['num_nodes'],
        use_efa=config.get('use_efa', False),
        checkpoint_pvc=config.get('checkpoint_pvc'),
        head_cpu=config.get('head_cpu', 4),
        head_memory=config.get('head_memory', '16Gi'),
        worker_cpu=config.get('worker_cpu', 16),
        worker_memory=config.get('worker_memory', '64Gi'),
        worker_gpu=config.get('worker_gpu', 8),
        namespace=config.get('namespace', 'default')
    )
    
    # Apply YAML using kubectl
    print(f"Creating RayCluster {config['job_name']}...")
    
    try:
        result = subprocess.run(
            ['kubectl', 'apply', '-f', '-'],
            input=yaml_content,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RayClusterError(f"Failed to create RayCluster: {e.stderr}")
    
    print(f"RayCluster {config['job_name']} created successfully!")


def delete_raycluster(name: str, namespace: str = "default") -> None:
    """
    Delete a RayCluster.
    
    Args:
        name: Name of the RayCluster to delete
        namespace: Kubernetes namespace
        
    Raises:
        RayClusterError: If deletion fails
    """
    print(f"Deleting RayCluster {name}...")
    
    try:
        run_command([
            'kubectl', 'delete', 'raycluster', name,
            '-n', namespace,
            '--ignore-not-found=true'
        ])
        print(f"RayCluster {name} deleted successfully!")
    except RayClusterError as e:
        raise RayClusterError(f"Failed to delete RayCluster: {e}")


def get_raycluster_status(name: str, namespace: str = "default") -> str:
    """
    Check if RayCluster is ready.
    
    Args:
        name: Name of the RayCluster
        namespace: Kubernetes namespace
        
    Returns:
        Status string: 'ready', 'pending', 'failed', or 'not_found'
        
    Raises:
        RayClusterError: If status check fails
    """
    try:
        result = run_command([
            'kubectl', 'get', 'raycluster', name,
            '-n', namespace,
            '-o', 'json'
        ], check=False)
        
        if result.returncode != 0:
            if "NotFound" in result.stderr:
                return 'not_found'
            raise RayClusterError(f"Failed to get status: {result.stderr}")
        
        cluster_info = json.loads(result.stdout)
        
        # Check status conditions
        status = cluster_info.get('status', {})
        state = status.get('state', 'unknown')
        
        if state == 'ready':
            return 'ready'
        elif state == 'failed':
            return 'failed'
        else:
            # Check if all pods are ready
            return _check_pods_ready(name, namespace)
            
    except json.JSONDecodeError as e:
        raise RayClusterError(f"Failed to parse cluster status: {e}")
    except Exception as e:
        if isinstance(e, RayClusterError):
            raise
        raise RayClusterError(f"Failed to get cluster status: {e}")


def _check_pods_ready(cluster_name: str, namespace: str) -> str:
    """Check if all pods for the cluster are ready."""
    try:
        result = run_command([
            'kubectl', 'get', 'pods',
            '-l', f'ray.io/cluster={cluster_name}',
            '-n', namespace,
            '-o', 'json'
        ])
        
        pod_info = json.loads(result.stdout)
        pods = pod_info.get('items', [])
        
        if not pods:
            return 'pending'
        
        all_ready = True
        for pod in pods:
            phase = pod.get('status', {}).get('phase', '')
            if phase != 'Running':
                all_ready = False
                break
            
            # Check container statuses
            container_statuses = pod.get('status', {}).get('containerStatuses', [])
            for cs in container_statuses:
                if not cs.get('ready', False):
                    all_ready = False
                    break
        
        return 'ready' if all_ready else 'pending'
        
    except Exception:
        return 'pending'


def list_rayclusters(namespace: str = "default") -> List[Dict[str, Any]]:
    """
    List all RayClusters in the namespace.
    
    Args:
        namespace: Kubernetes namespace
        
    Returns:
        List of RayCluster information dictionaries
    """
    try:
        result = run_command([
            'kubectl', 'get', 'rayclusters',
            '-n', namespace,
            '-o', 'json'
        ])
        
        clusters_info = json.loads(result.stdout)
        clusters = []
        
        for item in clusters_info.get('items', []):
            metadata = item.get('metadata', {})
            status = item.get('status', {})
            spec = item.get('spec', {})
            
            # Count workers
            worker_specs = spec.get('workerGroupSpecs', [])
            total_workers = sum(
                ws.get('replicas', 0) for ws in worker_specs
            )
            
            cluster_info = {
                'name': metadata.get('name'),
                'namespace': metadata.get('namespace'),
                'status': status.get('state', 'unknown'),
                'head_cpu': spec.get('headGroupSpec', {}).get('template', {}).get('spec', {}).get('containers', [{}])[0].get('resources', {}).get('limits', {}).get('cpu', 'unknown'),
                'worker_count': total_workers,
                'created': metadata.get('creationTimestamp')
            }
            clusters.append(cluster_info)
        
        return clusters
        
    except Exception as e:
        raise RayClusterError(f"Failed to list clusters: {e}")


def scale_raycluster(name: str, num_workers: int, namespace: str = "default") -> None:
    """
    Scale a RayCluster to the specified number of workers.
    
    Args:
        name: Name of the RayCluster
        num_workers: Number of worker nodes
        namespace: Kubernetes namespace
        
    Raises:
        RayClusterError: If scaling fails
    """
    print(f"Scaling RayCluster {name} to {num_workers} workers...")
    
    try:
        # Get current cluster
        result = run_command([
            'kubectl', 'get', 'raycluster', name,
            '-n', namespace,
            '-o', 'json'
        ])
        
        cluster_info = json.loads(result.stdout)
        
        # Update worker replicas
        worker_specs = cluster_info.get('spec', {}).get('workerGroupSpecs', [])
        if worker_specs:
            worker_specs[0]['replicas'] = num_workers
            worker_specs[0]['minReplicas'] = num_workers
            worker_specs[0]['maxReplicas'] = num_workers
        
        # Apply updated spec
        updated_yaml = _dict_to_yaml(cluster_info)
        
        subprocess.run(
            ['kubectl', 'apply', '-f', '-'],
            input=updated_yaml,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"RayCluster {name} scaled to {num_workers} workers!")
        
    except subprocess.CalledProcessError as e:
        raise RayClusterError(f"Failed to scale RayCluster: {e.stderr}")
    except Exception as e:
        raise RayClusterError(f"Failed to scale RayCluster: {e}")


def get_raycluster_endpoint(name: str, namespace: str = "default") -> Optional[str]:
    """
    Get the dashboard endpoint for a RayCluster.
    
    Args:
        name: Name of the RayCluster
        namespace: Kubernetes namespace
        
    Returns:
        Dashboard URL or None if not available
    """
    try:
        result = run_command([
            'kubectl', 'get', 'service',
            f'{name}-head-svc',
            '-n', namespace,
            '-o', 'json'
        ], check=False)
        
        if result.returncode != 0:
            return None
        
        service_info = json.loads(result.stdout)
        
        # Try to get external IP
        status = service_info.get('status', {})
        load_balancer = status.get('loadBalancer', {})
        ingress = load_balancer.get('ingress', [])
        
        if ingress:
            ip = ingress[0].get('ip')
            hostname = ingress[0].get('hostname')
            endpoint = ip or hostname
            if endpoint:
                return f"http://{endpoint}:8265"
        
        # If no external IP, return internal endpoint
        return f"http://{name}-head-svc.{namespace}.svc.cluster.local:8265"
        
    except Exception:
        return None


def wait_for_raycluster(name: str, namespace: str = "default", timeout: int = 600) -> bool:
    """
    Wait for RayCluster to be ready.
    
    Args:
        name: Name of the RayCluster
        namespace: Kubernetes namespace
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if cluster is ready, False if timeout
    """
    print(f"Waiting for RayCluster {name} to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = get_raycluster_status(name, namespace)
        
        if status == 'ready':
            print(f"RayCluster {name} is ready!")
            return True
        elif status == 'failed':
            raise RayClusterError(f"RayCluster {name} failed to start")
        
        time.sleep(10)
    
    raise RayClusterError(f"Timeout waiting for RayCluster {name}")


def verify_gpu_utilization(head_pod: str, namespace: str = "default") -> Dict[str, Any]:
    """
    Verify that GPUs are being utilized in the Ray cluster.
    
    Args:
        head_pod: Name of the head pod
        namespace: Kubernetes namespace
        
    Returns:
        Dictionary with GPU utilization info:
        {
            'gpus_available': int,
            'gpus_used': float,
            'utilization': List[Dict],  # Per-GPU utilization
            'healthy': bool
        }
    """
    try:
        # Check Ray status for GPU allocation
        result = run_command([
            'kubectl', 'exec', head_pod, '-n', namespace, '--',
            'ray', 'status'
        ], check=False)
        
        if result.returncode != 0:
            return {
                'gpus_available': 0,
                'gpus_used': 0,
                'utilization': [],
                'healthy': False,
                'error': 'Failed to get Ray status'
            }
        
        # Parse GPU info from ray status
        gpus_available = 0.0
        gpus_used = 0.0
        
        for line in result.stdout.split('\n'):
            if 'GPU' in line and '/' in line:
                # Parse line like " 4.0/4.0 GPU (4.0 used of 4.0 reserved)"
                match = re.search(r'(\d+\.?\d*)/(\d+\.?\d*)\s+GPU', line)
                if match:
                    gpus_used = float(match.group(1))
                    gpus_available = float(match.group(2))
        
        # Check nvidia-smi for actual GPU utilization
        nvidia_result = run_command([
            'kubectl', 'exec', head_pod, '-n', namespace, '--',
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], check=False)
        
        utilization = []
        if nvidia_result.returncode == 0:
            for line in nvidia_result.stdout.strip().split('\n'):
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        utilization.append({
                            'gpu_util': float(parts[0].strip()),
                            'memory_used': float(parts[1].strip()),
                            'memory_total': float(parts[2].strip())
                        })
        
        return {
            'gpus_available': gpus_available,
            'gpus_used': gpus_used,
            'utilization': utilization,
            'healthy': gpus_used > 0 or gpus_available == 0  # Healthy if using GPUs or no GPUs expected
        }
        
    except Exception as e:
        return {
            'gpus_available': 0,
            'gpus_used': 0,
            'utilization': [],
            'healthy': False,
            'error': str(e)
        }


def get_ray_job_command(
    head_pod: str,
    working_dir: str,
    command: str,
    use_kubectl_exec: bool = True
) -> str:
    """
    Get the correct command to run a job on the Ray cluster.
    
    IMPORTANT: For multi-GPU training, use_kubectl_exec MUST be True.
    Using 'ray job submit' isolates the job from GPU resources.
    
    Args:
        head_pod: Name of the head pod
        working_dir: Working directory in the pod
        command: Command to run
        use_kubectl_exec: If True, returns kubectl exec command (recommended).
                         If False, returns ray job submit command (not recommended for multi-GPU).
                         
    Returns:
        Command string to execute
        
    Example:
        >>> cmd = get_ray_job_command(
        ...     head_pod='my-cluster-head-0',
        ...     working_dir='/workspace',
        ...     command='python3 train.py'
        ... )
        >>> print(cmd)
        kubectl exec my-cluster-head-0 -- bash -c 'cd /workspace && python3 train.py'
    """
    if use_kubectl_exec:
        # Recommended: Run directly in pod
        return f"kubectl exec {head_pod} -- bash -c 'cd {working_dir} && {command}'"
    else:
        # Not recommended for multi-GPU: Ray job submit isolates from resources
        return f"kubectl exec {head_pod} -- ray job submit --working-dir {working_dir} -- {command}"


if __name__ == '__main__':
    # Example usage
    print("Ray Cluster Manager")
    print("==================")
    
    # Check if KubeRay is installed
    if check_kuberay_installed():
        print("KubeRay is installed")
    else:
        print("KubeRay is not installed")
        print("Install with: install_kuberay('cluster-name', 'region')")
