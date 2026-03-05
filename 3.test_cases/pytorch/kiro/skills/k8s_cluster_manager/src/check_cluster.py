#!/usr/bin/env python3
"""
Kubernetes Cluster Health Checker for Training Workloads

This module provides comprehensive health checks and validation for Kubernetes clusters
running distributed training workloads. It supports EKS clusters with GPU and EFA capabilities.

Usage:
    from k8s_cluster_manager.src.check_cluster import check_cluster_health
    status = check_cluster_health('my-cluster')
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class K8sClusterError(Exception):
    """Base exception for Kubernetes cluster operations."""
    pass


class ClusterNotFoundError(K8sClusterError):
    """Raised when a cluster cannot be found or accessed."""
    pass


class KubectlError(K8sClusterError):
    """Raised when kubectl command fails."""
    pass


def run_command(
    cmd: List[str],
    timeout: int = 60,
    capture_output: bool = True,
    check: bool = True
) -> subprocess.CompletedProcess:
    """
    Execute a shell command with proper error handling.
    
    Args:
        cmd: Command and arguments as a list
        timeout: Command timeout in seconds
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise exception on non-zero exit
        
    Returns:
        CompletedProcess instance with return code and output
        
    Raises:
        KubectlError: If command fails and check=True
    """
    logger.debug(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=False
        )
        
        if check and result.returncode != 0:
            error_msg = f"Command failed with exit code {result.returncode}: {result.stderr}"
            logger.error(error_msg)
            raise KubectlError(error_msg)
            
        return result
        
    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout} seconds: {' '.join(cmd)}"
        logger.error(error_msg)
        raise KubectlError(error_msg)
    except FileNotFoundError as e:
        error_msg = f"Command not found: {cmd[0]}. Is it installed and in PATH?"
        logger.error(error_msg)
        raise KubectlError(error_msg) from e


def check_kubectl_installed() -> bool:
    """
    Verify that kubectl is installed and accessible.
    
    Returns:
        True if kubectl is available, False otherwise
    """
    try:
        result = run_command(['kubectl', 'version', '--client'], check=False)
        if result.returncode == 0:
            logger.info("kubectl is installed and accessible")
            return True
        else:
            logger.error("kubectl is not properly installed")
            return False
    except Exception as e:
        logger.error(f"Failed to check kubectl: {e}")
        return False


def check_aws_cli_installed() -> bool:
    """
    Verify that AWS CLI is installed and accessible.
    
    Returns:
        True if AWS CLI is available, False otherwise
    """
    try:
        result = run_command(['aws', '--version'], check=False)
        if result.returncode == 0:
            logger.info("AWS CLI is installed and accessible")
            return True
        else:
            logger.error("AWS CLI is not properly installed")
            return False
    except Exception as e:
        logger.error(f"Failed to check AWS CLI: {e}")
        return False


def check_cluster_health(cluster_name: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive health check on an EKS cluster.
    
    This function checks:
    - Cluster existence and accessibility
    - API server connectivity
    - Node health and status
    - CoreDNS functionality
    - Overall cluster readiness
    
    Args:
        cluster_name: Name of the EKS cluster
        region: AWS region (optional, uses default if not provided)
        
    Returns:
        Dictionary containing:
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
        
    Raises:
        ClusterNotFoundError: If cluster cannot be found
        KubectlError: If kubectl commands fail
    """
    logger.info(f"Starting health check for cluster: {cluster_name}")
    
    result = {
        'cluster_name': cluster_name,
        'status': 'unknown',
        'accessible': False,
        'api_server': False,
        'node_count': 0,
        'ready_nodes': 0,
        'issues': [],
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'region': region or 'default'
    }
    
    # Check prerequisites
    if not check_kubectl_installed():
        result['issues'].append("kubectl is not installed or not in PATH")
        result['status'] = 'unhealthy'
        return result
    
    if not check_aws_cli_installed():
        result['issues'].append("AWS CLI is not installed or not in PATH")
        result['status'] = 'unhealthy'
        return result
    
    # Update kubeconfig for the cluster
    try:
        update_cmd = ['aws', 'eks', 'update-kubeconfig', '--name', cluster_name]
        if region:
            update_cmd.extend(['--region', region])
        
        update_result = run_command(update_cmd, check=False)
        if update_result.returncode != 0:
            result['issues'].append(f"Failed to update kubeconfig: {update_result.stderr}")
            result['status'] = 'unhealthy'
            return result
            
        logger.info(f"Successfully updated kubeconfig for cluster: {cluster_name}")
        result['accessible'] = True
        
    except Exception as e:
        error_msg = f"Failed to update kubeconfig: {str(e)}"
        logger.error(error_msg)
        result['issues'].append(error_msg)
        result['status'] = 'unhealthy'
        return result
    
    # Check API server connectivity
    try:
        api_result = run_command(['kubectl', 'cluster-info'], check=False)
        if api_result.returncode == 0:
            result['api_server'] = True
            logger.info("API server is accessible")
        else:
            result['issues'].append("API server is not accessible")
            result['status'] = 'unhealthy'
            return result
    except Exception as e:
        error_msg = f"Failed to connect to API server: {str(e)}"
        logger.error(error_msg)
        result['issues'].append(error_msg)
        result['status'] = 'unhealthy'
        return result
    
    # Get node information
    try:
        nodes_result = run_command([
            'kubectl', 'get', 'nodes', 
            '-o', 'json'
        ])
        
        nodes_data = json.loads(nodes_result.stdout)
        nodes = nodes_data.get('items', [])
        result['node_count'] = len(nodes)
        
        ready_count = 0
        for node in nodes:
            conditions = node.get('status', {}).get('conditions', [])
            for condition in conditions:
                if condition.get('type') == 'Ready':
                    if condition.get('status') == 'True':
                        ready_count += 1
                    else:
                        node_name = node.get('metadata', {}).get('name', 'unknown')
                        result['issues'].append(f"Node {node_name} is not ready")
                    break
        
        result['ready_nodes'] = ready_count
        logger.info(f"Found {ready_count}/{len(nodes)} ready nodes")
        
    except Exception as e:
        error_msg = f"Failed to get node information: {str(e)}"
        logger.error(error_msg)
        result['issues'].append(error_msg)
    
    # Check CoreDNS
    try:
        coredns_result = run_command([
            'kubectl', 'get', 'pods', '-n', 'kube-system',
            '-l', 'k8s-app=kube-dns',
            '-o', 'json'
        ], check=False)
        
        if coredns_result.returncode == 0:
            coredns_data = json.loads(coredns_result.stdout)
            coredns_pods = coredns_data.get('items', [])
            
            running_coredns = sum(
                1 for pod in coredns_pods
                if pod.get('status', {}).get('phase') == 'Running'
            )
            
            if running_coredns == 0 and len(coredns_pods) > 0:
                result['issues'].append("CoreDNS pods are not running")
            elif len(coredns_pods) == 0:
                result['issues'].append("CoreDNS pods not found")
            else:
                logger.info(f"CoreDNS is running ({running_coredns} pods)")
        else:
            result['issues'].append("Failed to check CoreDNS status")
            
    except Exception as e:
        error_msg = f"Error checking CoreDNS: {str(e)}"
        logger.warning(error_msg)
        result['issues'].append(error_msg)
    
    # Determine overall status
    if len(result['issues']) == 0 and result['ready_nodes'] == result['node_count']:
        result['status'] = 'healthy'
    elif result['ready_nodes'] > 0:
        result['status'] = 'degraded'
    else:
        result['status'] = 'unhealthy'
    
    logger.info(f"Cluster health check complete. Status: {result['status']}")
    return result


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and capacity in the cluster.
    
    This function verifies:
    - NVIDIA device plugin installation
    - GPU nodes and their capacity
    - GPU allocatable resources
    - Running GPU workloads
    
    Returns:
        Dictionary containing:
        {
            'gpu_available': bool,
            'device_plugin_installed': bool,
            'gpu_nodes': List[Dict],
            'total_gpu_capacity': int,
            'total_gpu_allocatable': int,
            'gpu_pods_running': int,
            'timestamp': str,
            'issues': List[str]
        }
        
    Raises:
        KubectlError: If kubectl commands fail
    """
    logger.info("Checking GPU availability...")
    
    result = {
        'gpu_available': False,
        'device_plugin_installed': False,
        'gpu_nodes': [],
        'total_gpu_capacity': 0,
        'total_gpu_allocatable': 0,
        'gpu_pods_running': 0,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'issues': []
    }
    
    # Check if kubectl is available
    if not check_kubectl_installed():
        result['issues'].append("kubectl is not available")
        return result
    
    # Check NVIDIA device plugin
    try:
        # Try multiple known labels for the NVIDIA device plugin
        plugin_labels = [
            'name=nvidia-device-plugin-ds',
            'app=nvidia-device-plugin-daemonset',
            'app.kubernetes.io/name=nvidia-device-plugin',
        ]
        
        for label in plugin_labels:
            if result['device_plugin_installed']:
                break
                
            device_plugin_result = run_command([
                'kubectl', 'get', 'pods', '-n', 'kube-system',
                '-l', label,
                '-o', 'json'
            ], check=False)
            
            if device_plugin_result.returncode == 0:
                plugin_data = json.loads(device_plugin_result.stdout)
                plugin_pods = plugin_data.get('items', [])
                
                running_plugins = sum(
                    1 for pod in plugin_pods
                    if pod.get('status', {}).get('phase') == 'Running'
                )
                
                if running_plugins > 0:
                    result['device_plugin_installed'] = True
                    logger.info(f"NVIDIA device plugin is running ({running_plugins} pods, label={label})")
        
        if not result['device_plugin_installed']:
            result['issues'].append("NVIDIA device plugin not found")
                
    except Exception as e:
        error_msg = f"Error checking NVIDIA device plugin: {str(e)}"
        logger.error(error_msg)
        result['issues'].append(error_msg)
    
    # Check GPU nodes
    try:
        nodes_result = run_command([
            'kubectl', 'get', 'nodes',
            '-o', 'json'
        ])
        
        nodes_data = json.loads(nodes_result.stdout)
        nodes = nodes_data.get('items', [])
        
        for node in nodes:
            node_name = node.get('metadata', {}).get('name', 'unknown')
            labels = node.get('metadata', {}).get('labels', {})
            capacity = node.get('status', {}).get('capacity', {})
            allocatable = node.get('status', {}).get('allocatable', {})
            
            # Check for GPU labels or resources
            has_gpu = False
            gpu_count = 0
            
            # Check for NVIDIA GPU label
            if 'nvidia.com/gpu.present' in labels or 'nvidia.com/gpu' in labels:
                has_gpu = True
            
            # Check capacity for GPU resources
            if 'nvidia.com/gpu' in capacity:
                has_gpu = True
                try:
                    gpu_count = int(capacity['nvidia.com/gpu'])
                except (ValueError, TypeError):
                    gpu_count = 0
            
            if has_gpu or gpu_count > 0:
                allocatable_gpu = 0
                if 'nvidia.com/gpu' in allocatable:
                    try:
                        allocatable_gpu = int(allocatable['nvidia.com/gpu'])
                    except (ValueError, TypeError):
                        allocatable_gpu = 0
                
                node_info = {
                    'name': node_name,
                    'capacity': gpu_count,
                    'allocatable': allocatable_gpu,
                    'instance_type': labels.get('node.kubernetes.io/instance-type', 'unknown')
                }
                
                result['gpu_nodes'].append(node_info)
                result['total_gpu_capacity'] += gpu_count
                result['total_gpu_allocatable'] += allocatable_gpu
                
                logger.info(f"Found GPU node: {node_name} with {gpu_count} GPUs")
        
        if len(result['gpu_nodes']) == 0:
            result['issues'].append("No GPU nodes found in cluster")
        else:
            logger.info(f"Found {len(result['gpu_nodes'])} GPU nodes with "
                       f"{result['total_gpu_capacity']} total GPUs")
            
    except Exception as e:
        error_msg = f"Error checking GPU nodes: {str(e)}"
        logger.error(error_msg)
        result['issues'].append(error_msg)
    
    # Count GPU pods
    try:
        pods_result = run_command([
            'kubectl', 'get', 'pods', '--all-namespaces',
            '-o', 'json'
        ], check=False)
        
        if pods_result.returncode == 0:
            pods_data = json.loads(pods_result.stdout)
            pods = pods_data.get('items', [])
            
            gpu_pod_count = 0
            for pod in pods:
                containers = pod.get('spec', {}).get('containers', [])
                for container in containers:
                    resources = container.get('resources', {})
                    limits = resources.get('limits', {})
                    requests = resources.get('requests', {})
                    
                    if 'nvidia.com/gpu' in limits or 'nvidia.com/gpu' in requests:
                        gpu_pod_count += 1
                        break
            
            result['gpu_pods_running'] = gpu_pod_count
            logger.info(f"Found {gpu_pod_count} pods requesting GPUs")
            
    except Exception as e:
        error_msg = f"Error counting GPU pods: {str(e)}"
        logger.warning(error_msg)
    
    # Determine GPU availability
    # If the device plugin is detected and GPU nodes exist, GPUs are available.
    # Also: if GPU nodes exist with capacity > 0 and allocatable > 0, the GPUs
    # are clearly working even if the plugin wasn't found under known labels.
    if (result['device_plugin_installed'] and 
        len(result['gpu_nodes']) > 0 and 
        result['total_gpu_capacity'] > 0):
        result['gpu_available'] = True
        logger.info("GPU is available in the cluster")
    elif (len(result['gpu_nodes']) > 0 and
          result['total_gpu_capacity'] > 0 and
          result['total_gpu_allocatable'] > 0):
        result['gpu_available'] = True
        logger.info("GPU is available in the cluster (detected from node resources)")
    else:
        logger.warning("GPU is not available in the cluster")
    
    return result


def check_efa_availability() -> Dict[str, Any]:
    """
    Check EFA (Elastic Fabric Adapter) availability on nodes.
    
    This function verifies:
    - EFA device plugin installation
    - Nodes with EFA devices
    - EFA capacity per node
    
    Returns:
        Dictionary containing:
        {
            'efa_available': bool,
            'device_plugin_installed': bool,
            'efa_nodes': List[Dict],
            'total_efa_capacity': int,
            'timestamp': str,
            'issues': List[str]
        }
        
    Raises:
        KubectlError: If kubectl commands fail
    """
    logger.info("Checking EFA availability...")
    
    result = {
        'efa_available': False,
        'device_plugin_installed': False,
        'efa_nodes': [],
        'total_efa_capacity': 0,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'issues': []
    }
    
    # Check if kubectl is available
    if not check_kubectl_installed():
        result['issues'].append("kubectl is not available")
        return result
    
    # Check EFA device plugin
    try:
        # Try multiple known labels for the EFA device plugin
        efa_plugin_labels = [
            'app=aws-efa-k8s-device-plugin',
            'name=dependencies-aws-efa-k8s-device-plugin',
            'app.kubernetes.io/name=aws-efa-k8s-device-plugin',
        ]
        
        for label in efa_plugin_labels:
            if result['device_plugin_installed']:
                break
                
            device_plugin_result = run_command([
                'kubectl', 'get', 'pods', '-n', 'kube-system',
                '-l', label,
                '-o', 'json'
            ], check=False)
            
            if device_plugin_result.returncode == 0:
                plugin_data = json.loads(device_plugin_result.stdout)
                plugin_pods = plugin_data.get('items', [])
                
                running_plugins = sum(
                    1 for pod in plugin_pods
                    if pod.get('status', {}).get('phase') == 'Running'
                )
                
                if running_plugins > 0:
                    result['device_plugin_installed'] = True
                    logger.info(f"EFA device plugin is running ({running_plugins} pods, label={label})")
        
        if not result['device_plugin_installed']:
            result['issues'].append("EFA device plugin not found")
            
    except Exception as e:
        error_msg = f"Error checking EFA device plugin: {str(e)}"
        logger.error(error_msg)
        result['issues'].append(error_msg)
    
    # Check EFA nodes
    try:
        nodes_result = run_command([
            'kubectl', 'get', 'nodes',
            '-o', 'json'
        ])
        
        nodes_data = json.loads(nodes_result.stdout)
        nodes = nodes_data.get('items', [])
        
        for node in nodes:
            node_name = node.get('metadata', {}).get('name', 'unknown')
            labels = node.get('metadata', {}).get('labels', {})
            capacity = node.get('status', {}).get('capacity', {})
            
            # Check for EFA resources
            efa_count = 0
            if 'vpc.amazonaws.com/efa' in capacity:
                try:
                    efa_count = int(capacity['vpc.amazonaws.com/efa'])
                except (ValueError, TypeError):
                    efa_count = 0
            
            # Check for EFA labels
            has_efa_label = 'vpc.amazonaws.com/efa.present' in labels
            
            if efa_count > 0 or has_efa_label:
                node_info = {
                    'name': node_name,
                    'capacity': efa_count,
                    'instance_type': labels.get('node.kubernetes.io/instance-type', 'unknown')
                }
                
                result['efa_nodes'].append(node_info)
                result['total_efa_capacity'] += efa_count
                
                logger.info(f"Found EFA node: {node_name} with {efa_count} EFA devices")
        
        if len(result['efa_nodes']) == 0:
            result['issues'].append("No EFA nodes found in cluster")
        else:
            logger.info(f"Found {len(result['efa_nodes'])} EFA nodes with "
                       f"{result['total_efa_capacity']} total EFA devices")
            
    except Exception as e:
        error_msg = f"Error checking EFA nodes: {str(e)}"
        logger.error(error_msg)
        result['issues'].append(error_msg)
    
    # Determine EFA availability
    # If the device plugin is detected and EFA nodes exist, EFA is available.
    # Also: if EFA nodes exist with capacity > 0, EFA is clearly working even
    # if the plugin wasn't found under known labels.
    if (result['device_plugin_installed'] and 
        len(result['efa_nodes']) > 0 and 
        result['total_efa_capacity'] > 0):
        result['efa_available'] = True
        logger.info("EFA is available in the cluster")
    elif (len(result['efa_nodes']) > 0 and
          result['total_efa_capacity'] > 0):
        result['efa_available'] = True
        logger.info("EFA is available in the cluster (detected from node resources)")
    else:
        logger.warning("EFA is not available in the cluster")
    
    return result


def check_addon_status(addon_name: str, namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Check the status of a Kubernetes addon or component.
    
    This function checks if an addon is installed and running properly.
    Supports both EKS managed addons and self-managed components.
    
    Args:
        addon_name: Name of the addon to check (e.g., 'kubeflow', 'kuberay', 'nvidia-device-plugin')
        namespace: Namespace where the addon is installed (optional)
        
    Returns:
        Dictionary containing:
        {
            'addon_name': str,
            'installed': bool,
            'running': bool,
            'pods': List[Dict],
            'services': List[Dict],
            'deployments': List[Dict],
            'timestamp': str,
            'issues': List[str]
        }
        
    Raises:
        KubectlError: If kubectl commands fail
    """
    logger.info(f"Checking addon status: {addon_name}")
    
    result = {
        'addon_name': addon_name,
        'installed': False,
        'running': False,
        'pods': [],
        'services': [],
        'deployments': [],
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'issues': []
    }
    
    # Check if kubectl is available
    if not check_kubectl_installed():
        result['issues'].append("kubectl is not available")
        return result
    
    # Define common addon configurations
    addon_configs = {
        'kubeflow': {
            'namespaces': ['kubeflow'],
            'labels': ['app=kubeflow'],
            'deployments': ['kubeflow-operator']
        },
        'kuberay': {
            'namespaces': ['kuberay', 'ray-system'],
            'labels': ['app.kubernetes.io/name=kuberay'],
            'deployments': ['kuberay-operator']
        },
        'nvidia-device-plugin': {
            'namespaces': ['kube-system'],
            'labels': ['name=nvidia-device-plugin-ds'],
            'daemonsets': ['nvidia-device-plugin-daemonset']
        },
        'aws-efa': {
            'namespaces': ['kube-system'],
            'labels': ['app=aws-efa-k8s-device-plugin'],
            'daemonsets': ['aws-efa-k8s-device-plugin']
        },
        'coredns': {
            'namespaces': ['kube-system'],
            'labels': ['k8s-app=kube-dns'],
            'deployments': ['coredns']
        },
        'kube-proxy': {
            'namespaces': ['kube-system'],
            'labels': ['k8s-app=kube-proxy'],
            'daemonsets': ['kube-proxy']
        }
    }
    
    # Get addon configuration
    config = addon_configs.get(addon_name.lower(), {})
    search_namespaces = [namespace] if namespace else config.get('namespaces', [''])
    
    # Search across namespaces
    for ns in search_namespaces:
        ns_flag = ['-n', ns] if ns else ['--all-namespaces']
        
        # Check for pods
        try:
            labels = config.get('labels', [])
            if labels:
                for label in labels:
                    pods_result = run_command([
                        'kubectl', 'get', 'pods'] + ns_flag + [
                        '-l', label,
                        '-o', 'json'
                    ], check=False)
                    
                    if pods_result.returncode == 0:
                        pods_data = json.loads(pods_result.stdout)
                        pods = pods_data.get('items', [])
                        
                        for pod in pods:
                            pod_info = {
                                'name': pod.get('metadata', {}).get('name'),
                                'namespace': pod.get('metadata', {}).get('namespace'),
                                'status': pod.get('status', {}).get('phase'),
                                'ready': all(
                                    c.get('ready', False)
                                    for c in pod.get('status', {}).get('containerStatuses', [])
                                )
                            }
                            result['pods'].append(pod_info)
                            
                            if pod_info['status'] == 'Running':
                                result['installed'] = True
                                
        except Exception as e:
            logger.warning(f"Error checking pods for addon {addon_name}: {e}")
        
        # Check for deployments
        try:
            deployments = config.get('deployments', [])
            for deployment_name in deployments:
                deploy_result = run_command([
                    'kubectl', 'get', 'deployment', deployment_name
                ] + ns_flag + ['-o', 'json'
                ], check=False)
                
                if deploy_result.returncode == 0:
                    deploy_data = json.loads(deploy_result.stdout)
                    deploy_info = {
                        'name': deploy_data.get('metadata', {}).get('name'),
                        'namespace': deploy_data.get('metadata', {}).get('namespace'),
                        'replicas': deploy_data.get('spec', {}).get('replicas', 0),
                        'ready_replicas': deploy_data.get('status', {}).get('readyReplicas', 0)
                    }
                    result['deployments'].append(deploy_info)
                    result['installed'] = True
                    
                    if deploy_info['ready_replicas'] >= deploy_info['replicas']:
                        result['running'] = True
                        
        except Exception as e:
            logger.warning(f"Error checking deployments for addon {addon_name}: {e}")
        
        # Check for services
        try:
            svc_result = run_command([
                'kubectl', 'get', 'services'
            ] + ns_flag + ['-o', 'json'
            ], check=False)
            
            if svc_result.returncode == 0:
                svc_data = json.loads(svc_result.stdout)
                services = svc_data.get('items', [])
                
                for svc in services:
                    svc_name = svc.get('metadata', {}).get('name', '')
                    if addon_name.lower() in svc_name.lower():
                        svc_info = {
                            'name': svc_name,
                            'namespace': svc.get('metadata', {}).get('namespace'),
                            'type': svc.get('spec', {}).get('type'),
                            'cluster_ip': svc.get('spec', {}).get('clusterIP')
                        }
                        result['services'].append(svc_info)
                        
        except Exception as e:
            logger.warning(f"Error checking services for addon {addon_name}: {e}")
    
    # Determine if addon is running (check both phase and container readiness)
    if result['pods']:
        running_pods = sum(
            1 for pod in result['pods']
            if pod['status'] == 'Running' and pod.get('ready', False)
        )
        if running_pods > 0:
            result['running'] = True
            logger.info(f"Addon {addon_name} is running ({running_pods} pods)")
        else:
            result['issues'].append(f"Addon {addon_name} has pods but none are fully ready")
    elif result['deployments']:
        ready_deploys = sum(
            1 for d in result['deployments']
            if d.get('ready_replicas', 0) >= d.get('replicas', 0)
        )
        if ready_deploys > 0:
            result['running'] = True
            logger.info(f"Addon {addon_name} is running ({ready_deploys} deployments)")
    
    if not result['installed']:
        result['issues'].append(f"Addon {addon_name} does not appear to be installed")
        logger.warning(f"Addon {addon_name} not found")
    elif not result['running']:
        result['issues'].append(f"Addon {addon_name} is installed but not running properly")
        logger.warning(f"Addon {addon_name} is not running properly")
    
    return result


def _parse_cpu(value: str) -> int:
    """
    Parse Kubernetes CPU resource value to millicores (int).
    
    Handles formats:
    - '4' -> 4000 (whole cores to millicores)
    - '31850m' -> 31850 (already millicores)
    - '500m' -> 500
    
    Returns:
        CPU value in millicores
    """
    value = str(value).strip()
    if value.endswith('m'):
        try:
            return int(value[:-1])
        except (ValueError, TypeError):
            return 0
    else:
        try:
            return int(value) * 1000
        except (ValueError, TypeError):
            return 0


def _parse_memory(value: str) -> int:
    """
    Parse Kubernetes memory resource value to bytes (int).
    
    Handles formats:
    - '128974848' -> 128974848 (plain bytes)
    - '129000Ki' -> 129000 * 1024
    - '128Mi' -> 128 * 1024 * 1024
    - '2Gi' -> 2 * 1024 * 1024 * 1024
    - '1Ti' -> 1 * 1024 * 1024 * 1024 * 1024
    
    Returns:
        Memory value in bytes
    """
    value = str(value).strip()
    suffixes = {
        'Ki': 1024,
        'Mi': 1024 ** 2,
        'Gi': 1024 ** 3,
        'Ti': 1024 ** 4,
        'Pi': 1024 ** 5,
        'K': 1000,
        'M': 1000 ** 2,
        'G': 1000 ** 3,
        'T': 1000 ** 4,
    }
    for suffix, multiplier in suffixes.items():
        if value.endswith(suffix):
            try:
                return int(value[:-len(suffix)]) * multiplier
            except (ValueError, TypeError):
                return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def _format_memory(bytes_val: int) -> str:
    """Format a memory value in bytes to a human-readable string with appropriate unit."""
    if bytes_val <= 0:
        return '0'
    if bytes_val >= 1024 ** 4:
        return f"{bytes_val / (1024 ** 4):.1f}Ti"
    if bytes_val >= 1024 ** 3:
        return f"{bytes_val / (1024 ** 3):.1f}Gi"
    if bytes_val >= 1024 ** 2:
        return f"{bytes_val / (1024 ** 2):.1f}Mi"
    if bytes_val >= 1024:
        return f"{bytes_val // 1024}Ki"
    return str(bytes_val)


def get_node_resources() -> Dict[str, Any]:
    """
    Get detailed resource information for all nodes in the cluster.
    
    This function retrieves:
    - CPU capacity and allocatable
    - Memory capacity and allocatable
    - GPU capacity and allocatable
    - EFA capacity
    - Node labels and conditions
    
    Returns:
        Dictionary containing:
        {
            'nodes': List[Dict],
            'total_cpu_capacity': int,
            'total_cpu_allocatable': int,
            'total_memory_capacity': str,
            'total_memory_allocatable': str,
            'total_gpu_capacity': int,
            'total_gpu_allocatable': int,
            'timestamp': str,
            'issues': List[str]
        }
        
    Raises:
        KubectlError: If kubectl commands fail
    """
    logger.info("Getting node resources...")
    
    result = {
        'nodes': [],
        'total_cpu_capacity': 0,
        'total_cpu_allocatable': 0,
        'total_memory_capacity': '0',
        'total_memory_allocatable': '0',
        'total_gpu_capacity': 0,
        'total_gpu_allocatable': 0,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'issues': []
    }
    
    # Check if kubectl is available
    if not check_kubectl_installed():
        result['issues'].append("kubectl is not available")
        return result
    
    total_memory_capacity_bytes = 0
    total_memory_allocatable_bytes = 0
    
    try:
        nodes_result = run_command([
            'kubectl', 'get', 'nodes',
            '-o', 'json'
        ])
        
        nodes_data = json.loads(nodes_result.stdout)
        nodes = nodes_data.get('items', [])
        
        for node in nodes:
            node_name = node.get('metadata', {}).get('name', 'unknown')
            labels = node.get('metadata', {}).get('labels', {})
            capacity = node.get('status', {}).get('capacity', {})
            allocatable = node.get('status', {}).get('allocatable', {})
            conditions = node.get('status', {}).get('conditions', [])
            
            # Get node conditions
            node_conditions = {}
            for condition in conditions:
                condition_type = condition.get('type')
                condition_status = condition.get('status')
                node_conditions[condition_type] = condition_status
            
            # Parse CPU
            cpu_capacity = capacity.get('cpu', '0')
            cpu_allocatable = allocatable.get('cpu', '0')
            
            # Parse memory
            memory_capacity = capacity.get('memory', '0')
            memory_allocatable = allocatable.get('memory', '0')
            
            # Parse GPU
            gpu_capacity = capacity.get('nvidia.com/gpu', '0')
            gpu_allocatable = allocatable.get('nvidia.com/gpu', '0')
            
            # Parse EFA
            efa_capacity = capacity.get('vpc.amazonaws.com/efa', '0')
            
            # Parse pods
            pods_capacity = capacity.get('pods', '0')
            pods_allocatable = allocatable.get('pods', '0')
            
            node_info = {
                'name': node_name,
                'instance_type': labels.get('node.kubernetes.io/instance-type', 'unknown'),
                'role': labels.get('kubernetes.io/role', 'worker'),
                'zone': labels.get('topology.kubernetes.io/zone', 'unknown'),
                'region': labels.get('topology.kubernetes.io/region', 'unknown'),
                'capacity': {
                    'cpu': cpu_capacity,
                    'memory': memory_capacity,
                    'gpu': int(gpu_capacity) if gpu_capacity.isdigit() else 0,
                    'efa': int(efa_capacity) if efa_capacity.isdigit() else 0,
                    'pods': int(pods_capacity) if pods_capacity.isdigit() else 0
                },
                'allocatable': {
                    'cpu': cpu_allocatable,
                    'memory': memory_allocatable,
                    'gpu': int(gpu_allocatable) if gpu_allocatable.isdigit() else 0,
                    'pods': int(pods_allocatable) if pods_allocatable.isdigit() else 0
                },
                'conditions': node_conditions,
                'ready': node_conditions.get('Ready') == 'True'
            }
            
            result['nodes'].append(node_info)
            
            # Accumulate totals
            result['total_cpu_capacity'] += _parse_cpu(cpu_capacity)
            result['total_cpu_allocatable'] += _parse_cpu(cpu_allocatable)
            
            total_memory_capacity_bytes += _parse_memory(memory_capacity)
            total_memory_allocatable_bytes += _parse_memory(memory_allocatable)
            
            try:
                result['total_gpu_capacity'] += int(gpu_capacity) if gpu_capacity.isdigit() else 0
            except (ValueError, TypeError):
                pass
                
            try:
                result['total_gpu_allocatable'] += int(gpu_allocatable) if gpu_allocatable.isdigit() else 0
            except (ValueError, TypeError):
                pass
        
        # Format memory totals
        result['total_memory_capacity'] = _format_memory(total_memory_capacity_bytes)
        result['total_memory_allocatable'] = _format_memory(total_memory_allocatable_bytes)
        
        logger.info(f"Retrieved resources for {len(result['nodes'])} nodes")
        
    except Exception as e:
        error_msg = f"Error getting node resources: {str(e)}"
        logger.error(error_msg)
        result['issues'].append(error_msg)
    
    return result


def run_full_cluster_check(cluster_name: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a comprehensive check of the cluster including all components.
    
    This is a convenience function that runs all health checks in sequence.
    
    Args:
        cluster_name: Name of the EKS cluster
        region: AWS region (optional)
        
    Returns:
        Dictionary containing all check results:
        {
            'cluster_health': Dict,
            'gpu_availability': Dict,
            'efa_availability': Dict,
            'node_resources': Dict,
            'addons': Dict[str, Dict],
            'overall_status': str,
            'timestamp': str
        }
    """
    logger.info(f"Running full cluster check for: {cluster_name}")
    
    result = {
        'cluster_name': cluster_name,
        'cluster_health': {},
        'gpu_availability': {},
        'efa_availability': {},
        'node_resources': {},
        'addons': {},
        'overall_status': 'unknown',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    # Run cluster health check
    try:
        result['cluster_health'] = check_cluster_health(cluster_name, region)
    except Exception as e:
        logger.error(f"Cluster health check failed: {e}")
        result['cluster_health'] = {'error': str(e)}
    
    # Run GPU check
    try:
        result['gpu_availability'] = check_gpu_availability()
    except Exception as e:
        logger.error(f"GPU check failed: {e}")
        result['gpu_availability'] = {'error': str(e)}
    
    # Run EFA check
    try:
        result['efa_availability'] = check_efa_availability()
    except Exception as e:
        logger.error(f"EFA check failed: {e}")
        result['efa_availability'] = {'error': str(e)}
    
    # Run node resources check
    try:
        result['node_resources'] = get_node_resources()
    except Exception as e:
        logger.error(f"Node resources check failed: {e}")
        result['node_resources'] = {'error': str(e)}
    
    # Check common addons
    addons_to_check = ['coredns', 'kube-proxy', 'nvidia-device-plugin', 'aws-efa']
    for addon in addons_to_check:
        try:
            result['addons'][addon] = check_addon_status(addon)
        except Exception as e:
            logger.error(f"Addon check failed for {addon}: {e}")
            result['addons'][addon] = {'error': str(e)}
    
    # Determine overall status
    statuses = []
    if 'status' in result['cluster_health']:
        statuses.append(result['cluster_health']['status'])
    
    if all(s == 'healthy' for s in statuses):
        result['overall_status'] = 'healthy'
    elif any(s == 'unhealthy' for s in statuses):
        result['overall_status'] = 'unhealthy'
    else:
        result['overall_status'] = 'degraded'
    
    logger.info(f"Full cluster check complete. Overall status: {result['overall_status']}")
    return result


if __name__ == '__main__':
    # Example usage when run as a script
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Kubernetes Cluster Health Checker'
    )
    parser.add_argument(
        'cluster_name',
        help='Name of the EKS cluster to check'
    )
    parser.add_argument(
        '--region',
        help='AWS region',
        default=None
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full cluster check'
    )
    parser.add_argument(
        '--check',
        choices=['health', 'gpu', 'efa', 'resources', 'addons'],
        help='Specific check to run'
    )
    parser.add_argument(
        '--addon-name',
        help='Addon name for addon check'
    )
    parser.add_argument(
        '--output',
        choices=['json', 'yaml', 'text'],
        default='json',
        help='Output format'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.full:
            output = run_full_cluster_check(args.cluster_name, args.region)
        elif args.check == 'health':
            output = check_cluster_health(args.cluster_name, args.region)
        elif args.check == 'gpu':
            output = check_gpu_availability()
        elif args.check == 'efa':
            output = check_efa_availability()
        elif args.check == 'resources':
            output = get_node_resources()
        elif args.check == 'addons':
            if args.addon_name:
                output = check_addon_status(args.addon_name)
            else:
                print("Error: --addon-name required for addon check")
                sys.exit(1)
        else:
            # Default to cluster health check
            output = check_cluster_health(args.cluster_name, args.region)
        
        # Output results
        if args.output == 'json':
            print(json.dumps(output, indent=2))
        elif args.output == 'yaml':
            try:
                import yaml
                print(yaml.dump(output, default_flow_style=False))
            except ImportError:
                print(json.dumps(output, indent=2))
        else:
            # Text output
            print(f"Cluster: {args.cluster_name}")
            print(f"Status: {output.get('status', 'unknown')}")
            if 'issues' in output and output['issues']:
                print("Issues:")
                for issue in output['issues']:
                    print(f"  - {issue}")
                    
    except Exception as e:
        logger.error(f"Check failed: {e}")
        sys.exit(1)
