"""
Ray Cluster Manager - Manage Ray and KubeRay clusters on EKS
"""

from .ray_manager import (
    check_kuberay_installed,
    install_kuberay,
    create_raycluster,
    delete_raycluster,
    generate_raycluster_yaml,
    get_raycluster_status,
    list_rayclusters,
    scale_raycluster,
    get_raycluster_endpoint,
    wait_for_raycluster,
    RayClusterError
)

__all__ = [
    'check_kuberay_installed',
    'install_kuberay',
    'create_raycluster',
    'delete_raycluster',
    'generate_raycluster_yaml',
    'get_raycluster_status',
    'list_rayclusters',
    'scale_raycluster',
    'get_raycluster_endpoint',
    'wait_for_raycluster',
    'RayClusterError'
]
