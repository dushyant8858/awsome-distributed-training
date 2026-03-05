# HyperPod Manager Skill

Manage, create, and monitor AWS SageMaker HyperPod clusters with Kubernetes integration.

## Overview

This skill provides utilities for:
- **Creating HyperPod clusters** with EKS orchestrator, capacity pre-checks, and deep health checks
- **Lifecycle scripts** upload to S3 for instance provisioning
- **HyperPod Helm dependencies** (NVIDIA, EFA, health monitoring, Kueue, etc.)
- **cert-manager installation** (required for HPTO)
- **Observability addon** (AMP + DCGM metrics, NO public endpoints)
- **Training operator addon** (HPTO with Pod Identity + cert-manager)
- Listing, describing, scaling, and deleting HyperPod clusters
- 3-layer capacity checking (quotas, AZ offerings, EC2 dry-run)
- Querying Amazon Managed Prometheus (AMP) for metrics
- Checking node health and observability pods

## Correct Deployment Order (CRITICAL)

The order matters. Deviating causes hard-to-debug failures.

```
1. provision_eks_infrastructure()     # eks-cluster-manager skill
2. install_hyperpod_dependencies()    # Helm chart (NVIDIA, EFA, health, Kueue)
3. setup_lifecycle_scripts()          # Upload on_create.sh to S3
4. create_hyperpod_cluster()          # Creates instances (auto-adds SageMaker=true tag)
5. install_observability_addon()      # AMP + DCGM (pods run on HyperPod nodes)
6. install_training_operator()        # Installs cert-manager + HPTO addon
```

**Why this order?**
- Step 2 must precede step 4: HyperPod `CreateCluster` fails if the Helm chart dependencies are missing ("missing one or more required dependencies")
- Step 4 must precede steps 5-6: Addon pods have `nodeSelector: sagemaker.amazonaws.com/compute-type: hyperpod` and need HyperPod nodes to schedule
- Step 6 auto-installs cert-manager: The HPTO addon requires cert-manager for webhook TLS certificates
- Step 4 auto-adds `SageMaker=true` tag: The HPTO managed policy has an IAM condition `StringEquals: { aws:ResourceTag/SageMaker: "true" }` on `DescribeClusterNode`

## Functions

### Prerequisites & Dependencies

### `install_hyperpod_dependencies(eks_cluster_name=None, region=None, helm_release="hyperpod-dependencies", namespace="kube-system", timeout=600)`

Install the HyperPod dependencies Helm chart. **MANDATORY before `create_hyperpod_cluster()`.**

The chart installs: NVIDIA device plugin, AWS EFA device plugin, health monitoring agent, deep health check, job auto-restart, Kubeflow training operator, MPI operator, Kueue, Neuron device plugin.

**Important:** Does NOT use `--wait` because no worker nodes exist yet when installing (pods will be Pending until HyperPod adds nodes).

```python
deps = install_hyperpod_dependencies(eks_cluster_name="my-eks")
# deps = {"release_name": "hyperpod-dependencies", "namespace": "kube-system", "status": "installed"}
```

### `install_cert_manager(eks_cluster_name=None, version="v1.17.1", region=None, wait_timeout=120)`

Install cert-manager on the EKS cluster. **Required prerequisite for HPTO.** Called automatically by `install_training_operator()`.

```python
cm = install_cert_manager(eks_cluster_name="my-eks")
# cm = {"version": "v1.17.1", "namespace": "cert-manager", "status": "installed"}
```

### Cluster Creation

### `setup_lifecycle_scripts(s3_bucket, s3_prefix="", custom_on_create_main=None, region=None)`
Upload lifecycle scripts to S3. Includes a default `on_create_main.sh` that:
- Moves containerd/kubelet to a larger disk (supports AL2 and AL2023)
- Loads Lustre kernel modules

```python
scripts = setup_lifecycle_scripts("my-bucket", s3_prefix="lifecycle")
# scripts = {
#   "s3_uri": "s3://my-bucket/lifecycle",
#   "on_create_script": "on_create.sh",
#   ...
# }
```

### `create_hyperpod_cluster(cluster_name, eks_cluster_arn, instance_groups, security_group_ids, subnet_ids, node_recovery="Automatic", pre_check_capacity=True, tags=None, region=None)`
Create a HyperPod cluster with EKS orchestrator. Runs a 3-layer capacity pre-check by default and enables deep health checks (InstanceStress + InstanceConnectivity) on all instance groups.

**Automatically adds `SageMaker=true` tag** (required by the HPTO managed policy).

```python
result = create_hyperpod_cluster(
    cluster_name="my-cluster",
    eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/my-eks",
    instance_groups=[{
        "InstanceGroupName": "gpu-workers",
        "InstanceType": "ml.g5.12xlarge",
        "InstanceCount": 2,
        "ExecutionRole": "arn:aws:iam::123456789012:role/my-exec-role",
        "LifeCycleConfig": {
            "OnCreate": "on_create.sh",
            "SourceS3Uri": "s3://my-bucket/lifecycle",
        },
        "InstanceStorageConfigs": [{"EbsVolumeConfig": {"VolumeSizeInGB": 500}}],
    }],
    security_group_ids=["sg-xxx"],
    subnet_ids=["subnet-xxx"],
)
```

### Addons

### `install_observability_addon(eks_cluster_name, eks_cluster_arn, name_prefix="sagemaker-hyperpod", training_metrics_level="BASIC", accelerated_compute_metrics_level="BASIC", node_metrics_level="BASIC", custom_metrics_level="BASIC", scrape_interval=30, create_amp_workspace=True, amp_workspace_id=None, region=None)`

Install the HyperPod observability stack:
1. Creates an AMP (Amazon Managed Prometheus) workspace
2. Creates IAM role with Pod Identity (SourceAccount + SourceArn conditions)
3. Installs `amazon-sagemaker-hyperpod-observability` EKS addon with DCGM/training/node metrics

**Security: NO public Grafana endpoint is created.** Use `kubectl port-forward` for dashboards.

```python
obs = install_observability_addon(
    eks_cluster_name="my-eks",
    eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/my-eks",
)
# obs = {"amp_workspace_id": "ws-xxx", "amp_endpoint": "https://...", "role_arn": "...", "addon_name": "..."}
```

### `install_training_operator(eks_cluster_name, name_prefix="sagemaker-hyperpod", region=None)`

Install the HyperPod Training Operator (HPTO):
1. **Installs cert-manager** (required prerequisite - auto-handled)
2. Creates IAM role with Pod Identity (no conditions - simpler trust than observability)
3. Attaches `AmazonSageMakerHyperPodTrainingOperatorAccess` managed policy
4. Installs `amazon-sagemaker-hyperpod-training-operator` EKS addon

```python
hpto = install_training_operator(eks_cluster_name="my-eks")
# hpto = {
#   "cert_manager_status": "installed",
#   "role_arn": "...",
#   "addon_name": "...",
#   "service_account": "hp-training-operator-controller-manager",
# }
```

### Cluster Lifecycle

### `list_clusters(region=None)`
List all HyperPod clusters in the account/region.

### `describe_cluster(cluster_name, region=None)`
Get full cluster details including instance groups, VPC config, orchestrator, and status.

### `scale_instance_group(cluster_name, group_name, target_count, region=None)`
Scale a HyperPod instance group up or down. Preserves all existing configuration (lifecycle config, execution role, storage, VPC overrides). Set `target_count=0` to scale down completely while keeping the cluster alive.

```python
# Scale down to 0 (stop all instances, keep cluster)
scale_instance_group("my-cluster", "my-ig-8x", 0)

# Scale back up to 4 instances
scale_instance_group("my-cluster", "my-ig-8x", 4)
```

### `wait_for_cluster_status(cluster_name, target_status="InService", timeout=600, poll_interval=30, region=None)`
Wait for a cluster to reach a target status (e.g., after scaling). Checks both cluster status and instance group current vs target counts.

### `delete_cluster(cluster_name, region=None, confirm=True)`
Delete a HyperPod cluster entirely. WARNING: destructive operation. All instances are terminated. Shared PVCs (EBS/EFS/FSx) are NOT deleted.

### Node & Health

### `is_hyperpod_cluster()`
Detect if the current cluster is a SageMaker HyperPod cluster by checking for the `sagemaker.amazonaws.com/compute-type` label on nodes.

### `get_hyperpod_nodes()`
Get all HyperPod nodes with their metadata and labels.

### `check_node_health()`
Check the health status of HyperPod nodes.

### `get_observability_pods()`
Get observability-related pods (DCGM exporter, Prometheus, etc.).

### Capacity

### `check_capacity(instance_type, requested_count, cluster_name=None, region=None)`
Three-layer pre-flight check before scaling:

1. **Service Quotas** - Per-instance-type and total-cluster quotas vs current usage
2. **AZ Offerings** - Whether the EC2 instance type is offered in the cluster's target AZs (catches subnet/AZ mismatches)
3. **EC2 Dry Run** - Best-effort probe of actual compute capacity using CreateFleet DryRun

```python
result = check_capacity("ml.g5.12xlarge", 2, cluster_name="my-cluster")
if result["feasible"]:
    scale_instance_group("my-cluster", "my-ig", 2)
else:
    for issue in result["issues"]:
        print(f"  BLOCKED: {issue}")
```

### Metrics

### `query_amp(workspace_id, query)`
Query Amazon Managed Prometheus using the workspace ID and PromQL query.

### `get_hyperpod_metrics(workspace_id=None)`
Get common HyperPod metrics including GPU utilization, memory, and health status.

## Full Example: Create a Complete HyperPod Stack

```python
from hyperpod_manager import (
    install_hyperpod_dependencies, setup_lifecycle_scripts,
    create_hyperpod_cluster, install_observability_addon,
    install_training_operator, wait_for_cluster_status, describe_cluster,
)

# 1. Install HyperPod Helm dependencies (NVIDIA, EFA, health monitoring, Kueue)
deps = install_hyperpod_dependencies(eks_cluster_name="my-eks")

# 2. Upload lifecycle scripts
scripts = setup_lifecycle_scripts("my-s3-bucket")

# 3. Create HyperPod cluster (capacity pre-check runs automatically)
#    Automatically adds SageMaker=true tag required by HPTO
cluster = create_hyperpod_cluster(
    cluster_name="my-training-cluster",
    eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/my-eks",
    instance_groups=[{
        "InstanceGroupName": "gpu-workers",
        "InstanceType": "ml.g5.12xlarge",
        "InstanceCount": 2,
        "ExecutionRole": "arn:aws:iam::123456789012:role/my-exec-role",
        "LifeCycleConfig": {
            "OnCreate": scripts["on_create_script"],
            "SourceS3Uri": scripts["s3_uri"],
        },
        "InstanceStorageConfigs": [{"EbsVolumeConfig": {"VolumeSizeInGB": 500}}],
    }],
    security_group_ids=["sg-xxx"],
    subnet_ids=["subnet-xxx"],
)

# 4. Wait for cluster to be ready (~15-25 min)
wait_for_cluster_status("my-training-cluster", timeout=1800)

# 5. Install addons (observability + training operator)
obs = install_observability_addon(
    eks_cluster_name="my-eks",
    eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/my-eks",
)
hpto = install_training_operator(eks_cluster_name="my-eks")
# ^ This automatically installs cert-manager first

# 6. Verify
info = describe_cluster("my-training-cluster")
for group in info["summary"]["instance_groups"]:
    print(f"  {group['name']}: {group['current_count']}/{group['target_count']} {group['instance_type']}")
```

## Known Gotchas

1. **Helm chart MUST be installed before `create_hyperpod_cluster()`** - Otherwise CreateCluster fails with "missing one or more required dependencies"
2. **cert-manager MUST be installed before HPTO addon** - Otherwise addon fails with "cert-manager is not installed on this cluster"
3. **`SageMaker=true` tag is required on the HyperPod cluster** - The HPTO managed policy has an IAM condition requiring this tag. `create_hyperpod_cluster()` adds it automatically.
4. **g5 instances don't support GPUDirect RDMA** - Set `FI_EFA_USE_DEVICE_RDMA=0` and `NCCL_PROTO=simple`
5. **g5.12xlarge not available in all AZs** - e.g., not in us-west-2d. The capacity checker catches this.
6. **`UpdateCluster` cannot add `OverrideVpcConfig` to existing groups** - Must create a NEW instance group.
7. **Addon pods need HyperPod nodes** - They have `nodeSelector: sagemaker.amazonaws.com/compute-type: hyperpod`. Install addons AFTER cluster is InService.
8. **Helm install should NOT use `--wait`** - No worker nodes exist yet at install time. DaemonSets will be Pending until HyperPod adds nodes.

## Kubernetes Version Compatibility

Both HyperPod addons support Kubernetes 1.28-1.34+:
- **HPTO**: K8s 1.28 - 1.34 (v1.2.0-eksbuild.1)
- **Observability**: K8s 1.30 - 1.35 (v1.0.5-eksbuild.2)

Recommended: K8s 1.31 or 1.32 for best compatibility with both addons.

## Security Guardrails

- **No public Grafana**: `install_observability_addon()` creates AMP only, no AMG workspace
- **Scoped IAM roles**: Observability role uses `SourceAccount` + `SourceArn` conditions
- **Capacity pre-checks**: `create_hyperpod_cluster()` blocks if quota/AZ/EC2 capacity insufficient
- **Deep health checks**: InstanceStress + InstanceConnectivity enabled by default
- **Automatic node recovery**: Enabled by default
- **Required tag enforcement**: `SageMaker=true` tag always added for HPTO compatibility

## Requirements

- boto3
- kubectl configured with cluster access
- helm (for `install_hyperpod_dependencies()`)
- git (for cloning the HyperPod Helm chart repo)
- AWS credentials with permissions for: SageMaker, EKS, IAM, S3, STS, AMP, EC2, Service Quotas
