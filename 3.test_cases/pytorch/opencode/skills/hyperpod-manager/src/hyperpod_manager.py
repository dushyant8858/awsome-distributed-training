"""HyperPod Manager - Main implementation for AWS SageMaker HyperPod cluster management."""

import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

import logging
logger = logging.getLogger("hyperpod_manager")
if not logger.handlers:
    import sys as _sys
    _h = logging.StreamHandler(_sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def _get_region() -> str:
    return boto3.Session().region_name or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


def _get_account_id() -> str:
    return boto3.client("sts").get_caller_identity()["Account"]

HYPERPOD_LABEL = "sagemaker.amazonaws.com/compute-type"
HYPERPOD_LABEL_VALUE = "hyperpod"
HYPERPOD_GROUP_LABEL = "sagemaker.amazonaws.com/instance-group-name"


def _run_kubectl(args: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run kubectl command with error handling."""
    cmd = ["kubectl"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"kubectl command failed: {' '.join(cmd)}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("kubectl not found in PATH")
        raise


def is_hyperpod_cluster() -> bool:
    """Detect if the current cluster is a SageMaker HyperPod cluster.
    
    Checks for the presence of nodes with the HyperPod label.
    
    Returns:
        bool: True if HyperPod cluster detected, False otherwise.
    """
    try:
        result = _run_kubectl([
            "get", "nodes",
            "-l", f"{HYPERPOD_LABEL}={HYPERPOD_LABEL_VALUE}",
            "-o", "json"
        ])
        data = json.loads(result.stdout)
        nodes = data.get("items", [])
        is_hyperpod = len(nodes) > 0
        logger.info(f"HyperPod cluster detected: {is_hyperpod} ({len(nodes)} nodes)")
        return is_hyperpod
    except Exception as e:
        logger.warning(f"Could not detect HyperPod cluster: {e}")
        return False


def get_hyperpod_nodes() -> List[Dict[str, Any]]:
    """Get all HyperPod nodes with their metadata and labels.
    
    Returns:
        List of node dictionaries containing name, labels, and status.
    """
    try:
        result = _run_kubectl([
            "get", "nodes",
            "-l", f"{HYPERPOD_LABEL}={HYPERPOD_LABEL_VALUE}",
            "-o", "json"
        ])
        data = json.loads(result.stdout)
        nodes = []
        
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            status = item.get("status", {})
            
            node_info = {
                "name": metadata.get("name"),
                "labels": metadata.get("labels", {}),
                "node_type": metadata.get("labels", {}).get(HYPERPOD_LABEL),
                "node_group": metadata.get("labels", {}).get(HYPERPOD_GROUP_LABEL),
                "instance_type": metadata.get("labels", {}).get("node.kubernetes.io/instance-type"),
                "capacity": status.get("capacity", {}),
                "allocatable": status.get("allocatable", {}),
                "conditions": status.get("conditions", []),
                "addresses": status.get("addresses", []),
            }
            nodes.append(node_info)
        
        logger.info(f"Found {len(nodes)} HyperPod nodes")
        return nodes
    except Exception as e:
        logger.error(f"Failed to get HyperPod nodes: {e}")
        return []


def query_amp(workspace_id: str, query: str) -> Dict[str, Any]:
    """Query Amazon Managed Prometheus (AMP) using the workspace ID.
    
    Args:
        workspace_id: The AMP workspace ID.
        query: PromQL query string.
    
    Returns:
        Dictionary containing query results.
    """
    try:
        region = boto3.Session().region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
        client = boto3.client("amp", region_name=region)
        
        # Construct the query URL
        workspace_url = f"https://aps-workspaces.{region}.amazonaws.com/workspaces/{workspace_id}"
        
        # NOTE: The AMP boto3 client does not have a direct query() method.
        # Real AMP PromQL queries should use the query_metrics() API or make
        # signed HTTP requests to the workspace URL. This implementation needs
        # real AMP workspace testing to validate the correct API call.
        try:
            response = client.query_metrics(
                WorkspaceId=workspace_id,
                Query=query,
            )
        except (AttributeError, ClientError) as api_err:
            logger.error(
                f"AMP query API call failed: {api_err}. "
                "The AMP boto3 client may not support this method directly. "
                "Consider using sigv4-signed HTTP requests to the AMP workspace endpoint instead."
            )
            return {"error": str(api_err), "workspace_url": workspace_url, "query": query}
        
        logger.info(f"AMP query executed: {query[:50]}...")
        return response
    except ClientError as e:
        logger.error(f"AWS API error querying AMP: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to query AMP: {e}")
        raise


def get_hyperpod_metrics(workspace_id: Optional[str] = None) -> Dict[str, Any]:
    """Get common HyperPod metrics including GPU utilization and health.
    
    Args:
        workspace_id: Optional AMP workspace ID. If not provided, attempts to discover.
    
    Returns:
        Dictionary containing various metrics.
    """
    metrics = {
        "gpu_utilization": None,
        "gpu_memory": None,
        "gpu_temperature": None,
        "network_io": None,
        "timestamp": None
    }
    
    if not workspace_id:
        logger.warning("No AMP workspace ID provided, skipping metrics query")
        return metrics
    
    try:
        # GPU utilization
        metrics["gpu_utilization"] = query_amp(
            workspace_id,
            'DCGM_FI_DEV_GPU_UTIL{instance=~".*"}'
        )
        
        # GPU memory free
        metrics["gpu_memory"] = query_amp(
            workspace_id,
            'DCGM_FI_DEV_FB_FREE{instance=~".*"}'
        )
        
        # GPU temperature
        metrics["gpu_temperature"] = query_amp(
            workspace_id,
            'DCGM_FI_DEV_GPU_TEMP{instance=~".*"}'
        )
        
        # Network receive bytes
        metrics["network_io"] = query_amp(
            workspace_id,
            'node_network_receive_bytes_total{device="eth0"}'
        )
        
        logger.info("HyperPod metrics collected successfully")
    except Exception as e:
        logger.error(f"Failed to collect HyperPod metrics: {e}")
    
    return metrics


def check_node_health() -> List[Dict[str, Any]]:
    """Check the health status of HyperPod nodes.
    
    Returns:
        List of node health status dictionaries.
    """
    nodes = get_hyperpod_nodes()
    health_status = []
    
    for node in nodes:
        conditions = node.get("conditions", [])
        ready_condition = next(
            (c for c in conditions if c.get("type") == "Ready"),
            None
        )
        
        status = {
            "name": node["name"],
            "ready": ready_condition.get("status") == "True" if ready_condition else False,
            "ready_reason": ready_condition.get("reason") if ready_condition else "Unknown",
            "ready_message": ready_condition.get("message") if ready_condition else "No Ready condition found",
            "node_type": node.get("node_type"),
            "node_group": node.get("node_group"),
            "instance_type": node.get("instance_type"),
        }
        
        # Check for other conditions
        status["conditions"] = {
            c.get("type", "Unknown"): {
                "status": c.get("status"),
                "reason": c.get("reason"),
                "message": c.get("message")
            }
            for c in conditions
        }
        
        health_status.append(status)
    
    healthy_count = sum(1 for h in health_status if h["ready"])
    logger.info(f"Node health check: {healthy_count}/{len(health_status)} nodes ready")
    
    return health_status


# ---------------------------------------------------------------------------
# Capacity Checking
# ---------------------------------------------------------------------------


def _ml_to_ec2_instance_type(ml_type: str) -> str:
    """Map SageMaker ml.* instance type to the underlying EC2 instance type.

    Example: ml.g5.12xlarge -> g5.12xlarge
    """
    if ml_type.startswith("ml."):
        return ml_type[3:]
    return ml_type


def _find_quota(sq_client, service_code: str, quota_name: str) -> Optional[float]:
    """Search Service Quotas for a quota by name (applied value, falling back to default).

    Returns the quota value or None if not found.
    """
    # Try applied quotas first (user may have requested an increase)
    try:
        paginator = sq_client.get_paginator("list_service_quotas")
        for page in paginator.paginate(ServiceCode=service_code):
            for q in page.get("Quotas", []):
                if q.get("QuotaName") == quota_name:
                    return q["Value"]
    except ClientError:
        pass

    # Fall back to default quotas
    try:
        paginator = sq_client.get_paginator("list_aws_default_service_quotas")
        for page in paginator.paginate(ServiceCode=service_code):
            for q in page.get("Quotas", []):
                if q.get("QuotaName") == quota_name:
                    return q["Value"]
    except ClientError:
        pass

    return None


def _check_az_offerings(
    ec2_client, ec2_type: str, subnet_ids: Optional[List[str]] = None, region: Optional[str] = None,
) -> Dict[str, Any]:
    """Check which AZs offer the EC2 instance type, and whether the target subnets are in those AZs.

    Returns dict with:
    - offered_azs: list of AZ names where the type is available
    - subnet_azs: dict mapping subnet_id -> az (if subnet_ids provided)
    - all_subnets_covered: bool, True if every target subnet is in an offered AZ
    - issues: list of problems
    """
    result = {"offered_azs": [], "subnet_azs": {}, "all_subnets_covered": True, "issues": []}

    # 1. Which AZs offer this type?
    try:
        resp = ec2_client.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": [ec2_type]}],
        )
        result["offered_azs"] = sorted(
            o["Location"] for o in resp.get("InstanceTypeOfferings", [])
        )
    except ClientError as e:
        result["issues"].append(f"Error querying instance type offerings: {e}")
        return result

    if not result["offered_azs"]:
        result["issues"].append(
            f"{ec2_type} is not offered in any AZ in this region. "
            "The instance type may not be available here."
        )
        result["all_subnets_covered"] = False
        return result

    # 2. If subnet_ids given, resolve their AZs and check coverage
    if subnet_ids:
        try:
            resp = ec2_client.describe_subnets(SubnetIds=subnet_ids)
            for sn in resp.get("Subnets", []):
                result["subnet_azs"][sn["SubnetId"]] = sn["AvailabilityZone"]
        except ClientError as e:
            result["issues"].append(f"Error describing subnets: {e}")

        for sid, az in result["subnet_azs"].items():
            if az not in result["offered_azs"]:
                result["all_subnets_covered"] = False
                result["issues"].append(
                    f"Subnet {sid} is in {az}, but {ec2_type} is not offered there. "
                    f"Available AZs: {result['offered_azs']}"
                )

    return result


def _check_ec2_capacity_dry_run(
    ec2_client, ec2_type: str, count: int, subnet_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Attempt an EC2 dry-run to probe whether capacity is likely available.

    Uses CreateFleet with Type=instant and DryRun=True.  The API will raise
    a DryRunOperation error if the request *would* have succeeded, or an
    InsufficientInstanceCapacity / other error if it would have failed.

    This is best-effort: SageMaker manages its own fleet, so EC2 availability
    does not guarantee HyperPod success, but it catches hard failures.

    Returns dict with:
    - capacity_likely: bool (True if dry-run succeeded, None if inconclusive)
    - detail: human-readable string
    """
    result = {"capacity_likely": None, "detail": ""}

    launch_spec = {
        "LaunchTemplateSpecification": None,  # not used, but we need overrides
    }

    overrides = [{"InstanceType": ec2_type}]
    if subnet_id:
        overrides[0]["SubnetId"] = subnet_id

    try:
        ec2_client.create_fleet(
            DryRun=True,
            Type="instant",
            LaunchTemplateConfigs=[{
                "Overrides": overrides,
            }],
            TargetCapacitySpecification={
                "TotalTargetCapacity": count,
                "DefaultTargetCapacityType": "on-demand",
            },
        )
        # Should not reach here (DryRun always raises)
        result["capacity_likely"] = True
        result["detail"] = "Dry-run succeeded (unexpected path)"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]

        if code == "DryRunOperation":
            # This means the request WOULD have succeeded
            result["capacity_likely"] = True
            result["detail"] = "EC2 dry-run passed: capacity likely available"
        elif "InsufficientInstanceCapacity" in code or "InsufficientInstanceCapacity" in msg:
            result["capacity_likely"] = False
            result["detail"] = f"EC2 reports insufficient capacity for {count}x {ec2_type}"
        elif "Unsupported" in msg or "not supported" in msg.lower():
            result["capacity_likely"] = False
            result["detail"] = f"{ec2_type} not supported: {msg}"
        else:
            # Other errors (permissions, missing launch template, etc.) - inconclusive
            result["capacity_likely"] = None
            result["detail"] = f"EC2 dry-run inconclusive ({code}): {msg}"

    return result


def check_capacity(
    instance_type: str,
    requested_count: int,
    cluster_name: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Check whether HyperPod capacity is available for an instance type.

    Performs a three-layer check:
    1. **Service Quota** - Per-type and total-cluster quotas vs current usage.
    2. **AZ Offerings** - Whether the EC2 instance type is offered in the
       target AZs (derived from the cluster's subnet config).
    3. **EC2 Dry Run** - Best-effort probe of actual compute capacity in the
       target AZ using CreateFleet DryRun.

    Args:
        instance_type: ML instance type (e.g. "ml.g5.12xlarge").
        requested_count: Number of additional instances you want to launch.
        cluster_name: Optional HyperPod cluster name. If provided, the check
                      resolves the cluster's subnets to verify AZ availability
                      and runs the dry-run in the correct subnet.
        region: AWS region. Auto-detected if not provided.

    Returns:
        Dictionary with:
        - feasible (bool): True if all checks pass.
        - quota_ok (bool): Service quota check passed.
        - az_ok (bool | None): AZ offering check passed (None if skipped).
        - ec2_capacity_ok (bool | None): EC2 dry-run passed (None if skipped/inconclusive).
        - instance_type_quota: Per-type quota limit.
        - instance_type_in_use: Currently running instances of this type.
        - total_cluster_quota: Total instances across all clusters.
        - total_in_use: Total instances currently running.
        - offered_azs: AZs where the EC2 type is available.
        - target_subnet_azs: Subnet-to-AZ mapping for the cluster.
        - issues: List of human-readable issues (empty if feasible).
    """
    region = region or boto3.Session().region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
    sm_client = boto3.client("sagemaker", region_name=region)
    sq_client = boto3.client("service-quotas", region_name=region)
    ec2_client = boto3.client("ec2", region_name=region)

    ec2_type = _ml_to_ec2_instance_type(instance_type)

    result = {
        "instance_type": instance_type,
        "ec2_instance_type": ec2_type,
        "requested_count": requested_count,
        "feasible": True,
        "quota_ok": True,
        "az_ok": None,
        "ec2_capacity_ok": None,
        "instance_type_quota": None,
        "instance_type_in_use": 0,
        "total_cluster_quota": None,
        "total_in_use": 0,
        "offered_azs": [],
        "target_subnet_azs": {},
        "issues": [],
    }

    # =====================================================================
    # Layer 1: Service Quota Check
    # =====================================================================
    logger.info(f"[1/3] Checking service quotas for {instance_type}...")

    quota_name_pattern = f"{instance_type} for cluster usage"
    try:
        per_type_quota = _find_quota(sq_client, "sagemaker", quota_name_pattern)
        if per_type_quota is not None:
            result["instance_type_quota"] = int(per_type_quota)
        else:
            result["issues"].append(
                f"Could not find Service Quota for '{quota_name_pattern}'. "
                "The instance type may not be supported for HyperPod in this region."
            )
            result["feasible"] = False
            result["quota_ok"] = False
    except Exception as e:
        result["issues"].append(f"Error querying Service Quotas: {e}")

    try:
        total_quota = _find_quota(
            sq_client, "sagemaker",
            "Total number of instances allowed across SageMaker HyperPod clusters"
        )
        if total_quota is not None:
            result["total_cluster_quota"] = int(total_quota)
    except Exception as e:
        result["issues"].append(f"Error querying total cluster quota: {e}")

    # Scan current usage
    try:
        paginator = sm_client.get_paginator("list_clusters")
        for page in paginator.paginate():
            for c in page.get("ClusterSummaries", []):
                cname = c["ClusterName"]
                if c.get("ClusterStatus") in ("Failed", "Deleting"):
                    continue
                try:
                    desc = sm_client.describe_cluster(ClusterName=cname)
                    for ig in desc.get("InstanceGroups", []):
                        cur = ig.get("CurrentCount", 0)
                        tgt = ig.get("TargetCount", 0)
                        count = max(cur, tgt)
                        result["total_in_use"] += count
                        if ig.get("InstanceType") == instance_type:
                            result["instance_type_in_use"] += count
                except ClientError:
                    pass
    except Exception as e:
        result["issues"].append(f"Error scanning clusters for usage: {e}")

    type_quota = result["instance_type_quota"]
    if type_quota is not None:
        available = type_quota - result["instance_type_in_use"]
        if type_quota == 0:
            result["feasible"] = False
            result["quota_ok"] = False
            result["issues"].append(
                f"Quota for {instance_type} is 0. "
                "Request a quota increase via the AWS Service Quotas console."
            )
        elif requested_count > available:
            result["feasible"] = False
            result["quota_ok"] = False
            result["issues"].append(
                f"Insufficient {instance_type} quota: "
                f"quota={type_quota}, in_use={result['instance_type_in_use']}, "
                f"available={available}, requested={requested_count}"
            )

    total_quota = result["total_cluster_quota"]
    if total_quota is not None:
        total_available = total_quota - result["total_in_use"]
        if requested_count > total_available:
            result["feasible"] = False
            result["quota_ok"] = False
            result["issues"].append(
                f"Insufficient total cluster quota: "
                f"quota={total_quota}, in_use={result['total_in_use']}, "
                f"available={total_available}, requested={requested_count}"
            )

    logger.info(
        f"  Quota: {instance_type} {result['instance_type_in_use']}/{type_quota} used, "
        f"total {result['total_in_use']}/{total_quota} used -> "
        f"{'OK' if result['quota_ok'] else 'FAILED'}"
    )

    # =====================================================================
    # Layer 2: AZ Offering Check
    # =====================================================================
    logger.info(f"[2/3] Checking AZ offerings for {ec2_type}...")

    # Resolve target subnets from cluster config if provided
    target_subnets = []
    if cluster_name:
        try:
            desc = sm_client.describe_cluster(ClusterName=cluster_name)
            # Check instance group overrides first (more specific)
            for ig in desc.get("InstanceGroups", []):
                if ig.get("InstanceType") == instance_type:
                    override = ig.get("OverrideVpcConfig", {})
                    if override.get("Subnets"):
                        target_subnets = override["Subnets"]
                        break
            # Fall back to cluster-level VPC config
            if not target_subnets:
                target_subnets = desc.get("VpcConfig", {}).get("Subnets", [])
        except ClientError as e:
            result["issues"].append(f"Could not resolve subnets for cluster '{cluster_name}': {e}")

    az_check = _check_az_offerings(ec2_client, ec2_type, target_subnets or None, region)
    result["offered_azs"] = az_check["offered_azs"]
    result["target_subnet_azs"] = az_check["subnet_azs"]

    if az_check["issues"]:
        result["issues"].extend(az_check["issues"])
        result["feasible"] = False
        result["az_ok"] = False
    elif az_check["offered_azs"]:
        result["az_ok"] = True
    # else: no info, leave as None

    logger.info(
        f"  AZ offerings: {ec2_type} available in {az_check['offered_azs'] or '(unknown)'}"
        + (f", target subnets in {list(az_check['subnet_azs'].values())}" if az_check['subnet_azs'] else "")
        + f" -> {'OK' if result['az_ok'] else 'FAILED' if result['az_ok'] is False else 'SKIPPED'}"
    )

    # =====================================================================
    # Layer 3: EC2 Dry-Run Capacity Check
    # =====================================================================
    logger.info(f"[3/3] EC2 dry-run capacity check for {requested_count}x {ec2_type}...")

    # Pick the first target subnet for the dry-run (if available)
    dry_run_subnet = target_subnets[0] if target_subnets else None
    dry_run = _check_ec2_capacity_dry_run(ec2_client, ec2_type, requested_count, dry_run_subnet)

    result["ec2_capacity_ok"] = dry_run["capacity_likely"]
    if dry_run["capacity_likely"] is False:
        result["feasible"] = False
        result["issues"].append(dry_run["detail"])
    elif dry_run["capacity_likely"] is None and dry_run["detail"]:
        # Inconclusive - log but don't fail
        logger.info(f"  {dry_run['detail']}")

    logger.info(
        f"  EC2 dry-run: {dry_run['detail']} -> "
        f"{'OK' if dry_run['capacity_likely'] else 'FAILED' if dry_run['capacity_likely'] is False else 'INCONCLUSIVE'}"
    )

    # =====================================================================
    # Final Summary
    # =====================================================================
    if result["feasible"]:
        logger.info(
            f"Capacity check PASSED for {requested_count}x {instance_type}: "
            f"quota OK, AZ {'OK' if result['az_ok'] else 'N/A'}, "
            f"EC2 {'OK' if result['ec2_capacity_ok'] else 'N/A'}"
        )
    else:
        logger.warning(f"Capacity check FAILED for {requested_count}x {instance_type}:")
        for issue in result["issues"]:
            logger.warning(f"  - {issue}")

    return result


# ---------------------------------------------------------------------------
# Cluster Lifecycle: describe, scale, delete
# ---------------------------------------------------------------------------


def describe_cluster(cluster_name: str, region: Optional[str] = None) -> Dict[str, Any]:
    """Describe a HyperPod cluster and its instance groups.

    Args:
        cluster_name: SageMaker HyperPod cluster name.
        region: AWS region. Auto-detected if not provided.

    Returns:
        Dictionary with cluster status, instance groups, orchestrator, VPC config, etc.
    """
    region = region or boto3.Session().region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
    client = boto3.client("sagemaker", region_name=region)

    try:
        resp = client.describe_cluster(ClusterName=cluster_name)
        # Strip ResponseMetadata
        resp.pop("ResponseMetadata", None)

        # Build a concise summary alongside the raw response
        groups = []
        for ig in resp.get("InstanceGroups", []):
            groups.append({
                "name": ig.get("InstanceGroupName"),
                "instance_type": ig.get("InstanceType"),
                "current_count": ig.get("CurrentCount"),
                "target_count": ig.get("TargetCount"),
                "status": ig.get("Status"),
            })

        summary = {
            "cluster_name": resp.get("ClusterName"),
            "cluster_arn": resp.get("ClusterArn"),
            "status": resp.get("ClusterStatus"),
            "instance_groups": groups,
            "orchestrator": resp.get("Orchestrator"),
            "node_recovery": resp.get("NodeRecovery"),
            "creation_time": str(resp.get("CreationTime", "")),
        }

        logger.info(
            f"Cluster '{cluster_name}' status={summary['status']}, "
            f"{len(groups)} instance group(s)"
        )
        return {"summary": summary, "raw": resp}
    except ClientError as e:
        logger.error(f"Failed to describe cluster '{cluster_name}': {e}")
        raise


def list_clusters(region: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all HyperPod clusters in the account/region.

    Args:
        region: AWS region. Auto-detected if not provided.

    Returns:
        List of cluster summary dicts (name, arn, status, creation_time).
    """
    region = region or boto3.Session().region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
    client = boto3.client("sagemaker", region_name=region)

    try:
        clusters = []
        paginator = client.get_paginator("list_clusters")
        for page in paginator.paginate():
            for c in page.get("ClusterSummaries", []):
                clusters.append({
                    "name": c.get("ClusterName"),
                    "arn": c.get("ClusterArn"),
                    "status": c.get("ClusterStatus"),
                    "creation_time": str(c.get("CreationTime", "")),
                })
        logger.info(f"Found {len(clusters)} HyperPod cluster(s) in {region}")
        return clusters
    except ClientError as e:
        logger.error(f"Failed to list clusters: {e}")
        raise


def scale_instance_group(
    cluster_name: str,
    group_name: str,
    target_count: int,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Scale a HyperPod instance group to a target count.

    This calls the SageMaker UpdateCluster API, preserving the existing
    instance group configuration (instance type, lifecycle config,
    execution role, storage, VPC overrides, etc.) and only changing the
    instance count.

    Args:
        cluster_name: SageMaker HyperPod cluster name.
        group_name: Instance group name to scale.
        target_count: Desired number of instances (0 to scale down completely).
        region: AWS region. Auto-detected if not provided.

    Returns:
        Dictionary with cluster ARN and update status.
    """
    region = region or boto3.Session().region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
    client = boto3.client("sagemaker", region_name=region)

    # Fetch current cluster config so we can preserve all fields
    try:
        current = client.describe_cluster(ClusterName=cluster_name)
    except ClientError as e:
        logger.error(f"Failed to describe cluster '{cluster_name}': {e}")
        raise

    # Find the target instance group
    target_ig = None
    for ig in current.get("InstanceGroups", []):
        if ig.get("InstanceGroupName") == group_name:
            target_ig = ig
            break

    if target_ig is None:
        available = [ig.get("InstanceGroupName") for ig in current.get("InstanceGroups", [])]
        raise ValueError(
            f"Instance group '{group_name}' not found in cluster '{cluster_name}'. "
            f"Available groups: {available}"
        )

    old_count = target_ig.get("CurrentCount", target_ig.get("TargetCount", 0))
    old_target = target_ig.get("TargetCount", old_count)

    # No-op if already at target
    if old_target == target_count:
        logger.info(
            f"Instance group '{group_name}' is already at target count {target_count}, no update needed"
        )
        return {
            "cluster_arn": current.get("ClusterArn"),
            "group_name": group_name,
            "old_count": old_count,
            "target_count": target_count,
            "status": "NoChange",
        }

    logger.info(
        f"Scaling '{group_name}' in '{cluster_name}': {old_count} -> {target_count} "
        f"({target_ig.get('InstanceType')})"
    )

    # Build the UpdateCluster instance group spec.
    # We must include ALL existing fields to avoid validation errors.
    update_ig = {
        "InstanceGroupName": target_ig["InstanceGroupName"],
        "InstanceType": target_ig["InstanceType"],
        "InstanceCount": target_count,
        "LifeCycleConfig": target_ig["LifeCycleConfig"],
        "ExecutionRole": target_ig["ExecutionRole"],
        "ThreadsPerCore": target_ig.get("ThreadsPerCore", 2),
    }

    # Preserve optional fields if they exist
    if target_ig.get("InstanceStorageConfigs"):
        update_ig["InstanceStorageConfigs"] = target_ig["InstanceStorageConfigs"]
    if target_ig.get("OverrideVpcConfig"):
        update_ig["OverrideVpcConfig"] = target_ig["OverrideVpcConfig"]

    try:
        resp = client.update_cluster(
            ClusterName=cluster_name,
            InstanceGroups=[update_ig],
        )
        cluster_arn = resp.get("ClusterArn")
        logger.info(f"Scale request accepted. Cluster ARN: {cluster_arn}")
        return {
            "cluster_arn": cluster_arn,
            "group_name": group_name,
            "old_count": old_count,
            "target_count": target_count,
            "status": "UpdateInProgress",
        }
    except ClientError as e:
        logger.error(f"Failed to scale instance group: {e}")
        raise


def wait_for_cluster_status(
    cluster_name: str,
    target_status: str = "InService",
    timeout: int = 600,
    poll_interval: int = 30,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Wait for a HyperPod cluster to reach a target status.

    Args:
        cluster_name: SageMaker HyperPod cluster name.
        target_status: Status to wait for (default: InService).
        timeout: Max seconds to wait (default: 600).
        poll_interval: Seconds between polls (default: 30).
        region: AWS region. Auto-detected if not provided.

    Returns:
        Final cluster description.

    Raises:
        TimeoutError: If status not reached within timeout.
    """
    import time as _time

    region = region or boto3.Session().region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
    client = boto3.client("sagemaker", region_name=region)
    start = _time.time()

    while True:
        resp = client.describe_cluster(ClusterName=cluster_name)
        status = resp.get("ClusterStatus")
        elapsed = int(_time.time() - start)

        # Also check instance group counts
        groups_info = []
        for ig in resp.get("InstanceGroups", []):
            cur = ig.get("CurrentCount", 0)
            tgt = ig.get("TargetCount", 0)
            groups_info.append(f"{ig.get('InstanceGroupName')}: {cur}/{tgt}")

        logger.info(f"Cluster '{cluster_name}' status={status} [{', '.join(groups_info)}] ({elapsed}s)")

        if status == target_status:
            # Also check if all instance groups have reached their target
            all_scaled = all(
                ig.get("CurrentCount", 0) == ig.get("TargetCount", 0)
                for ig in resp.get("InstanceGroups", [])
            )
            if all_scaled:
                logger.info(f"Cluster reached '{target_status}' with all groups at target count")
                resp.pop("ResponseMetadata", None)
                return resp

        if _time.time() - start > timeout:
            raise TimeoutError(
                f"Cluster '{cluster_name}' did not reach '{target_status}' within {timeout}s. "
                f"Current status: {status}"
            )

        _time.sleep(poll_interval)


def delete_cluster(
    cluster_name: str,
    region: Optional[str] = None,
    confirm: bool = True,
) -> Dict[str, Any]:
    """Delete a HyperPod cluster.

    WARNING: This is destructive. All instances will be terminated and the
    cluster resource will be removed. Data on instance-local storage will be
    lost. Shared PVCs backed by EBS/EFS/FSx are NOT deleted.

    Args:
        cluster_name: SageMaker HyperPod cluster name.
        region: AWS region. Auto-detected if not provided.
        confirm: If True (default), log a warning before deletion. Set to
                 False to skip the warning (for automation).

    Returns:
        Dictionary with deletion status.
    """
    region = region or boto3.Session().region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
    client = boto3.client("sagemaker", region_name=region)

    # Describe first to show what we're deleting
    try:
        current = client.describe_cluster(ClusterName=cluster_name)
    except ClientError as e:
        logger.error(f"Cluster '{cluster_name}' not found: {e}")
        raise

    groups = current.get("InstanceGroups", [])
    total_instances = sum(ig.get("CurrentCount", 0) for ig in groups)

    if confirm:
        logger.warning(
            f"DELETING cluster '{cluster_name}' with {total_instances} running instance(s) "
            f"across {len(groups)} group(s). This cannot be undone."
        )

    try:
        client.delete_cluster(ClusterName=cluster_name)
        logger.info(f"Delete request accepted for cluster '{cluster_name}'")
        return {
            "cluster_name": cluster_name,
            "status": "Deleting",
            "instances_terminated": total_instances,
        }
    except ClientError as e:
        logger.error(f"Failed to delete cluster '{cluster_name}': {e}")
        raise


def get_observability_pods(namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get observability-related pods (DCGM exporter, Prometheus, etc.).
    
    Args:
        namespace: Optional namespace to filter by.
    
    Returns:
        List of observability pod dictionaries.
    """
    observability_keywords = [
        "dcgm",
        "prometheus",
        "grafana",
        "exporter",
        "otel",
        "observability",
        "metrics"
    ]
    
    try:
        cmd = ["get", "pods", "-A", "-o", "json"]
        if namespace:
            cmd = ["get", "pods", "-n", namespace, "-o", "json"]
        
        result = _run_kubectl(cmd)
        data = json.loads(result.stdout)
        
        observability_pods = []
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            spec = item.get("spec", {})
            status = item.get("status", {})
            
            pod_name = metadata.get("name", "").lower()
            containers = spec.get("containers", [])
            
            # Check if pod name contains observability keywords
            is_observability = any(
                keyword in pod_name 
                for keyword in observability_keywords
            )
            
            # Check container images
            if not is_observability:
                for container in containers:
                    image = container.get("image", "").lower()
                    if any(keyword in image for keyword in observability_keywords):
                        is_observability = True
                        break
            
            if is_observability:
                pod_info = {
                    "name": metadata.get("name"),
                    "namespace": metadata.get("namespace"),
                    "status": status.get("phase"),
                    "ready": f"{sum(1 for c in status.get('containerStatuses', []) if c.get('ready'))}/{len(containers)}",
                    "restarts": sum(
                        c.get("restartCount", 0) 
                        for c in status.get("containerStatuses", [])
                    ),
                    "containers": [c.get("name") for c in containers],
                    "node_name": spec.get("nodeName"),
                    "start_time": status.get("startTime"),
                }
                observability_pods.append(pod_info)
        
        logger.info(f"Found {len(observability_pods)} observability pods")
        return observability_pods
    except Exception as e:
        logger.error(f"Failed to get observability pods: {e}")
        return []


# ============================================================================
# Lifecycle Scripts
# ============================================================================

# Minimal on_create.sh that serves as the entry point for HyperPod lifecycle.
# It calls on_create_main.sh (if present) and logs output to /var/log/provision/.
_ON_CREATE_SH = r"""#!/bin/bash
set -e
LOG_DIR="/var/log/provision"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/provisioning.log"
echo "$(date) - on_create.sh starting" | tee "$LOG_FILE"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
MAIN_SCRIPT="$SCRIPT_DIR/on_create_main.sh"

if [ -f "$MAIN_SCRIPT" ]; then
    echo "$(date) - Running on_create_main.sh" | tee -a "$LOG_FILE"
    bash "$MAIN_SCRIPT" >> "$LOG_FILE" 2>&1 || {
        echo "$(date) - on_create_main.sh FAILED (exit code $?)" | tee -a "$LOG_FILE"
        sleep 60  # allow log upload before exit
        exit 1
    }
    echo "$(date) - on_create_main.sh completed" | tee -a "$LOG_FILE"
else
    echo "$(date) - on_create_main.sh not found, skipping" | tee -a "$LOG_FILE"
fi

echo "$(date) - on_create.sh finished" | tee -a "$LOG_FILE"
"""

# on_create_main.sh: configures containerd to use larger disk and loads Lustre modules.
# Based on 1.architectures/7.sagemaker-hyperpod-eks/LifecycleScripts/base-config/on_create_main.sh
_ON_CREATE_MAIN_SH = r"""#!/bin/bash
set -e

echo "=== HyperPod on_create_main.sh ==="

# --- Move containerd/kubelet to larger disk ---
TARGET_DISK="/opt/sagemaker"
[ -d "/opt/dlami/nvme" ] && TARGET_DISK="/opt/dlami/nvme"

CONTAINERD_DIR="$TARGET_DISK/containerd"
KUBELET_DIR="$TARGET_DISK/kubelet"
mkdir -p "$CONTAINERD_DIR" "$KUBELET_DIR"

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
fi

if [[ "${ID}" == "amzn" && "${VERSION_ID}" == "2023" ]]; then
    echo "Configuring containerd for AL2023..."
    mkdir -p /etc/containerd
    cat > /etc/containerd/config.toml <<TOML
version = 2
root = "$CONTAINERD_DIR"

[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "nvidia"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
TOML
    mkdir -p /etc/systemd/system/containerd.service.d
    cat > /etc/systemd/system/containerd.service.d/override.conf <<OVERRIDE
[Service]
ExecStart=
ExecStart=/usr/bin/containerd --config /etc/containerd/config.toml
OVERRIDE
    systemctl daemon-reload
elif [[ "${ID}" == "amzn" && "${VERSION_ID}" == "2" ]]; then
    echo "Configuring containerd for AL2..."
    TOML_FILE="/etc/eks/containerd/containerd-config.toml"
    if [ -f "$TOML_FILE" ]; then
        sed -i "s|root = .*|root = \"$CONTAINERD_DIR\"|" "$TOML_FILE"
    fi
fi

# Move kubelet data
if [ -d /var/lib/kubelet ] && [ ! -L /var/lib/kubelet ]; then
    cp -a /var/lib/kubelet/* "$KUBELET_DIR/" 2>/dev/null || true
    rm -rf /var/lib/kubelet
fi
ln -sfn "$KUBELET_DIR" /var/lib/kubelet

echo "Containerd root: $CONTAINERD_DIR, Kubelet: $KUBELET_DIR"

# --- Load Lustre kernel modules ---
modprobe lnet 2>/dev/null || echo "lnet module not available"
modprobe lustre 2>/dev/null || echo "lustre module not available"
lctl network up 2>/dev/null || echo "lctl network up skipped"

echo "=== on_create_main.sh complete ==="
"""


def setup_lifecycle_scripts(
    s3_bucket: str,
    s3_prefix: str = "",
    custom_on_create_main: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, str]:
    """Upload lifecycle scripts to S3 for HyperPod instance provisioning.

    Uploads on_create.sh (entry point) and on_create_main.sh (main logic)
    to the specified S3 bucket. The LifeCycleConfig in CreateCluster should
    reference SourceS3Uri=s3://{bucket}/{prefix} and OnCreate=on_create.sh.

    The default on_create_main.sh:
    - Moves containerd/kubelet to a larger disk (supports AL2 and AL2023)
    - Loads Lustre kernel modules

    Args:
        s3_bucket: S3 bucket name.
        s3_prefix: Optional prefix (folder) within the bucket.
        custom_on_create_main: Optional custom script content to replace the
            default on_create_main.sh.
        region: AWS region.

    Returns:
        Dict with s3_uri (the SourceS3Uri for LifeCycleConfig) and uploaded keys.
    """
    region = region or _get_region()
    s3 = boto3.client("s3", region_name=region)

    prefix = s3_prefix.strip("/")
    if prefix:
        prefix += "/"

    on_create_key = f"{prefix}on_create.sh"
    on_create_main_key = f"{prefix}on_create_main.sh"

    main_content = custom_on_create_main if custom_on_create_main else _ON_CREATE_MAIN_SH

    s3.put_object(Bucket=s3_bucket, Key=on_create_key, Body=_ON_CREATE_SH.encode())
    logger.info(f"Uploaded s3://{s3_bucket}/{on_create_key}")

    s3.put_object(Bucket=s3_bucket, Key=on_create_main_key, Body=main_content.encode())
    logger.info(f"Uploaded s3://{s3_bucket}/{on_create_main_key}")

    s3_uri = f"s3://{s3_bucket}" if not prefix else f"s3://{s3_bucket}/{prefix.rstrip('/')}"

    return {
        "s3_uri": s3_uri,
        "on_create_key": on_create_key,
        "on_create_main_key": on_create_main_key,
        "on_create_script": "on_create.sh",
    }


# ============================================================================
# HyperPod Cluster Creation
# ============================================================================

def create_hyperpod_cluster(
    cluster_name: str,
    eks_cluster_arn: str,
    instance_groups: List[Dict[str, Any]],
    security_group_ids: List[str],
    subnet_ids: List[str],
    node_recovery: str = "Automatic",
    pre_check_capacity: bool = True,
    tags: Optional[Dict[str, str]] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a SageMaker HyperPod cluster with EKS orchestrator.

    Performs an optional 3-layer capacity pre-check before creating the
    cluster. Configures automatic node recovery and deep health checks
    by default.

    Args:
        cluster_name: HyperPod cluster name.
        eks_cluster_arn: ARN of the EKS cluster to use as orchestrator.
        instance_groups: List of instance group configs. Each must contain:
            - InstanceGroupName (str)
            - InstanceType (str, e.g., "ml.g5.12xlarge")
            - InstanceCount (int)
            - ExecutionRole (str, IAM role ARN)
            - LifeCycleConfig (dict with OnCreate and SourceS3Uri)
            Optional:
            - ThreadsPerCore (int, default 1)
            - InstanceStorageConfigs (list of EbsVolumeConfig dicts)
            - OnStartDeepHealthChecks (list, e.g., ["InstanceStress", "InstanceConnectivity"])
            - OverrideVpcConfig (dict with SecurityGroupIds and Subnets)
        security_group_ids: Security group IDs for the cluster VPC config.
        subnet_ids: Subnet IDs for the cluster VPC config.
        node_recovery: "Automatic" or "None" (default: "Automatic").
        pre_check_capacity: Run 3-layer capacity check before creating
            (default: True). Blocks creation if capacity is insufficient.
        tags: Optional tags for the cluster.
        region: AWS region.

    Returns:
        Dict with cluster_name, cluster_arn, status, and instance_groups.
    """
    region = region or _get_region()
    sm = boto3.client("sagemaker", region_name=region)

    logger.info("=" * 70)
    logger.info(f"Creating HyperPod cluster: {cluster_name}")
    logger.info(f"  EKS orchestrator: {eks_cluster_arn}")
    logger.info(f"  Instance groups: {len(instance_groups)}")
    logger.info(f"  Node recovery: {node_recovery}")
    logger.info("=" * 70)

    # --- Capacity pre-check ---
    if pre_check_capacity:
        logger.info("Running capacity pre-checks...")
        for ig in instance_groups:
            itype = ig["InstanceType"]
            icount = ig["InstanceCount"]
            if icount == 0:
                continue
            cap = check_capacity(itype, icount, region=region)
            if not cap["feasible"]:
                issues_str = "; ".join(cap["issues"])
                raise ValueError(
                    f"Capacity pre-check FAILED for {ig['InstanceGroupName']} "
                    f"({icount}x {itype}): {issues_str}"
                )
            logger.info(f"  {ig['InstanceGroupName']}: {icount}x {itype} - capacity OK")

    # --- Build instance group specs ---
    ig_specs = []
    for ig in instance_groups:
        spec = {
            "InstanceGroupName": ig["InstanceGroupName"],
            "InstanceType": ig["InstanceType"],
            "InstanceCount": ig["InstanceCount"],
            "ExecutionRole": ig["ExecutionRole"],
            "LifeCycleConfig": ig["LifeCycleConfig"],
            "ThreadsPerCore": ig.get("ThreadsPerCore", 1),
        }

        if ig.get("InstanceStorageConfigs"):
            spec["InstanceStorageConfigs"] = ig["InstanceStorageConfigs"]

        # Deep health checks - default to both if not specified and count > 0
        if ig["InstanceCount"] > 0:
            dhc = ig.get("OnStartDeepHealthChecks", ["InstanceStress", "InstanceConnectivity"])
            spec["OnStartDeepHealthChecks"] = dhc

        if ig.get("OverrideVpcConfig"):
            spec["OverrideVpcConfig"] = ig["OverrideVpcConfig"]

        ig_specs.append(spec)

    # --- CreateCluster call ---
    create_kwargs = {
        "ClusterName": cluster_name,
        "Orchestrator": {
            "Eks": {
                "ClusterArn": eks_cluster_arn,
            }
        },
        "VpcConfig": {
            "SecurityGroupIds": security_group_ids,
            "Subnets": subnet_ids,
        },
        "InstanceGroups": ig_specs,
        "NodeRecovery": node_recovery,
    }

    # Always include SageMaker=true tag - required by the HPTO managed policy
    # (AmazonSageMakerHyperPodTrainingOperatorAccess) which has an IAM condition:
    #   StringEquals: { aws:ResourceTag/SageMaker: "true" }
    all_tags = {"SageMaker": "true"}
    if tags:
        all_tags.update(tags)
    create_kwargs["Tags"] = [{"Key": k, "Value": v} for k, v in all_tags.items()]

    try:
        resp = sm.create_cluster(**create_kwargs)
        cluster_arn = resp["ClusterArn"]
        logger.info(f"HyperPod cluster creation initiated: {cluster_arn}")
    except ClientError as e:
        if "ResourceInUse" in str(e) or "already exists" in str(e).lower():
            logger.info(f"Cluster '{cluster_name}' already exists")
            desc = sm.describe_cluster(ClusterName=cluster_name)
            return {
                "cluster_name": cluster_name,
                "cluster_arn": desc["ClusterArn"],
                "status": desc.get("ClusterStatus"),
                "already_existed": True,
            }
        raise

    return {
        "cluster_name": cluster_name,
        "cluster_arn": cluster_arn,
        "status": "Creating",
        "already_existed": False,
        "instance_groups": [
            {
                "name": ig["InstanceGroupName"],
                "type": ig["InstanceType"],
                "count": ig["InstanceCount"],
            }
            for ig in instance_groups
        ],
    }


# ============================================================================
# Observability Addon (AMP + DCGM + EKS Addon)
# ============================================================================

def _create_observability_role(
    eks_cluster_arn: str,
    name_prefix: str = "sagemaker-hyperpod",
    region: Optional[str] = None,
) -> str:
    """Create IAM role for the HyperPod observability addon (Pod Identity).

    Trust policy: pods.eks.amazonaws.com with SourceAccount + SourceArn conditions.
    Inline policy: APS RemoteWrite + CloudWatch Logs.

    Returns:
        Role ARN.
    """
    region = region or _get_region()
    account_id = _get_account_id()
    iam = boto3.client("iam", region_name=region)
    role_name = f"{name_prefix}-observability-addon-role"

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowEksAuthToAssumeRoleForPodIdentity",
            "Effect": "Allow",
            "Principal": {"Service": "pods.eks.amazonaws.com"},
            "Action": ["sts:AssumeRole", "sts:TagSession"],
            "Condition": {
                "StringEquals": {
                    "aws:SourceAccount": account_id,
                    "aws:SourceArn": eks_cluster_arn,
                }
            },
        }],
    }

    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PrometheusAccess",
                "Effect": "Allow",
                "Action": ["aps:RemoteWrite"],
                "Resource": f"arn:aws:aps:{region}:{account_id}:workspace/*",
            },
            {
                "Sid": "CloudwatchLogsAccess",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup", "logs:CreateLogStream",
                    "logs:DescribeLogGroups", "logs:DescribeLogStreams",
                    "logs:PutLogEvents", "logs:GetLogEvents",
                    "logs:FilterLogEvents", "logs:GetLogRecord",
                    "logs:StartQuery", "logs:StopQuery", "logs:GetQueryResults",
                ],
                "Resource": [
                    f"arn:aws:logs:{region}:{account_id}:log-group:/aws/sagemaker/Clusters/*",
                    f"arn:aws:logs:{region}:{account_id}:log-group:/aws/sagemaker/Clusters/*:log-stream:*",
                ],
            },
        ],
    }

    try:
        resp = iam.create_role(
            Path="/service-role/",
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="HyperPod observability addon role for Pod Identity",
            Tags=[{"Key": "ManagedBy", "Value": "opencode-hyperpod-manager"}],
        )
        role_arn = resp["Role"]["Arn"]

        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{name_prefix}-observability-policy",
            PolicyDocument=json.dumps(inline_policy),
        )

        logger.info(f"Observability role created: {role_arn}")
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            resp = iam.get_role(RoleName=role_name)
            logger.info(f"Observability role already exists: {resp['Role']['Arn']}")
            return resp["Role"]["Arn"]
        raise


def install_observability_addon(
    eks_cluster_name: str,
    eks_cluster_arn: str,
    name_prefix: str = "sagemaker-hyperpod",
    training_metrics_level: str = "BASIC",
    accelerated_compute_metrics_level: str = "BASIC",
    node_metrics_level: str = "BASIC",
    custom_metrics_level: str = "BASIC",
    scrape_interval: int = 30,
    create_amp_workspace: bool = True,
    amp_workspace_id: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Install the HyperPod observability addon with Amazon Managed Prometheus.

    This function:
    1. Creates an AMP workspace (unless amp_workspace_id is provided)
    2. Creates an IAM role for Pod Identity with APS RemoteWrite + Logs permissions
    3. Installs the amazon-sagemaker-hyperpod-observability EKS addon

    Security: NO public Grafana endpoint is created. Metrics go to AMP only.
    Use `kubectl port-forward` or a private Grafana if you need dashboards.

    Args:
        eks_cluster_name: EKS cluster name.
        eks_cluster_arn: EKS cluster ARN.
        name_prefix: Prefix for resource names.
        training_metrics_level: BASIC or DISABLED (default: BASIC).
        accelerated_compute_metrics_level: BASIC or DISABLED (default: BASIC).
        node_metrics_level: BASIC or DISABLED (default: BASIC).
        custom_metrics_level: BASIC or DISABLED (default: BASIC).
        scrape_interval: Metrics scrape interval in seconds (default: 30).
        create_amp_workspace: Create a new AMP workspace (default: True).
        amp_workspace_id: Existing AMP workspace ID (skips creation).
        region: AWS region.

    Returns:
        Dict with amp_workspace_id, amp_endpoint, role_arn, addon_name.
    """
    region = region or _get_region()
    result = {}

    # --- Step 1: AMP Workspace ---
    if amp_workspace_id:
        amp = boto3.client("amp", region_name=region)
        ws = amp.describe_workspace(workspaceId=amp_workspace_id)["workspace"]
        amp_endpoint = ws["prometheusEndpoint"]
        amp_arn = ws["arn"]
        result["amp_workspace_id"] = amp_workspace_id
        result["amp_endpoint"] = amp_endpoint
        result["amp_arn"] = amp_arn
        logger.info(f"Using existing AMP workspace: {amp_workspace_id}")
    elif create_amp_workspace:
        amp = boto3.client("amp", region_name=region)
        alias = f"{name_prefix}-prometheus"
        try:
            ws_resp = amp.create_workspace(
                alias=alias,
                tags={"SageMaker": "true", "ManagedBy": "opencode-hyperpod-manager"},
            )
            amp_workspace_id = ws_resp["workspaceId"]
            amp_arn = ws_resp["arn"]

            # Wait for workspace to be ACTIVE
            logger.info(f"AMP workspace created: {amp_workspace_id}, waiting for ACTIVE...")
            for _ in range(60):
                ws = amp.describe_workspace(workspaceId=amp_workspace_id)["workspace"]
                if ws["status"]["statusCode"] == "ACTIVE":
                    break
                time.sleep(5)
            else:
                raise TimeoutError(f"AMP workspace {amp_workspace_id} did not become ACTIVE within 5 minutes")

            amp_endpoint = ws["prometheusEndpoint"]
            result["amp_workspace_id"] = amp_workspace_id
            result["amp_endpoint"] = amp_endpoint
            result["amp_arn"] = amp_arn
            logger.info(f"AMP workspace ACTIVE: {amp_endpoint}")
        except ClientError as e:
            if "ConflictException" in str(e):
                # Workspace with this alias may already exist; list and find it
                logger.info(f"AMP workspace with alias '{alias}' may already exist, searching...")
                paginator = amp.get_paginator("list_workspaces")
                for page in paginator.paginate(alias=alias):
                    for ws in page.get("workspaces", []):
                        if ws["alias"] == alias:
                            amp_workspace_id = ws["workspaceId"]
                            ws_full = amp.describe_workspace(workspaceId=amp_workspace_id)["workspace"]
                            amp_endpoint = ws_full["prometheusEndpoint"]
                            amp_arn = ws_full["arn"]
                            result["amp_workspace_id"] = amp_workspace_id
                            result["amp_endpoint"] = amp_endpoint
                            result["amp_arn"] = amp_arn
                            logger.info(f"Found existing AMP workspace: {amp_workspace_id}")
                            break
                if "amp_workspace_id" not in result:
                    raise
            else:
                raise
    else:
        raise ValueError("Either amp_workspace_id must be provided or create_amp_workspace must be True")

    # --- Step 2: IAM Role ---
    logger.info("Creating observability IAM role...")
    role_arn = _create_observability_role(eks_cluster_arn, name_prefix, region)
    result["role_arn"] = role_arn

    # --- Step 3: Install EKS Addon ---
    eks = boto3.client("eks", region_name=region)

    # Build configurationValues - strip trailing / from endpoint
    amp_ep = result["amp_endpoint"].rstrip("/")
    config_values = {
        "ampWorkspace": {
            "prometheusEndpoint": amp_ep,
            "arn": result["amp_arn"],
        },
        # NO amgWorkspace - we do NOT create public Grafana endpoints
        "metricsProvider": {
            "trainingMetrics": {"level": training_metrics_level, "scrapeInterval": scrape_interval},
            "inferenceMetrics": {"level": "BASIC", "scrapeInterval": scrape_interval},
            "taskGovernanceMetrics": {"level": "DISABLED", "scrapeInterval": scrape_interval},
            "scalingMetrics": {"level": "DISABLED", "scrapeInterval": scrape_interval},
            "customMetrics": {"level": custom_metrics_level, "scrapeInterval": scrape_interval},
            "nodeMetrics": {"level": node_metrics_level, "scrapeInterval": scrape_interval},
            "acceleratedComputeMetrics": {"level": accelerated_compute_metrics_level, "scrapeInterval": scrape_interval},
            "networkMetrics": {"level": "DISABLED", "scrapeInterval": scrape_interval},
        },
        "logging": {"enabled": False},
    }

    addon_name = "amazon-sagemaker-hyperpod-observability"
    try:
        eks.create_addon(
            clusterName=eks_cluster_name,
            addonName=addon_name,
            resolveConflicts="OVERWRITE",
            configurationValues=json.dumps(config_values),
            podIdentityAssociations=[{
                "roleArn": role_arn,
                "serviceAccount": "hyperpod-observability-operator-otel-collector",
            }],
            tags={"SageMaker": "true", "ManagedBy": "opencode-hyperpod-manager"},
        )
        logger.info(f"EKS addon '{addon_name}' installation initiated")
    except ClientError as e:
        if "already exists" in str(e).lower():
            logger.info(f"EKS addon '{addon_name}' already installed")
        else:
            raise

    result["addon_name"] = addon_name
    result["service_account"] = "hyperpod-observability-operator-otel-collector"

    logger.info("Observability addon installation complete")
    logger.info(f"  AMP workspace: {result.get('amp_workspace_id')}")
    logger.info(f"  Role: {role_arn}")
    logger.info(f"  Addon: {addon_name}")
    logger.info("  NOTE: No public Grafana endpoint created (use kubectl port-forward for dashboards)")

    return result


# ============================================================================
# Training Operator Addon (HPTO)
# ============================================================================

def _create_training_operator_role(
    name_prefix: str = "sagemaker-hyperpod",
    region: Optional[str] = None,
) -> str:
    """Create IAM role for the HyperPod training operator addon (Pod Identity).

    Trust policy: pods.eks.amazonaws.com (NO conditions - differs from observability).
    Managed policy: AmazonSageMakerHyperPodTrainingOperatorAccess.

    Returns:
        Role ARN.
    """
    region = region or _get_region()
    iam = boto3.client("iam", region_name=region)
    role_name = f"{name_prefix}-hpto-role"

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowEksAuthToAssumeRoleForPodIdentity",
            "Effect": "Allow",
            "Principal": {"Service": "pods.eks.amazonaws.com"},
            "Action": ["sts:AssumeRole", "sts:TagSession"],
        }],
    }

    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="HyperPod training operator role for Pod Identity",
            Tags=[{"Key": "ManagedBy", "Value": "opencode-hyperpod-manager"}],
        )
        role_arn = resp["Role"]["Arn"]

        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerHyperPodTrainingOperatorAccess",
        )

        logger.info(f"Training operator role created: {role_arn}")
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            resp = iam.get_role(RoleName=role_name)
            logger.info(f"Training operator role already exists: {resp['Role']['Arn']}")
            return resp["Role"]["Arn"]
        raise


_CERT_MANAGER_VERSION = "v1.17.1"
_CERT_MANAGER_MANIFEST_URL = (
    f"https://github.com/cert-manager/cert-manager/releases/download/"
    f"{_CERT_MANAGER_VERSION}/cert-manager.yaml"
)


def install_cert_manager(
    eks_cluster_name: Optional[str] = None,
    version: str = _CERT_MANAGER_VERSION,
    region: Optional[str] = None,
    wait_timeout: int = 120,
) -> Dict[str, Any]:
    """Install cert-manager on the EKS cluster.

    cert-manager is a REQUIRED prerequisite for the HyperPod Training
    Operator (HPTO) EKS addon. Without it, the addon will fail with:
    "cert-manager is not installed on this cluster."

    This function:
    1. Checks if cert-manager is already installed
    2. Applies the cert-manager manifest from GitHub releases
    3. Waits for all cert-manager pods to be ready

    Args:
        eks_cluster_name: EKS cluster name. If provided, updates kubeconfig.
        version: cert-manager version (default: v1.17.1).
        region: AWS region.
        wait_timeout: Seconds to wait for pods to be ready (default: 120).

    Returns:
        Dict with version, status, and namespace.
    """
    region = region or _get_region()
    result = {"version": version, "namespace": "cert-manager"}

    # Update kubeconfig if cluster name provided
    if eks_cluster_name:
        try:
            subprocess.run(
                ["aws", "eks", "update-kubeconfig", "--name", eks_cluster_name, "--region", region],
                capture_output=True, text=True, check=True, timeout=30,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update kubeconfig: {e.stderr}")
            raise

    # Check if already installed
    try:
        check = subprocess.run(
            ["kubectl", "get", "deployment", "cert-manager", "-n", "cert-manager",
             "-o", "jsonpath={.status.readyReplicas}"],
            capture_output=True, text=True, timeout=15,
        )
        if check.returncode == 0 and check.stdout.strip().isdigit() and int(check.stdout.strip()) > 0:
            logger.info(f"cert-manager already installed and running ({check.stdout.strip()} replicas ready)")
            result["status"] = "already_installed"
            return result
    except Exception:
        pass  # Not installed, continue

    # Install cert-manager
    manifest_url = (
        f"https://github.com/cert-manager/cert-manager/releases/download/"
        f"{version}/cert-manager.yaml"
    )
    logger.info(f"Installing cert-manager {version}...")
    try:
        install = subprocess.run(
            ["kubectl", "apply", "-f", manifest_url],
            capture_output=True, text=True, check=True, timeout=120,
        )
        logger.info(f"cert-manager manifests applied")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install cert-manager: {e.stderr}")
        raise

    # Wait for pods to be ready
    logger.info(f"Waiting up to {wait_timeout}s for cert-manager pods...")
    try:
        subprocess.run(
            ["kubectl", "wait", "--for=condition=ready", "pod",
             "-l", "app.kubernetes.io/instance=cert-manager",
             "-n", "cert-manager",
             f"--timeout={wait_timeout}s"],
            capture_output=True, text=True, check=True, timeout=wait_timeout + 30,
        )
        logger.info("cert-manager pods are ready")
        result["status"] = "installed"
    except subprocess.CalledProcessError as e:
        logger.warning(f"cert-manager pods not ready within timeout: {e.stderr}")
        result["status"] = "installed_not_ready"

    return result


def install_training_operator(
    eks_cluster_name: str,
    name_prefix: str = "sagemaker-hyperpod",
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Install the HyperPod Training Operator (HPTO) EKS addon.

    This function:
    1. Installs cert-manager (required prerequisite)
    2. Creates an IAM role for Pod Identity (trust: pods.eks.amazonaws.com, no conditions)
    3. Attaches AmazonSageMakerHyperPodTrainingOperatorAccess managed policy
    4. Installs the amazon-sagemaker-hyperpod-training-operator EKS addon

    NOTE: The HPTO managed policy requires the HyperPod cluster to have
    the tag SageMaker=true. The create_hyperpod_cluster() function adds
    this tag automatically.

    Args:
        eks_cluster_name: EKS cluster name.
        name_prefix: Prefix for resource names.
        region: AWS region.

    Returns:
        Dict with role_arn, addon_name, service_account, cert_manager_status.
    """
    region = region or _get_region()
    result = {}

    # --- Step 1: cert-manager (REQUIRED prerequisite) ---
    logger.info("Ensuring cert-manager is installed (required for HPTO)...")
    cm_result = install_cert_manager(eks_cluster_name=eks_cluster_name, region=region)
    result["cert_manager_status"] = cm_result.get("status")
    logger.info(f"  cert-manager: {cm_result.get('status')}")

    # --- Step 2: IAM Role ---
    logger.info("Creating training operator IAM role...")
    role_arn = _create_training_operator_role(name_prefix, region)
    result["role_arn"] = role_arn

    # --- Step 3: Install EKS Addon ---
    eks = boto3.client("eks", region_name=region)
    addon_name = "amazon-sagemaker-hyperpod-training-operator"

    try:
        eks.create_addon(
            clusterName=eks_cluster_name,
            addonName=addon_name,
            resolveConflicts="OVERWRITE",
            podIdentityAssociations=[{
                "roleArn": role_arn,
                "serviceAccount": "hp-training-operator-controller-manager",
            }],
        )
        logger.info(f"EKS addon '{addon_name}' installation initiated")
    except ClientError as e:
        if "already exists" in str(e).lower():
            logger.info(f"EKS addon '{addon_name}' already installed")
        else:
            raise

    result["addon_name"] = addon_name
    result["service_account"] = "hp-training-operator-controller-manager"

    logger.info("Training operator addon installation complete")
    logger.info(f"  cert-manager: {result.get('cert_manager_status')}")
    logger.info(f"  Role: {role_arn}")
    logger.info(f"  Addon: {addon_name}")

    return result


# ============================================================================
# HyperPod Dependencies (Helm Chart)
# ============================================================================

_HYPERPOD_HELM_REPO = "https://github.com/aws/sagemaker-hyperpod-cli.git"
_HYPERPOD_HELM_CHART_PATH = "helm_chart/HyperPodHelmChart"
_HYPERPOD_HELM_RELEASE = "hyperpod-dependencies"
_HYPERPOD_HELM_NAMESPACE = "kube-system"


def install_hyperpod_dependencies(
    eks_cluster_name: Optional[str] = None,
    region: Optional[str] = None,
    helm_release: str = _HYPERPOD_HELM_RELEASE,
    namespace: str = _HYPERPOD_HELM_NAMESPACE,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Install the HyperPod dependencies Helm chart on an EKS cluster.

    This is a MANDATORY prerequisite before creating a HyperPod cluster.
    The Helm chart installs:
    - NVIDIA device plugin (GPU support)
    - AWS EFA device plugin (low-latency networking)
    - Health monitoring agent
    - Deep health check components
    - Job auto-restart controller
    - Kubeflow training operator
    - MPI operator
    - Kueue (job scheduling)
    - Neuron device plugin (Trainium/Inferentia)

    The chart is cloned from the sagemaker-hyperpod-cli GitHub repo and
    installed into the kube-system namespace.

    Args:
        eks_cluster_name: EKS cluster name. If provided, updates kubeconfig first.
        region: AWS region.
        helm_release: Helm release name (default: hyperpod-dependencies).
        namespace: Target namespace (default: kube-system).
        timeout: Helm install timeout in seconds (default: 600).

    Returns:
        Dict with release_name, namespace, status, and components installed.
    """
    import tempfile
    import shutil

    region = region or _get_region()
    result = {"release_name": helm_release, "namespace": namespace}

    # Update kubeconfig if cluster name provided
    if eks_cluster_name:
        logger.info(f"Updating kubeconfig for {eks_cluster_name}...")
        try:
            subprocess.run(
                ["aws", "eks", "update-kubeconfig", "--name", eks_cluster_name, "--region", region],
                capture_output=True, text=True, check=True, timeout=30,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update kubeconfig: {e.stderr}")
            raise

    # Check if already installed
    try:
        check = subprocess.run(
            ["helm", "status", helm_release, "--namespace", namespace],
            capture_output=True, text=True, timeout=15,
        )
        if check.returncode == 0 and "deployed" in check.stdout.lower():
            logger.info(f"Helm release '{helm_release}' already installed in {namespace}")
            result["status"] = "already_installed"
            return result
    except Exception:
        pass  # Not installed, continue

    # Clone the repo to a temp directory
    tmp_dir = tempfile.mkdtemp(prefix="hyperpod-helm-")
    try:
        logger.info(f"Cloning sagemaker-hyperpod-cli repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1", _HYPERPOD_HELM_REPO, tmp_dir],
            capture_output=True, text=True, check=True, timeout=120,
        )

        chart_path = os.path.join(tmp_dir, _HYPERPOD_HELM_CHART_PATH)
        if not os.path.isdir(chart_path):
            raise FileNotFoundError(f"Helm chart not found at {chart_path}")

        # Add required Helm repositories
        logger.info("Adding required Helm repositories...")
        for repo_name, repo_url in [
            ("nvidia", "https://nvidia.github.io/k8s-device-plugin"),
            ("eks", "https://aws.github.io/eks-charts/"),
        ]:
            subprocess.run(
                ["helm", "repo", "add", repo_name, repo_url],
                capture_output=True, text=True, timeout=30,
            )
        subprocess.run(
            ["helm", "repo", "update"],
            capture_output=True, text=True, check=True, timeout=60,
        )

        # Update chart dependencies
        logger.info("Updating Helm chart dependencies...")
        dep_result = subprocess.run(
            ["helm", "dependencies", "update", chart_path],
            capture_output=True, text=True, timeout=120,
        )
        if dep_result.returncode != 0:
            logger.warning(f"helm dependencies update warnings: {dep_result.stderr}")

        # Install the chart
        # NOTE: We do NOT use --wait because the cluster may not have worker
        # nodes yet (they come from HyperPod CreateCluster). DaemonSets and
        # some Deployments will stay Pending until nodes join. This is expected.
        logger.info(f"Installing Helm chart '{helm_release}' into {namespace}...")
        install_result = subprocess.run(
            [
                "helm", "install", helm_release, chart_path,
                "--namespace", namespace,
                "--timeout", f"{timeout}s",
            ],
            capture_output=True, text=True, timeout=timeout + 60,
        )

        if install_result.returncode != 0:
            logger.error(f"Helm install failed: {install_result.stderr}")
            raise RuntimeError(f"Helm install failed: {install_result.stderr}")

        logger.info(f"Helm chart installed successfully")
        result["status"] = "installed"
        result["helm_output"] = install_result.stdout[-500:] if install_result.stdout else ""

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Verify pods are coming up
    logger.info("Verifying HyperPod dependency pods...")
    time.sleep(10)  # Give pods a moment to start
    try:
        pods_result = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace,
             "-l", "app.kubernetes.io/name=nvidia-device-plugin",
             "-o", "json"],
            capture_output=True, text=True, timeout=15,
        )
        if pods_result.returncode == 0:
            pods = json.loads(pods_result.stdout).get("items", [])
            result["nvidia_plugin_pods"] = len(pods)

        # Check aws-hyperpod namespace
        hp_pods = subprocess.run(
            ["kubectl", "get", "pods", "-n", "aws-hyperpod", "-o", "json"],
            capture_output=True, text=True, timeout=15,
        )
        if hp_pods.returncode == 0:
            hp_items = json.loads(hp_pods.stdout).get("items", [])
            result["hyperpod_pods"] = len(hp_items)
    except Exception as e:
        logger.warning(f"Could not verify pods: {e}")

    logger.info("HyperPod dependencies installation complete")
    logger.info(f"  Release: {helm_release}")
    logger.info(f"  Namespace: {namespace}")
    logger.info(f"  Status: {result.get('status')}")

    return result
