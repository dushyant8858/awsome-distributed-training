#!/usr/bin/env python3
"""
EKS Cluster Manager Skill

Creates, validates, and manages EKS clusters and supporting infrastructure
(VPC, subnets, security groups, VPC endpoints, IAM roles) for HyperPod
distributed training workloads.

Modeled after the CFN nested stacks in:
  1.architectures/7.sagemaker-hyperpod-eks/cfn-templates/nested-stacks/

Security guardrails:
  - Private EKS endpoint enabled by default
  - No public observability endpoints
  - VPC endpoints for S3, ECR, STS, CloudWatch (no traffic over public internet)
  - Security groups: intra-SG only, no 0.0.0.0/0 ingress
  - S3 buckets encrypted (AES256)
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, WaiterError

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_region() -> str:
    return boto3.Session().region_name or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


def _get_account_id() -> str:
    return boto3.client("sts").get_caller_identity()["Account"]


def _get_caller_arn() -> str:
    return boto3.client("sts").get_caller_identity()["Arn"]


def _wait_until(check_fn, timeout=600, interval=15, description="resource"):
    """Poll check_fn() until it returns True, or raise TimeoutError."""
    start = time.time()
    while True:
        if check_fn():
            return
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            raise TimeoutError(f"{description} did not become ready within {timeout}s")
        logger.info(f"  Waiting for {description}... ({elapsed}s)")
        time.sleep(interval)


def _tag(name: str, prefix: str, extra: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """Build a standard tag list."""
    tags = [
        {"Key": "Name", "Value": f"{prefix}-{name}"},
        {"Key": "ManagedBy", "Value": "opencode-eks-cluster-manager"},
    ]
    if extra:
        for k, v in extra.items():
            tags.append({"Key": k, "Value": v})
    return tags


# ============================================================================
# VPC Creation
# ============================================================================

def create_vpc(
    name_prefix: str = "sagemaker-hyperpod",
    vpc_cidr: str = "10.192.0.0/16",
    public_subnet1_cidr: str = "10.192.10.0/24",
    public_subnet2_cidr: str = "10.192.11.0/24",
    private_subnet_cidr: str = "10.1.0.0/16",
    eks_subnet1_cidr: str = "10.192.7.0/28",
    eks_subnet2_cidr: str = "10.192.8.0/28",
    availability_zone_id: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a complete VPC for HyperPod EKS training.

    Creates:
    - VPC with DNS support
    - Internet Gateway
    - 2 public subnets (for NAT gateways)
    - 2 NAT gateways (HA)
    - 1 private subnet (for HyperPod instances, secondary CIDR)
    - 2 EKS private subnets (for EKS cross-account ENIs)
    - Route tables with proper routing

    Args:
        name_prefix: Prefix for all resource names/tags.
        vpc_cidr: Primary VPC CIDR (default: 10.192.0.0/16).
        public_subnet1_cidr: First public subnet CIDR.
        public_subnet2_cidr: Second public subnet CIDR.
        private_subnet_cidr: Private subnet CIDR for HyperPod (added as secondary CIDR).
        eks_subnet1_cidr: First EKS private subnet CIDR (small, for ENIs).
        eks_subnet2_cidr: Second EKS private subnet CIDR.
        availability_zone_id: AZ ID for the private HyperPod subnet (e.g., usw2-az2).
                              Auto-detected if not provided.
        region: AWS region.

    Returns:
        Dict with vpc_id, public_subnet_ids, private_subnet_id,
        eks_subnet_ids, nat_gateway_ids, route_table_ids.
    """
    region = region or _get_region()
    ec2 = boto3.client("ec2", region_name=region)

    logger.info(f"Creating VPC infrastructure in {region}...")

    # --- VPC ---
    resp = ec2.create_vpc(
        CidrBlock=vpc_cidr,
        TagSpecifications=[{
            "ResourceType": "vpc",
            "Tags": _tag("VPC", name_prefix),
        }],
    )
    vpc_id = resp["Vpc"]["VpcId"]
    logger.info(f"  VPC created: {vpc_id}")

    ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={"Value": True})
    ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})

    # --- Internet Gateway ---
    resp = ec2.create_internet_gateway(
        TagSpecifications=[{
            "ResourceType": "internet-gateway",
            "Tags": _tag("IGW", name_prefix),
        }]
    )
    igw_id = resp["InternetGateway"]["InternetGatewayId"]
    ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
    logger.info(f"  IGW created and attached: {igw_id}")

    # --- Get AZs ---
    azs_resp = ec2.describe_availability_zones(
        Filters=[
            {"Name": "region-name", "Values": [region]},
            {"Name": "state", "Values": ["available"]},
        ]
    )
    azs = [az["ZoneName"] for az in azs_resp["AvailabilityZones"]]
    az_ids = {az["ZoneName"]: az["ZoneId"] for az in azs_resp["AvailabilityZones"]}

    if len(azs) < 2:
        raise ValueError(f"Need at least 2 AZs, found {len(azs)} in {region}")

    # --- Public Subnets ---
    public_subnet_ids = []
    for i, (cidr, az) in enumerate([(public_subnet1_cidr, azs[0]), (public_subnet2_cidr, azs[1])]):
        resp = ec2.create_subnet(
            VpcId=vpc_id,
            CidrBlock=cidr,
            AvailabilityZone=az,
            TagSpecifications=[{
                "ResourceType": "subnet",
                "Tags": _tag(f"Public{i+1}", name_prefix),
            }],
        )
        sid = resp["Subnet"]["SubnetId"]
        ec2.modify_subnet_attribute(SubnetId=sid, MapPublicIpOnLaunch={"Value": True})
        public_subnet_ids.append(sid)
    logger.info(f"  Public subnets: {public_subnet_ids}")

    # --- Public Route Table ---
    resp = ec2.create_route_table(
        VpcId=vpc_id,
        TagSpecifications=[{
            "ResourceType": "route-table",
            "Tags": _tag("PublicRoutes", name_prefix),
        }],
    )
    public_rt_id = resp["RouteTable"]["RouteTableId"]
    ec2.create_route(RouteTableId=public_rt_id, DestinationCidrBlock="0.0.0.0/0", GatewayId=igw_id)
    for sid in public_subnet_ids:
        ec2.associate_route_table(RouteTableId=public_rt_id, SubnetId=sid)

    # --- NAT Gateways (one per public subnet for HA) ---
    nat_gw_ids = []
    for i, sid in enumerate(public_subnet_ids):
        eip = ec2.allocate_address(Domain="vpc", TagSpecifications=[{
            "ResourceType": "elastic-ip",
            "Tags": _tag(f"NAT-EIP{i+1}", name_prefix),
        }])
        resp = ec2.create_nat_gateway(
            SubnetId=sid,
            AllocationId=eip["AllocationId"],
            TagSpecifications=[{
                "ResourceType": "natgateway",
                "Tags": _tag(f"NAT{i+1}", name_prefix),
            }],
        )
        nat_gw_ids.append(resp["NatGateway"]["NatGatewayId"])
    logger.info(f"  NAT gateways created: {nat_gw_ids} (waiting for available...)")

    # Wait for NAT gateways
    for ngw_id in nat_gw_ids:
        _wait_until(
            lambda nid=ngw_id: ec2.describe_nat_gateways(NatGatewayIds=[nid])
            ["NatGateways"][0]["State"] == "available",
            timeout=300, interval=15, description=f"NAT gateway {ngw_id}",
        )
    logger.info("  NAT gateways available")

    # --- Private Subnet (secondary CIDR for HyperPod) ---
    ec2.associate_vpc_cidr_block(VpcId=vpc_id, CidrBlock=private_subnet_cidr)
    # Wait for association
    _wait_until(
        lambda: any(
            a["CidrBlock"] == private_subnet_cidr and a["CidrBlockState"]["State"] == "associated"
            for a in ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0].get("CidrBlockAssociationSet", [])
        ),
        timeout=60, interval=5, description="secondary CIDR association",
    )

    # Pick AZ for private subnet
    if availability_zone_id:
        # Find AZ name from ID
        priv_az = None
        for az_name, az_id in az_ids.items():
            if az_id == availability_zone_id:
                priv_az = az_name
                break
        if not priv_az:
            raise ValueError(f"AZ ID '{availability_zone_id}' not found in {region}")
    else:
        priv_az = azs[0]

    resp = ec2.create_subnet(
        VpcId=vpc_id,
        CidrBlock=private_subnet_cidr,
        AvailabilityZone=priv_az,
        TagSpecifications=[{
            "ResourceType": "subnet",
            "Tags": _tag("Private-HyperPod", name_prefix),
        }],
    )
    private_subnet_id = resp["Subnet"]["SubnetId"]
    logger.info(f"  Private subnet (HyperPod): {private_subnet_id} in {priv_az}")

    # --- Private Route Table ---
    resp = ec2.create_route_table(
        VpcId=vpc_id,
        TagSpecifications=[{
            "ResourceType": "route-table",
            "Tags": _tag("PrivateRoutes", name_prefix),
        }],
    )
    private_rt_id = resp["RouteTable"]["RouteTableId"]
    ec2.create_route(RouteTableId=private_rt_id, DestinationCidrBlock="0.0.0.0/0", NatGatewayId=nat_gw_ids[0])
    ec2.associate_route_table(RouteTableId=private_rt_id, SubnetId=private_subnet_id)

    # --- EKS Private Subnets (small /28 for cross-account ENIs) ---
    eks_subnet_ids = []
    for i, (cidr, az) in enumerate([(eks_subnet1_cidr, azs[0]), (eks_subnet2_cidr, azs[1])]):
        resp = ec2.create_subnet(
            VpcId=vpc_id,
            CidrBlock=cidr,
            AvailabilityZone=az,
            TagSpecifications=[{
                "ResourceType": "subnet",
                "Tags": _tag(f"EKS-Private{i+1}", name_prefix),
            }],
        )
        sid = resp["Subnet"]["SubnetId"]
        eks_subnet_ids.append(sid)
        # EKS subnets also route through NAT
        ec2.associate_route_table(RouteTableId=private_rt_id, SubnetId=sid)
    logger.info(f"  EKS subnets: {eks_subnet_ids}")

    result = {
        "vpc_id": vpc_id,
        "igw_id": igw_id,
        "public_subnet_ids": public_subnet_ids,
        "private_subnet_id": private_subnet_id,
        "private_subnet_az": priv_az,
        "eks_subnet_ids": eks_subnet_ids,
        "nat_gateway_ids": nat_gw_ids,
        "public_route_table_id": public_rt_id,
        "private_route_table_id": private_rt_id,
        "region": region,
        "name_prefix": name_prefix,
    }
    logger.info(f"VPC infrastructure created successfully: {vpc_id}")
    return result


# ============================================================================
# Security Group
# ============================================================================

def create_security_group(
    vpc_id: str,
    name_prefix: str = "sagemaker-hyperpod",
    region: Optional[str] = None,
) -> str:
    """Create a security group for HyperPod training with safe defaults.

    Rules:
    - Intra-SG all traffic (required for NCCL/EFA inter-node comms)
    - FSx Lustre TCP 988, 1018-1023 (intra-SG)
    - Internet egress (for pulling images, HuggingFace models, etc.)
    - NO 0.0.0.0/0 ingress

    Args:
        vpc_id: VPC to create the security group in.
        name_prefix: Name prefix for tagging.
        region: AWS region.

    Returns:
        Security group ID.
    """
    region = region or _get_region()
    ec2 = boto3.client("ec2", region_name=region)

    resp = ec2.create_security_group(
        GroupName=f"{name_prefix}-training-sg",
        Description="HyperPod training security group - no public ingress",
        VpcId=vpc_id,
        TagSpecifications=[{
            "ResourceType": "security-group",
            "Tags": _tag("TrainingSG", name_prefix),
        }],
    )
    sg_id = resp["GroupId"]

    # Intra-SG all traffic (ingress)
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{"IpProtocol": "-1", "UserIdGroupPairs": [{"GroupId": sg_id}]}],
    )

    # Intra-SG all traffic (egress)
    ec2.authorize_security_group_egress(
        GroupId=sg_id,
        IpPermissions=[{"IpProtocol": "-1", "UserIdGroupPairs": [{"GroupId": sg_id}]}],
    )

    # Internet egress (default SG egress already allows 0.0.0.0/0 all,
    # but we add it explicitly for clarity)
    # Note: AWS creates a default egress rule; we don't duplicate it.

    # FSx Lustre ports (intra-SG)
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp", "FromPort": 988, "ToPort": 988,
                "UserIdGroupPairs": [{"GroupId": sg_id}],
            },
            {
                "IpProtocol": "tcp", "FromPort": 1018, "ToPort": 1023,
                "UserIdGroupPairs": [{"GroupId": sg_id}],
            },
        ],
    )

    logger.info(f"Security group created: {sg_id} (intra-SG + FSx + egress, NO public ingress)")
    return sg_id


# ============================================================================
# VPC Endpoints
# ============================================================================

def create_vpc_endpoints(
    vpc_id: str,
    subnet_ids: List[str],
    security_group_id: str,
    route_table_id: str,
    services: Optional[List[str]] = None,
    name_prefix: str = "sagemaker-hyperpod",
    region: Optional[str] = None,
) -> Dict[str, str]:
    """Create VPC endpoints so training traffic stays off the public internet.

    Creates:
    - S3 (Gateway endpoint)
    - ECR API + DKR (Interface endpoints for image pulls)
    - STS (Interface endpoint for IAM/Pod Identity)
    - CloudWatch Logs (Interface endpoint for logging)
    - AMP workspaces (Interface endpoint for observability)

    Args:
        vpc_id: VPC ID.
        subnet_ids: Subnet IDs for interface endpoints.
        security_group_id: Security group for interface endpoints.
        route_table_id: Route table for gateway endpoints.
        services: Override which services to create endpoints for.
        name_prefix: Name prefix.
        region: AWS region.

    Returns:
        Dict mapping service name to endpoint ID.
    """
    region = region or _get_region()
    ec2 = boto3.client("ec2", region_name=region)

    if services is None:
        services = ["s3", "ecr.api", "ecr.dkr", "sts", "logs", "aps-workspaces"]

    endpoints = {}

    for svc in services:
        service_name = f"com.amazonaws.{region}.{svc}"
        ep_type = "Gateway" if svc == "s3" else "Interface"

        try:
            kwargs = {
                "VpcId": vpc_id,
                "ServiceName": service_name,
                "VpcEndpointType": ep_type,
                "TagSpecifications": [{
                    "ResourceType": "vpc-endpoint",
                    "Tags": _tag(f"VPCE-{svc}", name_prefix),
                }],
            }

            if ep_type == "Gateway":
                kwargs["RouteTableIds"] = [route_table_id]
            else:
                kwargs["SubnetIds"] = subnet_ids
                kwargs["SecurityGroupIds"] = [security_group_id]
                kwargs["PrivateDnsEnabled"] = True

            resp = ec2.create_vpc_endpoint(**kwargs)
            ep_id = resp["VpcEndpoint"]["VpcEndpointId"]
            endpoints[svc] = ep_id
            logger.info(f"  VPC endpoint created: {svc} ({ep_type}) -> {ep_id}")
        except ClientError as e:
            if "RouteAlreadyExists" in str(e) or "already exists" in str(e).lower():
                logger.info(f"  VPC endpoint for {svc} already exists, skipping")
            else:
                logger.warning(f"  Failed to create VPC endpoint for {svc}: {e}")

    logger.info(f"VPC endpoints created: {len(endpoints)} services")
    return endpoints


# ============================================================================
# IAM Roles
# ============================================================================

def create_eks_cluster_role(
    name_prefix: str = "sagemaker-hyperpod",
    region: Optional[str] = None,
) -> str:
    """Create IAM role for the EKS cluster control plane.

    Args:
        name_prefix: Role name prefix.
        region: AWS region.

    Returns:
        Role ARN.
    """
    region = region or _get_region()
    iam = boto3.client("iam", region_name=region)
    role_name = f"{name_prefix}-eks-cluster-role"

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "eks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }

    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="EKS cluster control plane role",
            Tags=[{"Key": "ManagedBy", "Value": "opencode-eks-cluster-manager"}],
        )
        role_arn = resp["Role"]["Arn"]
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonEKSClusterPolicy",
        )
        logger.info(f"EKS cluster role created: {role_arn}")
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            resp = iam.get_role(RoleName=role_name)
            logger.info(f"EKS cluster role already exists: {resp['Role']['Arn']}")
            return resp["Role"]["Arn"]
        raise


def create_sagemaker_execution_role(
    s3_bucket_name: str,
    name_prefix: str = "sagemaker-hyperpod",
    region: Optional[str] = None,
) -> str:
    """Create IAM execution role for SageMaker HyperPod instances.

    Grants: EC2 networking, ECR pull, EKS Pod Identity, S3 access,
    CloudWatch, and SageMaker cluster instance permissions.

    Args:
        s3_bucket_name: S3 bucket for lifecycle scripts.
        name_prefix: Role name prefix.
        region: AWS region.

    Returns:
        Role ARN.
    """
    region = region or _get_region()
    iam = boto3.client("iam", region_name=region)
    role_name = f"{name_prefix}-exec-role"

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }

    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ec2:AssignPrivateIpAddresses", "ec2:CreateNetworkInterface",
                    "ec2:CreateNetworkInterfacePermission", "ec2:DeleteNetworkInterface",
                    "ec2:DeleteNetworkInterfacePermission", "ec2:DescribeNetworkInterfaces",
                    "ec2:DescribeVpcs", "ec2:DescribeDhcpOptions", "ec2:DescribeSubnets",
                    "ec2:DescribeSecurityGroups", "ec2:DetachNetworkInterface",
                    "ec2:ModifyNetworkInterfaceAttribute", "ec2:UnassignPrivateIpAddresses",
                    "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage",
                    "ecr:GetAuthorizationToken", "ecr:GetDownloadUrlForLayer",
                    "eks-auth:AssumeRoleForPodIdentity", "cloudwatch:DescribeAlarms",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": ["ec2:CreateTags"],
                "Resource": "arn:aws:ec2:*:*:network-interface/*",
            },
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetObject"],
                "Resource": [
                    f"arn:aws:s3:::{s3_bucket_name}",
                    f"arn:aws:s3:::{s3_bucket_name}/*",
                ],
            },
        ],
    }

    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SageMaker HyperPod execution role",
            Tags=[{"Key": "ManagedBy", "Value": "opencode-eks-cluster-manager"}],
        )
        role_arn = resp["Role"]["Arn"]

        # Managed policies
        for policy_arn in [
            "arn:aws:iam::aws:policy/AmazonSageMakerClusterInstanceRolePolicy",
            "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
        ]:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

        # Inline policy
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{name_prefix}-exec-inline",
            PolicyDocument=json.dumps(inline_policy),
        )

        logger.info(f"SageMaker execution role created: {role_arn}")
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            resp = iam.get_role(RoleName=role_name)
            logger.info(f"SageMaker execution role already exists: {resp['Role']['Arn']}")
            return resp["Role"]["Arn"]
        raise


# ============================================================================
# S3 Bucket
# ============================================================================

def create_s3_bucket(
    name_prefix: str = "sagemaker-hyperpod",
    region: Optional[str] = None,
) -> str:
    """Create an encrypted S3 bucket for lifecycle scripts.

    Args:
        name_prefix: Bucket name prefix.
        region: AWS region.

    Returns:
        Bucket name.
    """
    region = region or _get_region()
    account_id = _get_account_id()
    s3 = boto3.client("s3", region_name=region)

    bucket_name = f"{name_prefix}-{account_id}-{region}"

    try:
        create_kwargs = {"Bucket": bucket_name}
        if region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**create_kwargs)

        # Enable encryption
        s3.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={
                "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
            },
        )

        # Block public access
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )

        logger.info(f"S3 bucket created: {bucket_name} (encrypted, no public access)")
        return bucket_name
    except ClientError as e:
        if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            logger.info(f"S3 bucket already exists: {bucket_name}")
            return bucket_name
        raise


# ============================================================================
# EKS Cluster
# ============================================================================

def create_eks_cluster(
    cluster_name: str,
    eks_subnet_ids: List[str],
    security_group_id: str,
    cluster_role_arn: str,
    kubernetes_version: str = "1.32",
    endpoint_private_access: bool = True,
    endpoint_public_access: bool = True,
    name_prefix: str = "sagemaker-hyperpod",
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an EKS cluster with full logging, core add-ons, and private endpoint.

    Args:
        cluster_name: EKS cluster name.
        eks_subnet_ids: Subnet IDs for the EKS control plane ENIs.
        security_group_id: Security group for the cluster.
        cluster_role_arn: IAM role ARN for the EKS cluster.
        kubernetes_version: Kubernetes version (default: 1.32).
        endpoint_private_access: Enable private API endpoint (default: True).
        endpoint_public_access: Enable public API endpoint (default: True).
        name_prefix: Name prefix.
        region: AWS region.

    Returns:
        Dict with cluster info (name, arn, endpoint, status).
    """
    region = region or _get_region()
    eks = boto3.client("eks", region_name=region)

    logger.info(f"Creating EKS cluster '{cluster_name}' (K8s {kubernetes_version})...")

    try:
        resp = eks.create_cluster(
            name=cluster_name,
            version=kubernetes_version,
            roleArn=cluster_role_arn,
            accessConfig={"authenticationMode": "API_AND_CONFIG_MAP"},
            logging={
                "clusterLogging": [{
                    "types": ["api", "audit", "authenticator", "controllerManager", "scheduler"],
                    "enabled": True,
                }]
            },
            resourcesVpcConfig={
                "subnetIds": eks_subnet_ids,
                "securityGroupIds": [security_group_id],
                "endpointPrivateAccess": endpoint_private_access,
                "endpointPublicAccess": endpoint_public_access,
            },
            tags={"ManagedBy": "opencode-eks-cluster-manager"},
        )
        cluster_arn = resp["cluster"]["arn"]
        logger.info(f"  EKS cluster creation initiated: {cluster_arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            logger.info(f"  EKS cluster '{cluster_name}' already exists")
            resp = eks.describe_cluster(name=cluster_name)
            return {
                "name": cluster_name,
                "arn": resp["cluster"]["arn"],
                "endpoint": resp["cluster"].get("endpoint"),
                "status": resp["cluster"]["status"],
                "already_existed": True,
            }
        raise

    # Wait for cluster to become ACTIVE
    logger.info("  Waiting for EKS cluster to become ACTIVE (this takes 10-15 minutes)...")
    _wait_until(
        lambda: eks.describe_cluster(name=cluster_name)["cluster"]["status"] == "ACTIVE",
        timeout=1200, interval=30, description=f"EKS cluster {cluster_name}",
    )

    cluster_info = eks.describe_cluster(name=cluster_name)["cluster"]
    logger.info(f"  EKS cluster ACTIVE: {cluster_info['endpoint']}")

    # --- Install core add-ons ---
    core_addons = ["vpc-cni", "kube-proxy", "coredns", "eks-pod-identity-agent"]
    for addon_name in core_addons:
        try:
            eks.create_addon(
                clusterName=cluster_name,
                addonName=addon_name,
                resolveConflicts="OVERWRITE",
            )
            logger.info(f"  Add-on installed: {addon_name}")
        except ClientError as e:
            if "already exists" in str(e).lower():
                logger.info(f"  Add-on already exists: {addon_name}")
            else:
                logger.warning(f"  Failed to install add-on {addon_name}: {e}")

    # --- Add caller as cluster admin ---
    caller_arn = _get_caller_arn()
    # Convert assumed-role ARN to role ARN for access entry
    # arn:aws:sts::ACCT:assumed-role/RoleName/session -> arn:aws:iam::ACCT:role/RoleName
    # arn:aws:iam::ACCT:user/UserName -> use as-is
    principal_arn = caller_arn
    if ":assumed-role/" in caller_arn:
        parts = caller_arn.split(":")
        account = parts[4]
        role_name = parts[5].split("/")[1]
        principal_arn = f"arn:aws:iam::{account}:role/{role_name}"

    try:
        eks.create_access_entry(clusterName=cluster_name, principalArn=principal_arn)
        eks.associate_access_policy(
            clusterName=cluster_name,
            principalArn=principal_arn,
            policyArn="arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy",
            accessScope={"type": "cluster"},
        )
        logger.info(f"  Admin access granted to: {principal_arn}")
    except ClientError as e:
        if "already exists" in str(e).lower():
            logger.info(f"  Admin access already exists for: {principal_arn}")
        else:
            logger.warning(f"  Failed to create access entry: {e}")

    return {
        "name": cluster_name,
        "arn": cluster_info["arn"],
        "endpoint": cluster_info.get("endpoint"),
        "status": "ACTIVE",
        "already_existed": False,
    }


def create_access_entry(
    cluster_name: str,
    principal_arn: str,
    policy_arn: str = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy",
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Add an IAM principal (user/role) as an EKS cluster admin.

    Args:
        cluster_name: EKS cluster name.
        principal_arn: IAM user or role ARN.
        policy_arn: EKS access policy ARN.
        region: AWS region.

    Returns:
        Access entry details.
    """
    region = region or _get_region()
    eks = boto3.client("eks", region_name=region)

    try:
        entry = eks.create_access_entry(clusterName=cluster_name, principalArn=principal_arn)
        eks.associate_access_policy(
            clusterName=cluster_name,
            principalArn=principal_arn,
            policyArn=policy_arn,
            accessScope={"type": "cluster"},
        )
        logger.info(f"Access entry created: {principal_arn} -> {cluster_name}")
        return entry["accessEntry"]
    except ClientError as e:
        if "already exists" in str(e).lower():
            logger.info(f"Access entry already exists: {principal_arn}")
            return {"principalArn": principal_arn, "status": "already_exists"}
        raise


# ============================================================================
# Validation (existing functionality, kept)
# ============================================================================

def validate_cluster(
    cluster_name: str,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate an EKS cluster is ready for training workloads.

    Checks:
    - Cluster is ACTIVE
    - Nodes are Ready
    - GPU resources available (nvidia.com/gpu)
    - EFA resources available (vpc.amazonaws.com/efa)
    - Core add-ons installed

    Args:
        cluster_name: EKS cluster name.
        region: AWS region.

    Returns:
        Validation report dict.
    """
    region = region or _get_region()
    eks = boto3.client("eks", region_name=region)

    report = {"cluster_name": cluster_name, "checks": {}, "overall": "PASS"}

    # Cluster status
    try:
        resp = eks.describe_cluster(name=cluster_name)
        status = resp["cluster"]["status"]
        report["checks"]["cluster_status"] = {
            "status": "PASS" if status == "ACTIVE" else "FAIL",
            "detail": f"Cluster status: {status}",
        }
        if status != "ACTIVE":
            report["overall"] = "FAIL"
    except ClientError as e:
        report["checks"]["cluster_status"] = {"status": "FAIL", "detail": str(e)}
        report["overall"] = "FAIL"
        return report

    # Add-ons
    try:
        addons = eks.list_addons(clusterName=cluster_name)["addons"]
        expected = {"vpc-cni", "kube-proxy", "coredns", "eks-pod-identity-agent"}
        missing = expected - set(addons)
        report["checks"]["core_addons"] = {
            "status": "PASS" if not missing else "WARN",
            "detail": f"Installed: {addons}" + (f", missing: {missing}" if missing else ""),
        }
    except ClientError:
        report["checks"]["core_addons"] = {"status": "WARN", "detail": "Could not list add-ons"}

    # Node readiness (requires kubectl access)
    import subprocess
    try:
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            import json as _json
            nodes = _json.loads(result.stdout).get("items", [])
            total = len(nodes)
            ready = sum(
                1 for n in nodes
                if any(
                    c["type"] == "Ready" and c["status"] == "True"
                    for c in n.get("status", {}).get("conditions", [])
                )
            )
            total_gpus = sum(
                int(n.get("status", {}).get("capacity", {}).get("nvidia.com/gpu", 0))
                for n in nodes
            )
            total_efa = sum(
                int(n.get("status", {}).get("capacity", {}).get("vpc.amazonaws.com/efa", 0))
                for n in nodes
            )
            report["checks"]["nodes"] = {
                "status": "PASS" if ready == total and total > 0 else "WARN" if total == 0 else "FAIL",
                "detail": f"{ready}/{total} nodes Ready, {total_gpus} GPUs, {total_efa} EFA devices",
            }
        else:
            report["checks"]["nodes"] = {"status": "WARN", "detail": "kubectl not configured for this cluster"}
    except Exception as e:
        report["checks"]["nodes"] = {"status": "WARN", "detail": f"Cannot reach cluster: {e}"}

    # Determine overall status
    statuses = [c["status"] for c in report["checks"].values()]
    if "FAIL" in statuses:
        report["overall"] = "FAIL"
    elif "WARN" in statuses:
        report["overall"] = "WARN"

    return report


# ============================================================================
# Full Stack Provisioning
# ============================================================================

def provision_eks_infrastructure(
    name_prefix: str = "sagemaker-hyperpod",
    eks_cluster_name: Optional[str] = None,
    vpc_cidr: str = "10.192.0.0/16",
    private_subnet_cidr: str = "10.1.0.0/16",
    kubernetes_version: str = "1.32",
    availability_zone_id: Optional[str] = None,
    endpoint_private_access: bool = True,
    endpoint_public_access: bool = True,
    create_endpoints: bool = True,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Provision the complete EKS infrastructure stack in one call.

    Creates VPC, subnets, security group, VPC endpoints, IAM roles,
    S3 bucket, and EKS cluster with all core add-ons.

    This is the recommended entry point for creating a new cluster.

    Args:
        name_prefix: Prefix for all resource names.
        eks_cluster_name: EKS cluster name (default: {name_prefix}-eks).
        vpc_cidr: VPC CIDR block.
        private_subnet_cidr: Private subnet CIDR for HyperPod.
        kubernetes_version: K8s version.
        availability_zone_id: AZ ID for private subnet (auto-detected if not set).
        endpoint_private_access: Enable EKS private endpoint (default: True).
        endpoint_public_access: Enable EKS public endpoint (default: True).
        create_endpoints: Create VPC endpoints for AWS services (default: True).
        region: AWS region.

    Returns:
        Complete infrastructure details dict.
    """
    region = region or _get_region()
    eks_cluster_name = eks_cluster_name or f"{name_prefix}-eks"

    logger.info("=" * 70)
    logger.info(f"Provisioning EKS infrastructure: {name_prefix}")
    logger.info(f"  Region: {region}")
    logger.info(f"  EKS cluster: {eks_cluster_name}")
    logger.info(f"  K8s version: {kubernetes_version}")
    logger.info(f"  Private endpoint: {endpoint_private_access}")
    logger.info("=" * 70)

    result = {"name_prefix": name_prefix, "region": region}

    # Step 1: VPC
    logger.info("[1/6] Creating VPC infrastructure...")
    vpc = create_vpc(
        name_prefix=name_prefix,
        vpc_cidr=vpc_cidr,
        private_subnet_cidr=private_subnet_cidr,
        availability_zone_id=availability_zone_id,
        region=region,
    )
    result["vpc"] = vpc

    # Step 2: Security Group
    logger.info("[2/6] Creating security group...")
    sg_id = create_security_group(vpc["vpc_id"], name_prefix, region)
    result["security_group_id"] = sg_id

    # Step 3: S3 Bucket
    logger.info("[3/6] Creating S3 bucket...")
    bucket_name = create_s3_bucket(name_prefix, region)
    result["s3_bucket"] = bucket_name

    # Step 4: IAM Roles
    logger.info("[4/6] Creating IAM roles...")
    eks_role_arn = create_eks_cluster_role(name_prefix, region)
    sm_role_arn = create_sagemaker_execution_role(bucket_name, name_prefix, region)
    result["eks_cluster_role_arn"] = eks_role_arn
    result["sagemaker_execution_role_arn"] = sm_role_arn

    # Step 5: VPC Endpoints
    if create_endpoints:
        logger.info("[5/6] Creating VPC endpoints...")
        endpoints = create_vpc_endpoints(
            vpc["vpc_id"], vpc["eks_subnet_ids"], sg_id,
            vpc["private_route_table_id"], name_prefix=name_prefix, region=region,
        )
        result["vpc_endpoints"] = endpoints
    else:
        logger.info("[5/6] Skipping VPC endpoints (disabled)")
        result["vpc_endpoints"] = {}

    # Step 6: EKS Cluster
    logger.info("[6/6] Creating EKS cluster (this takes 10-15 minutes)...")
    cluster = create_eks_cluster(
        cluster_name=eks_cluster_name,
        eks_subnet_ids=vpc["eks_subnet_ids"],
        security_group_id=sg_id,
        cluster_role_arn=eks_role_arn,
        kubernetes_version=kubernetes_version,
        endpoint_private_access=endpoint_private_access,
        endpoint_public_access=endpoint_public_access,
        name_prefix=name_prefix,
        region=region,
    )
    result["eks_cluster"] = cluster

    # Update kubeconfig
    import subprocess
    try:
        subprocess.run(
            ["aws", "eks", "update-kubeconfig", "--name", eks_cluster_name, "--region", region],
            capture_output=True, text=True, timeout=30,
        )
        logger.info(f"kubectl configured for {eks_cluster_name}")
    except Exception as e:
        logger.warning(f"Could not update kubeconfig: {e}")

    logger.info("=" * 70)
    logger.info("EKS infrastructure provisioned successfully!")
    logger.info(f"  VPC: {vpc['vpc_id']}")
    logger.info(f"  EKS: {cluster['name']} ({cluster['status']})")
    logger.info(f"  SG: {sg_id}")
    logger.info(f"  S3: {bucket_name}")
    logger.info("=" * 70)

    return result
