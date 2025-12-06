# ============================================================================
# NEURECTOMY - Production Environment Configuration
# High Availability, Multi-AZ, Enterprise-Grade Settings
# ============================================================================

environment = "prod"
region      = "us-east-1"
dr_region   = "us-west-2"

# ============================================================================
# Network Configuration - Multi-AZ
# ============================================================================

vpc_cidr           = "10.2.0.0/16"
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

# Enable VPC flow logs for security and compliance
enable_vpc_flow_logs = true

# ============================================================================
# EKS Cluster
# ============================================================================

cluster_name    = "neurectomy-cluster"
cluster_version = "1.29"

# Restrict public access in production
cluster_endpoint_public_access = true
cluster_endpoint_public_access_cidrs = [
  # Add your corporate IP ranges here
  # "203.0.113.0/24",
  "0.0.0.0/0"  # Remove this in real production
]

# Full logging in production
cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

# ============================================================================
# Node Groups - Production Grade
# ============================================================================

general_node_instance_types = ["m6i.xlarge", "m6i.2xlarge"]
general_node_min_size       = 3
general_node_max_size       = 10
general_node_desired_size   = 3

ml_node_instance_types = ["c6i.2xlarge", "c6i.4xlarge"]

# Enable GPU for production ML training
enable_gpu_nodes        = true
gpu_node_instance_types = ["g5.xlarge", "g5.2xlarge"]

# ============================================================================
# Add-ons - Full Feature Set
# ============================================================================

enable_cluster_autoscaler           = true
enable_metrics_server               = true
enable_aws_load_balancer_controller = true
enable_external_dns                 = true
enable_cert_manager                 = true
enable_external_secrets             = true
enable_argocd                       = true
enable_prometheus_stack             = true

# ============================================================================
# Database - Managed RDS
# ============================================================================

enable_rds            = true
rds_instance_class    = "db.r6g.large"
rds_allocated_storage = 100
rds_multi_az          = true

# ============================================================================
# Storage
# ============================================================================

enable_efs          = true
efs_throughput_mode = "provisioned"

# ============================================================================
# Monitoring - Full Retention
# ============================================================================

log_retention_days        = 90
enable_container_insights = true

# ============================================================================
# Security - Enterprise Grade
# ============================================================================

enable_guardduty        = true
enable_security_hub     = true
kms_key_deletion_window = 30

# ============================================================================
# Domain Configuration
# ============================================================================

# domain_name    = "neurectomy.example.com"
# hosted_zone_id = "Z1234567890ABC"

# ============================================================================
# Cost Management
# ============================================================================

enable_spot_instances = false  # Use on-demand for reliability in prod
budget_limit          = 2000
budget_alert_emails   = [
  # "devops@example.com",
  # "finance@example.com"
]

# ============================================================================
# Additional Tags
# ============================================================================

additional_tags = {
  CostCenter   = "production"
  DataClass    = "confidential"
  Compliance   = "soc2"
  BackupPolicy = "daily"
}
