# ============================================================================
# NEURECTOMY - Terraform Variables Definition
# Complete variable definitions with validation and documentation
# ============================================================================

# ============================================================================
# Core Configuration
# ============================================================================

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "neurectomy"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*$", var.project_name))
    error_message = "Project name must start with a letter and contain only lowercase letters, numbers, and hyphens."
  }
}

variable "region" {
  description = "AWS region for primary deployment"
  type        = string
  default     = "us-east-1"
}

variable "dr_region" {
  description = "AWS region for disaster recovery"
  type        = string
  default     = "us-west-2"
}

# ============================================================================
# Networking Configuration
# ============================================================================

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "availability_zones" {
  description = "Availability zones for multi-AZ deployment"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]

  validation {
    condition     = length(var.availability_zones) >= 2
    error_message = "At least 2 availability zones are required for high availability."
  }
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC flow logs for network monitoring"
  type        = bool
  default     = true
}

# ============================================================================
# EKS Cluster Configuration
# ============================================================================

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "neurectomy-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.29"

  validation {
    condition     = can(regex("^1\\.(2[789]|30)$", var.cluster_version))
    error_message = "Cluster version must be 1.27, 1.28, 1.29, or 1.30."
  }
}

variable "cluster_endpoint_public_access" {
  description = "Enable public access to cluster API endpoint"
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "CIDR blocks allowed to access cluster API endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "cluster_enabled_log_types" {
  description = "List of control plane logging to enable"
  type        = list(string)
  default     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
}

# ============================================================================
# Node Group Configuration
# ============================================================================

variable "general_node_instance_types" {
  description = "Instance types for general workload node group"
  type        = list(string)
  default     = ["t3.large", "t3.xlarge"]
}

variable "general_node_min_size" {
  description = "Minimum number of nodes in general node group"
  type        = number
  default     = 1

  validation {
    condition     = var.general_node_min_size >= 0
    error_message = "Minimum node size must be >= 0."
  }
}

variable "general_node_max_size" {
  description = "Maximum number of nodes in general node group"
  type        = number
  default     = 10

  validation {
    condition     = var.general_node_max_size >= 1
    error_message = "Maximum node size must be >= 1."
  }
}

variable "general_node_desired_size" {
  description = "Desired number of nodes in general node group"
  type        = number
  default     = 2
}

variable "ml_node_instance_types" {
  description = "Instance types for ML inference node group"
  type        = list(string)
  default     = ["c6i.xlarge", "c6i.2xlarge"]
}

variable "enable_gpu_nodes" {
  description = "Enable GPU node group for ML training"
  type        = bool
  default     = false
}

variable "gpu_node_instance_types" {
  description = "Instance types for GPU training node group"
  type        = list(string)
  default     = ["g5.xlarge", "g5.2xlarge"]
}

# ============================================================================
# Add-ons Configuration
# ============================================================================

variable "enable_cluster_autoscaler" {
  description = "Enable Cluster Autoscaler"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Enable Metrics Server for HPA"
  type        = bool
  default     = true
}

variable "enable_aws_load_balancer_controller" {
  description = "Enable AWS Load Balancer Controller"
  type        = bool
  default     = true
}

variable "enable_external_dns" {
  description = "Enable ExternalDNS for automatic DNS management"
  type        = bool
  default     = true
}

variable "enable_cert_manager" {
  description = "Enable cert-manager for TLS certificate management"
  type        = bool
  default     = true
}

variable "enable_external_secrets" {
  description = "Enable External Secrets Operator"
  type        = bool
  default     = true
}

variable "enable_argocd" {
  description = "Enable ArgoCD for GitOps"
  type        = bool
  default     = true
}

variable "enable_prometheus_stack" {
  description = "Enable Prometheus monitoring stack"
  type        = bool
  default     = true
}

# ============================================================================
# Database Configuration
# ============================================================================

variable "enable_rds" {
  description = "Enable RDS PostgreSQL instance"
  type        = bool
  default     = false
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 50
}

variable "rds_multi_az" {
  description = "Enable Multi-AZ for RDS"
  type        = bool
  default     = false
}

# ============================================================================
# Storage Configuration
# ============================================================================

variable "enable_efs" {
  description = "Enable EFS for shared storage"
  type        = bool
  default     = true
}

variable "efs_throughput_mode" {
  description = "EFS throughput mode (bursting or provisioned)"
  type        = string
  default     = "bursting"

  validation {
    condition     = contains(["bursting", "provisioned"], var.efs_throughput_mode)
    error_message = "EFS throughput mode must be 'bursting' or 'provisioned'."
  }
}

# ============================================================================
# Monitoring & Observability
# ============================================================================

variable "log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention value."
  }
}

variable "enable_container_insights" {
  description = "Enable CloudWatch Container Insights"
  type        = bool
  default     = true
}

# ============================================================================
# Security Configuration
# ============================================================================

variable "enable_guardduty" {
  description = "Enable GuardDuty for threat detection"
  type        = bool
  default     = true
}

variable "enable_security_hub" {
  description = "Enable Security Hub for security findings"
  type        = bool
  default     = false
}

variable "kms_key_deletion_window" {
  description = "KMS key deletion window in days"
  type        = number
  default     = 30

  validation {
    condition     = var.kms_key_deletion_window >= 7 && var.kms_key_deletion_window <= 30
    error_message = "KMS key deletion window must be between 7 and 30 days."
  }
}

# ============================================================================
# Domain & DNS Configuration
# ============================================================================

variable "domain_name" {
  description = "Primary domain name for the application"
  type        = string
  default     = ""
}

variable "hosted_zone_id" {
  description = "Route53 hosted zone ID for DNS records"
  type        = string
  default     = ""
}

# ============================================================================
# Cost Management
# ============================================================================

variable "enable_spot_instances" {
  description = "Enable spot instances for cost savings"
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Maximum price for spot instances (empty = on-demand price)"
  type        = string
  default     = ""
}

variable "budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 500
}

variable "budget_alert_emails" {
  description = "Email addresses for budget alerts"
  type        = list(string)
  default     = []
}

# ============================================================================
# Tags
# ============================================================================

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
