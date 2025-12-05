# ============================================================================
# NEURECTOMY - Terraform Environment Configurations
# ============================================================================

# Development Environment
# terraform apply -var-file="environments/dev.tfvars"
# ============================================================================

# environments/dev.tfvars
environment        = "dev"
region             = "us-east-1"
cluster_version    = "1.29"
vpc_cidr           = "10.0.0.0/16"
availability_zones = ["us-east-1a", "us-east-1b"]
enable_gpu_nodes   = false

# environments/staging.tfvars
# environment        = "staging"
# region             = "us-east-1"
# cluster_version    = "1.29"
# vpc_cidr           = "10.1.0.0/16"
# availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
# enable_gpu_nodes   = false

# environments/prod.tfvars
# environment        = "prod"
# region             = "us-east-1"
# cluster_version    = "1.29"
# vpc_cidr           = "10.2.0.0/16"
# availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
# enable_gpu_nodes   = true
