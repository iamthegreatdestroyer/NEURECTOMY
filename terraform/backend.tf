# ============================================================================
# NEURECTOMY - Terraform Backend Configuration
# Remote state with S3 and DynamoDB locking
# ============================================================================

terraform {
  backend "s3" {
    # ========================================================================
    # IMPORTANT: Replace with your actual values
    # ========================================================================
    bucket = "neurectomy-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"

    # Enable encryption at rest
    encrypt = true

    # DynamoDB table for state locking
    dynamodb_table = "neurectomy-terraform-locks"

    # Optional: Use assume role for cross-account deployments
    # role_arn = "arn:aws:iam::ACCOUNT_ID:role/TerraformRole"

    # Optional: SSE-KMS encryption
    # kms_key_id = "arn:aws:kms:us-east-1:ACCOUNT_ID:key/KEY_ID"
  }
}

# ============================================================================
# Bootstrap Script (run once to create backend infrastructure)
# ============================================================================
# 
# Run this AWS CLI script to create the S3 bucket and DynamoDB table:
#
# aws s3api create-bucket \
#     --bucket neurectomy-terraform-state \
#     --region us-east-1
#
# aws s3api put-bucket-versioning \
#     --bucket neurectomy-terraform-state \
#     --versioning-configuration Status=Enabled
#
# aws s3api put-bucket-encryption \
#     --bucket neurectomy-terraform-state \
#     --server-side-encryption-configuration \
#     '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"aws:kms"}}]}'
#
# aws s3api put-public-access-block \
#     --bucket neurectomy-terraform-state \
#     --public-access-block-configuration \
#     'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'
#
# aws dynamodb create-table \
#     --table-name neurectomy-terraform-locks \
#     --attribute-definitions AttributeName=LockID,AttributeType=S \
#     --key-schema AttributeName=LockID,KeyType=HASH \
#     --billing-mode PAY_PER_REQUEST \
#     --region us-east-1
#
# ============================================================================
