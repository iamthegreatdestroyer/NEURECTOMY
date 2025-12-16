# Neurectomy Infrastructure

This directory contains shared infrastructure components for the Neurectomy Unified Architecture.

## Directory Structure

- **scripts/** - Backup, restore, and maintenance scripts
- **security/** - Security scanning and compliance tools
- **kubernetes/** - Kubernetes manifests for deployment
- **terraform/** - Infrastructure-as-code for cloud resources
- **helm/** - Helm charts for unified deployment

## Components Hosted Here

### Backup & Recovery

- `scripts/backup_manager.py` - Automated backup system
- `scripts/restore_manager.py` - Point-in-time recovery

### Security

- `security/vulnerability_scanner.py` - Multi-layer security scanning
- `security/secrets_manager.py` - Encrypted secrets management
- `security/compliance/gdpr_manager.py` - GDPR compliance tools

### Deployment

- `kubernetes/` - Production Kubernetes configurations
- `terraform/` - AWS/cloud infrastructure provisioning
- `helm/` - Unified Helm chart for all services

## Usage

Infrastructure scripts can be run from this directory:

```bash
# Run backup
python infrastructure/scripts/backup_manager.py

# Security scan
python infrastructure/security/vulnerability_scanner.py

# Deploy to Kubernetes
kubectl apply -f infrastructure/kubernetes/
```

## Notes

- This infrastructure serves ALL projects: Neurectomy, Ryot LLM, ΣLANG, ΣVAULT
- Scripts run at the system level and coordinate across all services
- Phase 14 deployment configs are also stored here
