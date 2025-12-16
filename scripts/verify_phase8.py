#!/usr/bin/env python3
"""Phase 8 Verification - Deployment & DevOps"""

import sys
from pathlib import Path


def verify_dockerfile():
    try:
        dockerfile = Path("Dockerfile")
        if not dockerfile.exists():
            print("❌ Dockerfile not found")
            return False
        
        content = dockerfile.read_text()
        checks = [
            "FROM python:3.11-slim" in content,
            "multi-stage" in content.lower() or "as builder" in content,
            "uvicorn" in content,
            "HEALTHCHECK" in content,
        ]
        
        if all(checks):
            print("✓ Dockerfile verified")
            return True
        else:
            print("❌ Dockerfile missing required elements")
            return False
    except Exception as e:
        print(f"❌ Dockerfile verification failed: {e}")
        return False


def verify_docker_compose():
    try:
        compose = Path("docker-compose.yml")
        if not compose.exists():
            print("❌ docker-compose.yml not found")
            return False
        
        content = compose.read_text()
        checks = [
            "neurectomy-api" in content,
            "prometheus" in content,
            "grafana" in content,
            "redis" in content,
        ]
        
        if all(checks):
            print("✓ docker-compose.yml verified")
            return True
        else:
            print("❌ docker-compose.yml missing services")
            return False
    except Exception as e:
        print(f"❌ docker-compose.yml verification failed: {e}")
        return False


def verify_k8s_manifests():
    try:
        k8s_dir = Path("deploy/k8s")
        if not k8s_dir.exists():
            print("❌ deploy/k8s directory not found")
            return False
        
        deployment = k8s_dir / "deployment.yaml"
        if not deployment.exists():
            print("❌ deployment.yaml not found")
            return False
        
        content = deployment.read_text()
        checks = [
            "kind: Deployment" in content,
            "kind: Service" in content,
            "kind: HorizontalPodAutoscaler" in content,
            "livenessProbe" in content,
            "readinessProbe" in content,
        ]
        
        if all(checks):
            print("✓ Kubernetes manifests verified")
            return True
        else:
            print("❌ Kubernetes manifests missing required resources")
            return False
    except Exception as e:
        print(f"❌ Kubernetes manifests verification failed: {e}")
        return False


def verify_ci_pipeline():
    try:
        ci_file = Path(".github/workflows/ci.yml")
        if not ci_file.exists():
            print("❌ CI pipeline not found")
            return False
        
        content = ci_file.read_text()
        checks = [
            "name:" in content,
            "jobs:" in content,
            "test" in content.lower(),
            "build" in content.lower(),
        ]
        
        if all(checks):
            print("✓ CI/CD pipeline verified")
            return True
        else:
            print("❌ CI/CD pipeline missing required jobs")
            return False
    except Exception as e:
        print(f"❌ CI/CD pipeline verification failed: {e}")
        return False


def verify_dockerignore():
    try:
        dockerignore = Path(".dockerignore")
        if dockerignore.exists():
            print("✓ .dockerignore found")
            return True
        else:
            print("⚠ .dockerignore not found (recommended)")
            return True  # Not critical
    except Exception as e:
        print(f"⚠ .dockerignore check failed: {e}")
        return True  # Not critical


def main():
    print("=" * 60)
    print("  PHASE 8: Deployment & DevOps - Verification")
    print("=" * 60)
    print()
    
    results = [
        verify_dockerfile(),
        verify_docker_compose(),
        verify_k8s_manifests(),
        verify_ci_pipeline(),
        verify_dockerignore(),
    ]
    
    print()
    if all(results):
        print("=" * 60)
        print("  ✅ PHASE 8 VERIFICATION COMPLETE")
        print("=" * 60)
        print()
        print("  Deployment artifacts ready:")
        print("    • Dockerfile (multi-stage)")
        print("    • docker-compose.yml (4 services)")
        print("    • Kubernetes manifests")
        print("    • CI/CD pipeline")
        print()
        print("  Deploy with:")
        print("    docker-compose up -d")
        print("    kubectl apply -f deploy/k8s/")
        return 0
    else:
        print("=" * 60)
        print("  ❌ PHASE 8 VERIFICATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
