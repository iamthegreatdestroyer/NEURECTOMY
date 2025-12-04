"""
API Routes Aggregator.

@SYNAPSE - Combines all route modules into a unified router.
"""

from fastapi import APIRouter

from src.api import llm, embeddings, training, agents, health, analytics, auth

router = APIRouter()

# Include all route modules
router.include_router(
    health.router,
    tags=["health"],
)

router.include_router(
    auth.router,
    tags=["authentication"],
)

router.include_router(
    llm.router,
    prefix="/llm",
    tags=["llm"],
)

router.include_router(
    embeddings.router,
    prefix="/embeddings",
    tags=["embeddings"],
)

router.include_router(
    training.router,
    prefix="/training",
    tags=["training"],
)

router.include_router(
    agents.router,
    prefix="/agents",
    tags=["agents"],
)

router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"],
)
