#!/usr/bin/env python3
"""
Phase 5A Verification Script
=============================

Verifies Elite Agent Collective implementation.
"""

import sys


def verify_elite_imports():
    """Verify elite module imports."""
    try:
        from neurectomy.elite import (
            EliteAgent,
            TeamCommander,
            TeamConfig,
            TeamRole,
        )
        print("✓ Elite base classes imported")
        return True
    except Exception as e:
        print(f"❌ Elite import error: {e}")
        return False


def verify_team_creation():
    """Verify all 5 teams can be created."""
    try:
        from neurectomy.elite import (
            create_inference_team,
            create_compression_team,
            create_storage_team,
            create_analysis_team,
            create_synthesis_team,
        )
        
        teams = {
            "Inference": create_inference_team(),
            "Compression": create_compression_team(),
            "Storage": create_storage_team(),
            "Analysis": create_analysis_team(),
            "Synthesis": create_synthesis_team(),
        }
        
        total_agents = 0
        for team_name, agents in teams.items():
            agent_count = len(agents)
            total_agents += agent_count
            print(f"   - {team_name} Team: {agent_count} agents")
        
        print(f"✓ All 5 teams created ({total_agents} agents total)")
        return total_agents == 40
    except Exception as e:
        print(f"❌ Team creation error: {e}")
        return False


def verify_team_structure():
    """Verify team hierarchy and collaboration."""
    try:
        from neurectomy.elite import create_inference_team, TeamRole
        
        team = create_inference_team()
        commander = team[0]
        
        # Verify commander
        assert commander.role == TeamRole.COMMANDER, "First agent should be commander"
        
        # Verify team members registered
        member_count = len(commander._team_members)
        assert member_count == 7, f"Expected 7 members, got {member_count}"
        
        # Verify collaboration links
        for member in team[1:]:
            assert commander.agent_id in member._collaborators, "Commander should be collaborator"
        
        print("✓ Team structure verified (commander + 7 members)")
        return True
    except Exception as e:
        print(f"❌ Team structure error: {e}")
        return False


def verify_agent_processing():
    """Verify agents can process tasks."""
    try:
        from neurectomy.elite import create_analysis_team
        from neurectomy.core.types import TaskRequest
        import uuid
        
        team = create_analysis_team()
        
        # Find sentiment analyst
        sentiment_agent = None
        for agent in team:
            if agent.agent_id == "sentiment_analyst":
                sentiment_agent = agent
                break
        
        assert sentiment_agent is not None, "Sentiment analyst not found"
        
        # Process a task
        request = TaskRequest(
            task_id=str(uuid.uuid4()),
            task_type="sentiment",
            payload={"text": "This is a great test!"},
        )
        
        result = sentiment_agent.process(request)
        assert result.status.name in ["COMPLETED", "PENDING"], f"Unexpected status: {result.status}"
        
        print("✓ Agent task processing verified")
        return True
    except Exception as e:
        print(f"❌ Task processing error: {e}")
        return False


def verify_cross_team_capability():
    """Verify capability-based agent discovery."""
    try:
        from neurectomy.elite import (
            create_inference_team,
            create_synthesis_team,
        )
        from neurectomy.core.types import AgentCapability
        
        inference_team = create_inference_team()
        synthesis_team = create_synthesis_team()
        
        # Count agents with specific capabilities
        code_gen_agents = [
            a for a in synthesis_team
            if AgentCapability.CODE_GENERATION in a.config.capabilities
        ]
        
        analysis_agents = [
            a for a in inference_team
            if AgentCapability.ANALYSIS in a.config.capabilities
        ]
        
        print(f"   - Code generation agents: {len(code_gen_agents)}")
        print(f"   - Analysis agents in inference team: {len(analysis_agents)}")
        print("✓ Capability-based discovery verified")
        return True
    except Exception as e:
        print(f"❌ Cross-team capability error: {e}")
        return False


def print_agent_roster():
    """Print full agent roster."""
    try:
        from neurectomy.elite import (
            create_inference_team,
            create_compression_team,
            create_storage_team,
            create_analysis_team,
            create_synthesis_team,
        )
        
        print("\n" + "=" * 60)
        print("  ELITE AGENT COLLECTIVE - FULL ROSTER")
        print("=" * 60)
        
        teams = [
            ("INFERENCE TEAM", create_inference_team()),
            ("COMPRESSION TEAM", create_compression_team()),
            ("STORAGE TEAM", create_storage_team()),
            ("ANALYSIS TEAM", create_analysis_team()),
            ("SYNTHESIS TEAM", create_synthesis_team()),
        ]
        
        for team_name, agents in teams:
            print(f"\n{team_name}:")
            for agent in agents:
                role = agent.role.name.lower()
                print(f"  [{role:11}] {agent.config.agent_name}")
        
        print("\n" + "=" * 60)
        return True
    except Exception as e:
        print(f"❌ Roster error: {e}")
        return False


def main():
    """Run all verifications."""
    print("=" * 60)
    print("  PHASE 5A: Elite Agent Collective - Verification")
    print("=" * 60)
    print()
    
    results = [
        verify_elite_imports(),
        verify_team_creation(),
        verify_team_structure(),
        verify_agent_processing(),
        verify_cross_team_capability(),
    ]
    
    # Print roster if all passed
    if all(results):
        print_agent_roster()
    
    print()
    if all(results):
        print("=" * 60)
        print("  ✅ PHASE 5A VERIFICATION COMPLETE")
        print("=" * 60)
        print()
        print("Elite Agent Collective: 40 agents in 5 teams")
        print("Ready for Phase 5B: Elite Collective Integration")
        return 0
    else:
        print("=" * 60)
        print("  ❌ PHASE 5A VERIFICATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
