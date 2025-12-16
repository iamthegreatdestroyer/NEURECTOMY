"""
Example: Elite Agent Collective Task Delegation
Demonstrates task routing across the 40-agent collective
"""

import asyncio
from neurectomy.agents.coordination.task_delegation import (
    TaskDelegator,
    AgentCapability,
    Task,
    TaskPriority,
)


# Elite Agent Profiles
ELITE_AGENTS = {
    "APEX": {
        "capabilities": {"code_review", "architecture", "optimization"},
        "max_concurrent": 2,
        "specialization": {
            "code_review": 1.0,
            "architecture": 0.95,
            "optimization": 0.85,
        },
    },
    "CIPHER": {
        "capabilities": {"security_audit", "encryption", "threat_modeling"},
        "max_concurrent": 3,
        "specialization": {
            "security_audit": 1.0,
            "encryption": 0.98,
            "threat_modeling": 0.90,
        },
    },
    "ARCHITECT": {
        "capabilities": {"system_design", "architecture", "scaling"},
        "max_concurrent": 2,
        "specialization": {
            "system_design": 1.0,
            "architecture": 0.98,
            "scaling": 0.92,
        },
    },
    "TENSOR": {
        "capabilities": {"inference", "training", "optimization"},
        "max_concurrent": 4,
        "specialization": {
            "inference": 1.0,
            "training": 0.95,
            "optimization": 0.85,
        },
    },
    "VELOCITY": {
        "capabilities": {"optimization", "profiling", "benchmarking"},
        "max_concurrent": 3,
        "specialization": {
            "optimization": 1.0,
            "profiling": 0.95,
            "benchmarking": 0.90,
        },
    },
    "FORTRESS": {
        "capabilities": {"security_audit", "penetration_testing", "incident_response"},
        "max_concurrent": 2,
        "specialization": {
            "security_audit": 0.95,
            "penetration_testing": 1.0,
            "incident_response": 0.98,
        },
    },
}


async def main():
    """Run elite agent collective delegation example"""
    print("=" * 70)
    print("ELITE AGENT COLLECTIVE - TASK DELEGATION SYSTEM")
    print("=" * 70)
    print()
    
    # Create delegator
    delegator = TaskDelegator()
    
    # Register elite agents
    print("Registering Elite Agents...")
    print("-" * 70)
    
    for agent_id, config in ELITE_AGENTS.items():
        agent = AgentCapability(
            agent_id=agent_id,
            capabilities=set(config["capabilities"]),
            max_concurrent_tasks=config["max_concurrent"],
            specialization_score=config["specialization"],
        )
        delegator.register_agent(agent)
        print(f"  ✓ {agent_id}: {', '.join(config['capabilities'])}")
    
    print()
    print("Initial Collective Status:")
    status = delegator.get_collective_status()
    print(f"  Agents: {status['total_agents']}")
    print(f"  Total Capacity: {status['total_capacity']} concurrent tasks")
    print()
    
    # Create various tasks
    print("=" * 70)
    print("DELEGATING TASKS")
    print("=" * 70)
    print()
    
    tasks = [
        ("code_001", "code_review", TaskPriority.HIGH, "Review authentication module"),
        ("sec_001", "security_audit", TaskPriority.CRITICAL, "Audit API endpoints"),
        ("arch_001", "system_design", TaskPriority.MEDIUM, "Design caching layer"),
        ("inf_001", "inference", TaskPriority.HIGH, "Run model inference"),
        ("opt_001", "optimization", TaskPriority.MEDIUM, "Profile database queries"),
        ("pent_001", "penetration_testing", TaskPriority.HIGH, "Test web vulnerabilities"),
        ("code_002", "code_review", TaskPriority.MEDIUM, "Review storage module"),
        ("inf_002", "inference", TaskPriority.LOW, "Batch inference run"),
    ]
    
    results = []
    for task_id, task_type, priority, description in tasks:
        task = Task(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            payload={"description": description},
            required_capabilities={task_type},
        )
        
        result = await delegator.delegate_task(task)
        results.append(result)
        
        status_icon = "✓" if result.success else "✗"
        agent_info = f"→ {result.agent_id}" if result.agent_id else "→ PENDING"
        priority_str = f"[{priority.name:8}]"
        
        print(f"{status_icon} {task_id:10} {priority_str} {task_type:20} {agent_info}")
        
        if result.score is not None:
            print(f"          Score: {result.score:.3f}")
    
    print()
    print("=" * 70)
    print("AGENT STATUS AFTER DELEGATION")
    print("=" * 70)
    print()
    
    for agent_id in sorted(ELITE_AGENTS.keys()):
        agent_status = delegator.get_agent_status(agent_id)
        if agent_status:
            util_pct = agent_status["utilization"] * 100
            util_bar = "█" * int(util_pct / 5) + "░" * (20 - int(util_pct / 5))
            
            print(f"{agent_id:12} │{util_bar}│ {agent_status['active_tasks']:2}/{agent_status['max_tasks']} "
                  f"({util_pct:5.1f}%)")
    
    print()
    print("=" * 70)
    print("COLLECTIVE STATUS")
    print("=" * 70)
    print()
    
    status = delegator.get_collective_status()
    print(f"Total Agents: {status['total_agents']}")
    print(f"Total Capacity: {status['total_capacity']} concurrent tasks")
    print(f"Active Tasks: {status['active_tasks']}")
    print(f"Pending Tasks: {status['pending_tasks']}")
    print(f"Utilization: {status['utilization']:.1%}")
    
    print()
    print("=" * 70)
    print("DELEGATION STATISTICS")
    print("=" * 70)
    print()
    
    stats = delegator.get_stats()
    print(f"Total Delegations: {stats['total_delegations']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed/Pending: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    if stats['average_score'] > 0:
        print(f"Average Score: {stats['average_score']:.3f}")
    
    # Simulate task completions
    print()
    print("=" * 70)
    print("SIMULATING TASK COMPLETIONS")
    print("=" * 70)
    print()
    
    for result in results[:3]:  # Complete first 3 tasks
        if result.agent_id:
            print(f"Completing {result.task_id} from {result.agent_id}...")
            await delegator.mark_task_complete(
                result.task_id,
                result.agent_id,
                success=True,
            )
    
    print()
    print("After completions:")
    status = delegator.get_collective_status()
    print(f"Active Tasks: {status['active_tasks']}")
    print(f"Pending Tasks: {status['pending_tasks']}")
    print(f"Utilization: {status['utilization']:.1%}")
    
    # Show how pending tasks were reassigned
    if status['pending_tasks'] == 0 and len([r for r in results if not r.success]) > 0:
        print("\n✓ Pending tasks were processed and assigned!")
    
    print()
    print("=" * 70)
    print("AGENT STATUS AFTER COMPLETIONS")
    print("=" * 70)
    print()
    
    for agent_id in sorted(ELITE_AGENTS.keys()):
        agent_status = delegator.get_agent_status(agent_id)
        if agent_status:
            util_pct = agent_status["utilization"] * 100
            util_bar = "█" * int(util_pct / 5) + "░" * (20 - int(util_pct / 5))
            
            print(f"{agent_id:12} │{util_bar}│ {agent_status['active_tasks']:2}/{agent_status['max_tasks']} "
                  f"({util_pct:5.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
