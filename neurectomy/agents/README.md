# Agent Supervisor

Agent health monitoring and automatic recovery system for the Elite Agent Collective.

## Overview

The Agent Supervisor monitors the health of all agents in the collective and automatically recovers failed agents with configurable retry logic.

## Features

- **Heartbeat Monitoring**: Configurable timeout detection
- **Health Status Tracking**: healthy → degraded → failed → recovering
- **Automatic Recovery**: Configurable recovery attempts with exponential delay
- **Health Reporting**: Detailed health summary and per-agent status
- **40-Agent Support**: Designed to monitor the entire Elite Agent Collective

## Usage

```python
from neurectomy.agents.supervisor import AgentSupervisor

# Initialize supervisor
supervisor = AgentSupervisor(
    heartbeat_timeout=30,  # 30 second timeout
    max_failures=3,        # Max 3 failures before giving up
    recovery_delay=5,      # 5 second delay between attempts
    check_interval=5       # Check every 5 seconds
)

# Register all 40 agents
for i in range(40):
    agent_id = f"agent_{i:02d}"
    supervisor.register_agent(agent_id)

# Start monitoring
await supervisor.start_monitoring()

# Agents report heartbeats
async def agent_loop(agent_id):
    while True:
        # Do agent work
        await perform_agent_tasks()

        # Report heartbeat
        await supervisor.report_heartbeat(agent_id)

        await asyncio.sleep(10)  # Report every 10 seconds

# Get health status
summary = supervisor.get_health_summary()
print(f"Healthy: {summary['healthy']}")
print(f"Degraded: {summary['degraded']}")
print(f"Failed: {summary['failed']}")

# Get failed agents for manual intervention
failed = supervisor.get_failed_agents()
for agent_id, health in failed.items():
    print(f"{agent_id}: {health.failure_count} failures")
```

## Health States

### HEALTHY

- Agent is reporting heartbeats regularly
- No failures detected

### DEGRADED

- Agent missed a heartbeat
- Waiting for recovery or next heartbeat
- Will transition to FAILED if timeout occurs again

### FAILED

- Agent has missed multiple heartbeats
- Automatic recovery in progress
- After max_failures attempts, requires manual intervention

### RECOVERING

- Automatic recovery attempt in progress
- Temporary state during restart

## Configuration

```python
supervisor = AgentSupervisor(
    heartbeat_timeout=30,    # Heartbeat timeout in seconds
    max_failures=3,          # Max failures before giving up
    recovery_delay=5,        # Delay between recovery attempts
    check_interval=5         # Health check interval
)
```

## Methods

### `register_agent(agent_id: str) -> AgentHealth`

Register an agent for monitoring.

### `async report_heartbeat(agent_id: str)`

Report a heartbeat from an agent. Auto-registers if not registered.

### `async start_monitoring()`

Start the monitoring loop.

### `async stop_monitoring()`

Stop the monitoring loop.

### `get_agent_health(agent_id: str) -> AgentHealth`

Get health status for a specific agent.

### `get_all_health() -> Dict[str, AgentHealth]`

Get health status for all agents.

### `get_health_summary() -> Dict[str, int]`

Get summary counts by status.

### `get_failed_agents() -> Dict[str, AgentHealth]`

Get all failed agents.

### `get_degraded_agents() -> Dict[str, AgentHealth]`

Get all degraded agents.

## Integration with Elite Agent Collective

The supervisor integrates with the 40-agent Elite Agent Collective:

```python
# Monitor all 40 agents
@APEX @ARCHITECT @CIPHER @AXIOM @VELOCITY ...  # All 40 agents
supervisor = AgentSupervisor()

for agent_name in AGENT_NAMES:
    supervisor.register_agent(agent_name)

await supervisor.start_monitoring()

# Each agent in its event loop
async def agent_heartbeat(agent_id):
    while running:
        # Do work
        result = await agent_execute_task()

        # Report health
        await supervisor.report_heartbeat(agent_id)

        await asyncio.sleep(heartbeat_interval)
```

## Error Handling

```python
# Monitor for issues
summary = supervisor.get_health_summary()

if summary["failed"] > 0:
    failed_agents = supervisor.get_failed_agents()
    for agent_id, health in failed_agents.items():
        logger.critical(
            f"Agent {agent_id} failed {health.failure_count} times. "
            f"Last error: {health.last_error}"
        )
```

## Best Practices

1. **Regular Heartbeats**: Agents should report heartbeats every N seconds
2. **Timeout Tuning**: Set timeout > max(agent_task_time)
3. **Recovery Delay**: Increase delay for longer startup times
4. **Monitoring**: Check health summary regularly
5. **Alerts**: Alert on degraded/failed agents
6. **Manual Intervention**: Handle max_failures exceeded cases

## Testing

```python
import pytest

@pytest.mark.asyncio
async def test_agent_recovery():
    supervisor = AgentSupervisor(heartbeat_timeout=2)
    supervisor.register_agent("test_agent")

    await supervisor.start_monitoring()
    await asyncio.sleep(3)  # Wait for detection

    health = supervisor.get_agent_health("test_agent")
    assert health.status in [AgentStatus.DEGRADED, AgentStatus.FAILED]

    await supervisor.stop_monitoring()
```

## See Also

- [PHASE-15B: Resilience Patterns](../resilience/)
- [Elite Agent Collective](../../elite/)
- [MNEMONIC Memory System](../../memory/)
