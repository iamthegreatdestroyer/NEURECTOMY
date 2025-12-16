"""
Event Bus: Pub/Sub Event System
Publish-Subscribe event system for asynchronous communication between components
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional, Set, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """System event types"""
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_SKIPPED = "task.skipped"
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    AGENT_HEARTBEAT = "agent.heartbeat"
    AGENT_FAILED = "agent.failed"
    AGENT_RECOVERED = "agent.recovered"
    AGENT_REGISTERED = "agent.registered"
    COMPRESSION_STARTED = "compression.started"
    COMPRESSION_COMPLETED = "compression.completed"
    STORAGE_UPLOADED = "storage.uploaded"
    STORAGE_RETRIEVED = "storage.retrieved"
    STORAGE_DELETED = "storage.deleted"
    ERROR_OCCURRED = "error.occurred"
    WARNING_ISSUED = "warning.issued"
    INFO_LOGGED = "info.logged"


@dataclass
class Event:
    """Event definition"""
    event_type: EventType
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    
    def __post_init__(self):
        """Validate event on creation"""
        if not self.source:
            raise ValueError("source is required")
    
    def matches(self, pattern: str) -> bool:
        """
        Check if event matches a pattern
        
        Patterns:
        - "event.type" - exact match
        - "event.*" - wildcard match
        - "*" - match all
        """
        event_str = self.event_type.value
        
        if pattern == "*":
            return True
        
        if pattern == event_str:
            return True
        
        if pattern.endswith("*"):
            prefix = pattern[:-1]  # Remove *
            return event_str.startswith(prefix)
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


@dataclass
class EventSubscription:
    """Event subscription"""
    subscription_id: str
    pattern: str
    handler: Callable
    filter_fn: Optional[Callable[[Event], bool]] = None
    is_async: bool = False
    
    async def can_handle(self, event: Event) -> bool:
        """Check if subscription can handle event"""
        # Pattern matching
        if not event.matches(self.pattern):
            return False
        
        # Custom filter
        if self.filter_fn:
            if asyncio.iscoroutinefunction(self.filter_fn):
                return await self.filter_fn(event)
            else:
                return self.filter_fn(event)
        
        return True


class EventBus:
    """
    Publish-Subscribe Event Bus
    
    Features:
    - Event publishing and subscription
    - Pattern-based event filtering
    - Custom event filters
    - Async event handlers
    - Event history tracking
    - Multiple subscribers per event
    
    Example:
        >>> bus = EventBus()
        >>> async def on_task_complete(event):
        ...     print(f"Task completed: {event.data}")
        >>> bus.subscribe("task.completed", on_task_complete)
        >>> event = Event(EventType.TASK_COMPLETED, "workflow", {"task_id": "t1"})
        >>> await bus.emit(event)
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize event bus
        
        Args:
            max_history: Maximum events to keep in history (default: 1000)
        """
        self.subscriptions: Dict[str, List[EventSubscription]] = {}
        self.event_history: List[Event] = []
        self.max_history = max_history
        self.event_counts: Dict[str, int] = {}
        self.active_subscriptions: Set[str] = set()
    
    def subscribe(
        self,
        pattern: str,
        handler: Callable,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """
        Subscribe to events matching a pattern
        
        Args:
            pattern: Event pattern (e.g., "task.*", "task.completed", "*")
            handler: Async callable that handles the event
            filter_fn: Optional filter function for additional filtering
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid4())
        is_async = asyncio.iscoroutinefunction(handler)
        
        subscription = EventSubscription(
            subscription_id=subscription_id,
            pattern=pattern,
            handler=handler,
            filter_fn=filter_fn,
            is_async=is_async,
        )
        
        if pattern not in self.subscriptions:
            self.subscriptions[pattern] = []
        
        self.subscriptions[pattern].append(subscription)
        self.active_subscriptions.add(subscription_id)
        
        logger.info(f"Subscribed to pattern '{pattern}' with ID {subscription_id}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events
        
        Args:
            subscription_id: Subscription ID to remove
            
        Returns:
            True if found and removed
        """
        for pattern, subs in self.subscriptions.items():
            for i, sub in enumerate(subs):
                if sub.subscription_id == subscription_id:
                    subs.pop(i)
                    self.active_subscriptions.discard(subscription_id)
                    logger.info(f"Unsubscribed: {subscription_id}")
                    return True
        
        return False
    
    async def emit(self, event: Event):
        """
        Emit an event to all subscribers
        
        Args:
            event: Event to emit
        """
        logger.debug(f"Emitting event: {event.event_type.value} from {event.source}")
        
        # Add to history
        self._add_to_history(event)
        
        # Track event count
        event_key = event.event_type.value
        self.event_counts[event_key] = self.event_counts.get(event_key, 0) + 1
        
        # Find matching subscriptions
        matching_subs = []
        
        for pattern, subs in self.subscriptions.items():
            for sub in subs:
                if await sub.can_handle(event):
                    matching_subs.append(sub)
        
        if not matching_subs:
            logger.debug(f"No subscribers for event: {event.event_type.value}")
            return
        
        logger.debug(f"Found {len(matching_subs)} subscriber(s)")
        
        # Call all matching handlers
        tasks = []
        for sub in matching_subs:
            try:
                if sub.is_async:
                    tasks.append(sub.handler(event))
                else:
                    # Wrap sync handler in async
                    tasks.append(asyncio.create_task(
                        asyncio.to_thread(sub.handler, event)
                    ))
            except Exception as e:
                logger.error(f"Error calling handler: {e}", exc_info=True)
        
        # Wait for all handlers to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Handler {i} raised exception: {result}")
    
    async def emit_many(self, events: List[Event]):
        """
        Emit multiple events
        
        Args:
            events: List of events to emit
        """
        for event in events:
            await self.emit(event)
    
    def _add_to_history(self, event: Event):
        """Add event to history with max size limit"""
        self.event_history.append(event)
        
        # Trim history if needed
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
    
    def get_history(
        self,
        pattern: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """
        Get event history with optional filtering
        
        Args:
            pattern: Event type pattern (e.g., "task.*")
            source: Filter by event source
            limit: Maximum events to return
            
        Returns:
            List of events matching criteria
        """
        filtered = self.event_history
        
        if pattern:
            filtered = [e for e in filtered if e.matches(pattern)]
        
        if source:
            filtered = [e for e in filtered if e.source == source]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def get_subscription_count(self, pattern: Optional[str] = None) -> int:
        """
        Get number of active subscriptions
        
        Args:
            pattern: Optional pattern to filter subscriptions
            
        Returns:
            Number of subscriptions
        """
        if pattern:
            return len(self.subscriptions.get(pattern, []))
        
        return sum(len(subs) for subs in self.subscriptions.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_events_emitted": sum(self.event_counts.values()),
            "event_counts": self.event_counts.copy(),
            "total_subscriptions": len(self.active_subscriptions),
            "patterns": list(self.subscriptions.keys()),
            "history_size": len(self.event_history),
            "max_history": self.max_history,
        }
    
    def clear_history(self):
        """Clear event history"""
        self.event_history.clear()
        logger.info("Event history cleared")
    
    def reset_stats(self):
        """Reset statistics"""
        self.event_counts.clear()
        logger.info("Statistics reset")


class EventBusGlobal:
    """Global event bus singleton"""
    _instance: Optional[EventBus] = None
    
    @classmethod
    def get_instance(cls) -> EventBus:
        """Get or create global event bus"""
        if cls._instance is None:
            cls._instance = EventBus()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset global event bus (for testing)"""
        cls._instance = None
