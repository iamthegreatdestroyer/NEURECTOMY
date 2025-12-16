"""
Tests for Event Bus
"""

import pytest
import asyncio
from neurectomy.orchestration.event_bus import (
    EventBus,
    Event,
    EventType,
    EventSubscription,
    EventBusGlobal,
)


@pytest.fixture
def event_bus():
    """Create event bus for tests"""
    return EventBus()


@pytest.fixture
def sample_event():
    """Create sample event"""
    return Event(
        event_type=EventType.TASK_COMPLETED,
        source="test_workflow",
        data={"task_id": "task_1", "result": "success"},
    )


class TestEventType:
    """Test EventType enum"""
    
    def test_event_types_exist(self):
        """Test that event types are defined"""
        assert EventType.TASK_STARTED
        assert EventType.TASK_COMPLETED
        assert EventType.TASK_FAILED
        assert EventType.WORKFLOW_STARTED
        assert EventType.AGENT_HEARTBEAT


class TestEvent:
    """Test Event dataclass"""
    
    def test_event_creation(self):
        """Test event creation"""
        event = Event(
            event_type=EventType.TASK_COMPLETED,
            source="test",
            data={"key": "value"},
        )
        
        assert event.event_type == EventType.TASK_COMPLETED
        assert event.source == "test"
        assert event.data == {"key": "value"}
    
    def test_event_missing_source(self):
        """Test event requires source"""
        with pytest.raises(ValueError):
            Event(
                event_type=EventType.TASK_COMPLETED,
                source="",
                data={},
            )
    
    def test_event_has_id(self):
        """Test event gets unique ID"""
        event1 = Event(EventType.TASK_COMPLETED, "test")
        event2 = Event(EventType.TASK_COMPLETED, "test")
        
        assert event1.event_id
        assert event2.event_id
        assert event1.event_id != event2.event_id
    
    def test_event_exact_match(self):
        """Test exact pattern matching"""
        event = Event(EventType.TASK_COMPLETED, "test")
        
        assert event.matches("task.completed")
        assert not event.matches("task.failed")
    
    def test_event_wildcard_match(self):
        """Test wildcard pattern matching"""
        event = Event(EventType.TASK_COMPLETED, "test")
        
        assert event.matches("task.*")
        assert event.matches("*")
        assert not event.matches("workflow.*")
    
    def test_event_to_dict(self):
        """Test event serialization"""
        event = Event(
            event_type=EventType.TASK_COMPLETED,
            source="test",
            data={"result": "success"},
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["event_type"] == "task.completed"
        assert event_dict["source"] == "test"
        assert event_dict["data"] == {"result": "success"}
        assert "timestamp" in event_dict
        assert "event_id" in event_dict


class TestEventSubscription:
    """Test EventSubscription"""
    
    @pytest.mark.asyncio
    async def test_subscription_creation(self):
        """Test subscription creation"""
        async def handler(event):
            pass
        
        sub = EventSubscription(
            subscription_id="sub_1",
            pattern="task.*",
            handler=handler,
        )
        
        assert sub.subscription_id == "sub_1"
        assert sub.pattern == "task.*"
        assert sub.is_async
    
    @pytest.mark.asyncio
    async def test_subscription_can_handle(self):
        """Test subscription matching"""
        async def handler(event):
            pass
        
        sub = EventSubscription(
            subscription_id="sub_1",
            pattern="task.*",
            handler=handler,
        )
        
        event = Event(EventType.TASK_COMPLETED, "test")
        assert await sub.can_handle(event)
        
        event2 = Event(EventType.WORKFLOW_STARTED, "test")
        assert not await sub.can_handle(event2)
    
    @pytest.mark.asyncio
    async def test_subscription_filter(self):
        """Test subscription with filter function"""
        async def handler(event):
            pass
        
        def filter_fn(event):
            return event.data.get("task_id") == "task_1"
        
        sub = EventSubscription(
            subscription_id="sub_1",
            pattern="task.*",
            handler=handler,
            filter_fn=filter_fn,
        )
        
        event1 = Event(EventType.TASK_COMPLETED, "test", {"task_id": "task_1"})
        assert await sub.can_handle(event1)
        
        event2 = Event(EventType.TASK_COMPLETED, "test", {"task_id": "task_2"})
        assert not await sub.can_handle(event2)


class TestEventBus:
    """Test EventBus"""
    
    def test_bus_creation(self):
        """Test event bus creation"""
        bus = EventBus()
        assert bus.max_history == 1000
        assert len(bus.subscriptions) == 0
    
    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self, event_bus):
        """Test basic subscribe and emit"""
        events_received = []
        
        async def handler(event):
            events_received.append(event)
        
        event_bus.subscribe("task.completed", handler)
        
        event = Event(EventType.TASK_COMPLETED, "test")
        await event_bus.emit(event)
        
        assert len(events_received) == 1
        assert events_received[0].event_type == EventType.TASK_COMPLETED
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers"""
        events1 = []
        events2 = []
        
        async def handler1(event):
            events1.append(event)
        
        async def handler2(event):
            events2.append(event)
        
        event_bus.subscribe("task.*", handler1)
        event_bus.subscribe("task.*", handler2)
        
        event = Event(EventType.TASK_COMPLETED, "test")
        await event_bus.emit(event)
        
        assert len(events1) == 1
        assert len(events2) == 1
    
    @pytest.mark.asyncio
    async def test_pattern_filtering(self, event_bus):
        """Test pattern-based filtering"""
        events = []
        
        async def handler(event):
            events.append(event)
        
        event_bus.subscribe("task.*", handler)
        
        # This should match
        await event_bus.emit(Event(EventType.TASK_COMPLETED, "test"))
        # This should not match
        await event_bus.emit(Event(EventType.WORKFLOW_STARTED, "test"))
        
        assert len(events) == 1
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing"""
        events = []
        
        async def handler(event):
            events.append(event)
        
        sub_id = event_bus.subscribe("task.*", handler)
        
        await event_bus.emit(Event(EventType.TASK_COMPLETED, "test"))
        assert len(events) == 1
        
        event_bus.unsubscribe(sub_id)
        
        await event_bus.emit(Event(EventType.TASK_COMPLETED, "test"))
        assert len(events) == 1  # Should not increase
    
    @pytest.mark.asyncio
    async def test_event_history(self, event_bus):
        """Test event history tracking"""
        event1 = Event(EventType.TASK_COMPLETED, "test")
        event2 = Event(EventType.TASK_FAILED, "test")
        
        await event_bus.emit(event1)
        await event_bus.emit(event2)
        
        history = event_bus.get_history()
        assert len(history) == 2
    
    @pytest.mark.asyncio
    async def test_history_filtering(self, event_bus):
        """Test event history filtering"""
        await event_bus.emit(Event(EventType.TASK_COMPLETED, "workflow1"))
        await event_bus.emit(Event(EventType.TASK_COMPLETED, "workflow2"))
        await event_bus.emit(Event(EventType.TASK_FAILED, "workflow1"))
        
        # Filter by pattern
        completed = event_bus.get_history(pattern="task.completed")
        assert len(completed) == 2
        
        # Filter by source
        workflow1_events = event_bus.get_history(source="workflow1")
        assert len(workflow1_events) == 2
    
    @pytest.mark.asyncio
    async def test_sync_handler(self, event_bus):
        """Test sync handler support"""
        events = []
        
        def sync_handler(event):
            events.append(event)
        
        event_bus.subscribe("task.*", sync_handler)
        
        await event_bus.emit(Event(EventType.TASK_COMPLETED, "test"))
        
        # Give async thread time to complete
        await asyncio.sleep(0.1)
        assert len(events) == 1
    
    def test_subscription_count(self, event_bus):
        """Test subscription counting"""
        async def handler(event):
            pass
        
        event_bus.subscribe("task.*", handler)
        event_bus.subscribe("task.*", handler)
        event_bus.subscribe("workflow.*", handler)
        
        assert event_bus.get_subscription_count("task.*") == 2
        assert event_bus.get_subscription_count("workflow.*") == 1
        assert event_bus.get_subscription_count() == 3
    
    @pytest.mark.asyncio
    async def test_stats(self, event_bus):
        """Test statistics collection"""
        async def handler(event):
            pass
        
        event_bus.subscribe("task.*", handler)
        
        await event_bus.emit(Event(EventType.TASK_COMPLETED, "test"))
        await event_bus.emit(Event(EventType.TASK_COMPLETED, "test"))
        await event_bus.emit(Event(EventType.TASK_FAILED, "test"))
        
        stats = event_bus.get_stats()
        
        assert stats["total_events_emitted"] == 3
        assert stats["total_subscriptions"] == 1
    
    @pytest.mark.asyncio
    async def test_max_history(self):
        """Test history size limit"""
        bus = EventBus(max_history=3)
        
        async def handler(event):
            pass
        
        bus.subscribe("*", handler)
        
        for i in range(5):
            await bus.emit(Event(EventType.TASK_COMPLETED, f"test_{i}"))
        
        assert len(bus.event_history) == 3
    
    def test_clear_history(self, event_bus):
        """Test clearing history"""
        event_bus.event_history.append(
            Event(EventType.TASK_COMPLETED, "test")
        )
        
        assert len(event_bus.event_history) > 0
        event_bus.clear_history()
        assert len(event_bus.event_history) == 0
    
    def test_reset_stats(self, event_bus):
        """Test resetting statistics"""
        event_bus.event_counts["task.completed"] = 5
        
        assert event_bus.event_counts["task.completed"] == 5
        event_bus.reset_stats()
        assert len(event_bus.event_counts) == 0


class TestEventBusGlobal:
    """Test global event bus"""
    
    def test_singleton(self):
        """Test event bus is singleton"""
        bus1 = EventBusGlobal.get_instance()
        bus2 = EventBusGlobal.get_instance()
        
        assert bus1 is bus2
    
    def test_reset_singleton(self):
        """Test resetting singleton"""
        bus1 = EventBusGlobal.get_instance()
        original_id = id(bus1)
        
        EventBusGlobal.reset()
        
        bus2 = EventBusGlobal.get_instance()
        assert id(bus2) != original_id
