"""Tests for the EventBus."""

from src.event_bus import EventBus


def test_subscribe_and_publish():
    bus = EventBus()
    received = []
    bus.subscribe("test.topic", lambda data: received.append(data))

    bus.publish("test.topic", {"key": "value"})

    assert len(received) == 1
    assert received[0] == {"key": "value"}


def test_multiple_subscribers():
    bus = EventBus()
    results_a = []
    results_b = []
    bus.subscribe("topic", lambda d: results_a.append(d))
    bus.subscribe("topic", lambda d: results_b.append(d))

    bus.publish("topic", 42)

    assert results_a == [42]
    assert results_b == [42]


def test_publish_to_nonexistent_topic():
    bus = EventBus()
    # Should not raise
    bus.publish("no.subscribers", "data")


def test_unsubscribe():
    bus = EventBus()
    received = []
    callback = lambda d: received.append(d)
    bus.subscribe("topic", callback)
    bus.unsubscribe("topic", callback)

    bus.publish("topic", "ignored")
    assert len(received) == 0


def test_subscriber_exception_does_not_break_others():
    bus = EventBus()
    results = []

    def bad_callback(data):
        raise ValueError("boom")

    bus.subscribe("topic", bad_callback)
    bus.subscribe("topic", lambda d: results.append(d))

    bus.publish("topic", "hello")
    assert results == ["hello"]
