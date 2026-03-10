from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, topic: str, callback: Callable) -> None:
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        try:
            self._subscribers[topic].remove(callback)
        except ValueError:
            pass

    def publish(self, topic: str, data: Any = None) -> None:
        for callback in self._subscribers.get(topic, []):
            try:
                callback(data)
            except Exception:
                logger.exception("Error in subscriber for topic '%s'", topic)
