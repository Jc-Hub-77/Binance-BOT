from typing import List, Callable, Dict, TypeVar, Generic
from datetime import datetime
from .types import Signal, Order, ExecutionResult # Assuming types.py is in the same directory

# Generic Event Type
T = TypeVar('T')

class Event:
    def __init__(self):
        self.timestamp: datetime = datetime.utcnow()

class SignalEvent(Event):
    def __init__(self, signal: Signal):
        super().__init__()
        self.signal: Signal = signal

class OrderEvent(Event): # This event might be more granular in a real system e.g. OrderPlacedEvent, OrderFilledEvent, OrderCancelledEvent
    def __init__(self, order: Order, result: ExecutionResult):
        super().__init__()
        self.order: Order = order
        self.result: ExecutionResult = result

# A more generic event for other purposes if needed
class GenericEvent(Event, Generic[T]):
    def __init__(self, payload: T):
        super().__init__()
        self.payload: T = payload

class EventBus:
    def __init__(self):
        self._subscribers: Dict[type, List[Callable]] = {}

    def subscribe(self, event_type: type, handler: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        # print(f"Subscribed {handler.__name__} to {event_type.__name__}") # For debugging

    def unsubscribe(self, event_type: type, handler: Callable):
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                # Handler not found
                pass

    def publish(self, event: Event):
        event_type = type(event)
        # print(f"Publishing event: {event_type.__name__}, {event.__dict__}") # For debugging
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    # print(f"  Calling handler: {handler.__name__}") # For debugging
                    handler(event) # Consider if handlers should be async and awaited
                except Exception as e:
                    # Proper logging should be used here in a real application
                    print(f"Error in event handler {handler.__name__} for event {event_type.__name__}: {e}")

# Example usage (optional, for testing purposes)
if __name__ == '__main__':
    bus = EventBus()

    def my_signal_handler(event: SignalEvent):
        print(f"Signal Handler received: {event.signal.asset}")

    def my_order_handler(event: OrderEvent):
        print(f"Order Handler received for order: {event.order.symbol}, Success: {event.result.success}")

    bus.subscribe(SignalEvent, my_signal_handler)
    bus.subscribe(OrderEvent, my_order_handler)

    test_signal = Signal(asset="BTC/USDT", side=Side.BUY, size=1.0)
    test_order = Order(symbol="ETH/USDT", side=Side.SELL, order_type=OrderType.MARKET, amount=0.5)
    test_result = ExecutionResult(order_id="123", success=True, filled_amount=0.5, average_price=2000.0)

    bus.publish(SignalEvent(test_signal))
    bus.publish(OrderEvent(test_order, test_result))

    bus.unsubscribe(SignalEvent, my_signal_handler)
    bus.publish(SignalEvent(test_signal)) # Should not be handled by my_signal_handler anymore
    bus.publish(OrderEvent(test_order, test_result)) # Should still be handled by my_order_handler
