from abc import ABC, abstractmethod
from typing import List, Any, Dict # Added Any, Dict for market_data
from trading_bot.core.types import Signal
from trading_bot.core.events import EventBus, SignalEvent

class Strategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    Strategies are responsible for processing incoming signals (from detectors or other sources)
    and potentially generating their own signals based on market analysis.
    """
    def __init__(self, event_bus: EventBus = None, **kwargs): # Added **kwargs to consume unused params from config
        """
        Initializes the Strategy.
        Args:
            event_bus: An optional EventBus instance to publish generated signals.
            **kwargs: Allows for additional parameters from configuration to be accepted.
        """
        self.event_bus = event_bus
        # You can store other common parameters from kwargs if needed, e.g.
        # self.strategy_name = kwargs.get('name', self.__class__.__name__)
        # logger.debug(f"Strategy '{self.strategy_name}' initialized with kwargs: {kwargs}")


    @abstractmethod
    def process_signal(self, signal: Signal) -> List[Signal]:
        """
        Process an incoming trading signal.
        This method should define how the strategy reacts to external signals.
        It can choose to ignore the signal, modify it, or generate new signals based on it.

        Args:
            signal: The incoming Signal object.

        Returns:
            A list of Signal objects to be acted upon. This list can be empty.
        """
        pass

    @abstractmethod
    def analyze_market(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Perform market analysis based on provided market data and generate signals.
        This method allows the strategy to be proactive and generate signals
        even without an external trigger signal.

        Args:
            market_data: A dictionary containing market data. The structure of this dictionary
                         can vary (e.g., {'BTC/USDT': {'klines': [...], 'orderbook': ...}}).

        Returns:
            A list of Signal objects generated from the analysis. This list can be empty.
        """
        pass

    def emit_signals(self, signals: List[Signal]):
        """
        Helper method to publish a list of signals to the event bus, if available.

        Args:
            signals: A list of Signal objects to be published.
        """
        if self.event_bus and signals:
            for signal in signals:
                if not isinstance(signal, Signal):
                    # Potentially log a warning or raise an error
                    # print(f"Warning: Attempting to emit non-Signal object: {signal}")
                    continue # Or handle more gracefully
                self.event_bus.publish(SignalEvent(signal))
                # print(f"Emitted signal: {signal.asset} {signal.side} from {self.__class__.__name__}") # For debugging
        elif not self.event_bus:
            # Log that event bus is not available if signals were intended to be emitted
            # print(f"Warning: Event bus not configured for {self.__class__.__name__}, cannot emit signals.")
            pass
