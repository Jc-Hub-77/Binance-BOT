from typing import List
from trading_bot.core.types import Signal
from trading_bot.core.events import EventBus # Added for consistency, though not used in this simple version
from .base import Strategy
# from loguru import logger # Optional: for logging

class DetectorDrivenStrategy(Strategy):
    """
    A simple strategy that primarily acts based on signals received from an external detector.
    It can filter signals based on a minimum confidence level.
    """
    def __init__(self, event_bus: EventBus = None, min_confidence: float = 0.0, **kwargs):
        """
        Initializes the DetectorDrivenStrategy.

        Args:
            event_bus: An optional EventBus instance.
            min_confidence: The minimum confidence score a signal must have to be processed.
                            Signals with confidence below this threshold will be ignored.
                            Defaults to 0.0 (all signals pass).
            **kwargs: Additional parameters from configuration.
        """
        super().__init__(event_bus=event_bus, **kwargs)
        self.min_confidence = float(min_confidence) # Ensure it's a float
        # logger.info(f"DetectorDrivenStrategy initialized with min_confidence: {self.min_confidence}")

    def process_signal(self, signal: Signal) -> List[Signal]:
        """
        Processes an incoming signal from the detector.
        If the signal's confidence meets or exceeds `min_confidence`, it's returned for execution.
        Otherwise, an empty list is returned, effectively ignoring the signal.

        Args:
            signal: The incoming Signal object.

        Returns:
            A list containing the signal if its confidence is sufficient, otherwise an empty list.
        """
        if signal.confidence >= self.min_confidence:
            # logger.debug(f"Processing signal: {signal.asset} {signal.side}, confidence {signal.confidence} >= {self.min_confidence}")
            return [signal]
        else:
            # logger.debug(f"Ignoring signal: {signal.asset} {signal.side}, confidence {signal.confidence} < {self.min_confidence}")
            return []

    def analyze_market(self, market_data: dict) -> List[Signal]:
        """
        This strategy does not perform its own market analysis to generate signals.
        It relies solely on external detector signals.

        Args:
            market_data: Market data (unused by this strategy).

        Returns:
            An empty list, as this strategy does not generate signals from market analysis.
        """
        # logger.debug("DetectorDrivenStrategy.analyze_market called, but no internal analysis performed.")
        return []

if __name__ == '__main__':
    from trading_bot.core.types import Side, SignalSource

    # Example Usage
    strategy_conf_pass = DetectorDrivenStrategy(min_confidence=0.7)
    strategy_conf_block = DetectorDrivenStrategy(min_confidence=0.9)
    strategy_no_conf = DetectorDrivenStrategy() # min_confidence = 0.0

    signal_high_conf = Signal(asset="BTC/USDT", side=Side.BUY, size=1.0, confidence=0.8, source=SignalSource.DETECTOR)
    signal_low_conf = Signal(asset="ETH/USDT", side=Side.SELL, size=0.5, confidence=0.6, source=SignalSource.DETECTOR)

    # Test case 1: Signal confidence (0.8) >= min_confidence (0.7) -> Should pass
    results1 = strategy_conf_pass.process_signal(signal_high_conf)
    assert len(results1) == 1
    assert results1[0] == signal_high_conf
    print(f"Test 1 Passed: {results1[0]}")

    # Test case 2: Signal confidence (0.6) < min_confidence (0.7) -> Should be blocked
    results2 = strategy_conf_pass.process_signal(signal_low_conf)
    assert len(results2) == 0
    print(f"Test 2 Passed: Signal blocked as expected.")

    # Test case 3: Signal confidence (0.8) < min_confidence (0.9) -> Should be blocked
    results3 = strategy_conf_block.process_signal(signal_high_conf)
    assert len(results3) == 0
    print(f"Test 3 Passed: Signal blocked as expected.")

    # Test case 4: No min_confidence set (defaults to 0.0) -> Should pass all
    results4_high = strategy_no_conf.process_signal(signal_high_conf)
    results4_low = strategy_no_conf.process_signal(signal_low_conf)
    assert len(results4_high) == 1
    assert len(results4_low) == 1
    print(f"Test 4 Passed: All signals pass with default min_confidence.")

    # Test analyze_market (should always be empty)
    market_analysis_signals = strategy_conf_pass.analyze_market({"BTC/USDT": {"price": 50000}})
    assert len(market_analysis_signals) == 0
    print(f"Test 5 Passed: analyze_market returned no signals as expected.")

    print("\nAll DetectorDrivenStrategy tests completed.")
