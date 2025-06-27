import pytest # Pytest for test running and fixtures
from trading_bot.core.types import Signal, Side, SignalSource, OrderType # Core data types
from trading_bot.core.events import EventBus # For strategies that use event bus

# Strategies to test
from trading_bot.modules.strategy.base import Strategy as BaseStrategy # Base class for typing if needed
from trading_bot.modules.strategy.detector_driven import DetectorDrivenStrategy
from trading_bot.modules.strategy.combined import CombinedStrategy

# --- Test DetectorDrivenStrategy ---
class TestDetectorDrivenStrategy:
    def test_process_signal_above_confidence_threshold(self):
        """Signal with confidence >= min_confidence should pass."""
        strategy = DetectorDrivenStrategy(min_confidence=0.7)
        signal = Signal(asset="BTC/USDT", side=Side.BUY, size=1.0, confidence=0.8, source=SignalSource.DETECTOR)

        processed_signals = strategy.process_signal(signal)

        assert len(processed_signals) == 1
        assert processed_signals[0] == signal
        print("TestDetectorDrivenStrategy.test_process_signal_above_confidence_threshold PASSED")

    def test_process_signal_below_confidence_threshold(self):
        """Signal with confidence < min_confidence should be filtered out."""
        strategy = DetectorDrivenStrategy(min_confidence=0.7)
        signal = Signal(asset="ETH/USDT", side=Side.SELL, size=0.5, confidence=0.6, source=SignalSource.DETECTOR)

        processed_signals = strategy.process_signal(signal)

        assert len(processed_signals) == 0
        print("TestDetectorDrivenStrategy.test_process_signal_below_confidence_threshold PASSED")

    def test_process_signal_at_confidence_threshold(self):
        """Signal with confidence == min_confidence should pass."""
        strategy = DetectorDrivenStrategy(min_confidence=0.7)
        signal = Signal(asset="LTC/USDT", side=Side.BUY, size=2.0, confidence=0.7, source=SignalSource.DETECTOR)

        processed_signals = strategy.process_signal(signal)

        assert len(processed_signals) == 1
        assert processed_signals[0] == signal
        print("TestDetectorDrivenStrategy.test_process_signal_at_confidence_threshold PASSED")

    def test_process_signal_default_confidence(self):
        """Signal should pass if min_confidence is default (0.0)."""
        strategy = DetectorDrivenStrategy() # min_confidence defaults to 0.0
        signal = Signal(asset="XRP/USDT", side=Side.SELL, size=100.0, confidence=0.1, source=SignalSource.DETECTOR) # Low confidence

        processed_signals = strategy.process_signal(signal)

        assert len(processed_signals) == 1
        assert processed_signals[0] == signal
        print("TestDetectorDrivenStrategy.test_process_signal_default_confidence PASSED")

    def test_analyze_market_returns_empty_list(self):
        """analyze_market should always return an empty list for DetectorDrivenStrategy."""
        strategy = DetectorDrivenStrategy()
        market_data = {"BTC/USDT": {"price": 50000}} # Example market data

        analyzed_signals = strategy.analyze_market(market_data)

        assert len(analyzed_signals) == 0
        print("TestDetectorDrivenStrategy.test_analyze_market_returns_empty_list PASSED")


# --- Test CombinedStrategy ---
# Helper for mock market data
def get_mock_market_data(prices_btc=None, prices_eth=None):
    data = {}
    if prices_btc: data["BTC/USDT"] = {"prices": prices_btc}
    if prices_eth: data["ETH/USDT"] = {"prices": prices_eth}
    return data

# Prices for MA crossover tests (50 data points needed for CombinedStrategy's _analyze_single_asset)
prices_bullish_trend = list(range(50, 0, -1)) + list(range(1, 51)) # V-shape, ends bullish
prices_bearish_trend = list(range(0, 50)) + list(range(49, -1, -1)) # A-shape, ends bearish
prices_flat_trend = [50.0] * 100 # Flat prices

class TestCombinedStrategy:
    @pytest.fixture
    def event_bus_mock(self):
        return EventBus() # Real EventBus can be used, or a mock if complex interactions are tested

    def test_initialization_signal_buffer(self, event_bus_mock):
        """Test item 2 from 'things to check.txt': signal_buffer initialization."""
        strategy = CombinedStrategy(event_bus=event_bus_mock)
        assert hasattr(strategy, 'signal_buffer')
        assert isinstance(strategy.signal_buffer, dict)
        print("TestCombinedStrategy.test_initialization_signal_buffer PASSED")

    def test_process_signal_no_confirmation_required(self, event_bus_mock):
        """If require_confirmation is False, detector signal should pass through."""
        strategy = CombinedStrategy(event_bus=event_bus_mock, require_confirmation=False)
        detector_signal = Signal("BTC/USDT", Side.BUY, 1.0, 0.8, SignalSource.DETECTOR)

        processed = strategy.process_signal(detector_signal)

        assert len(processed) == 1
        assert processed[0] == detector_signal
        assert "BTC/USDT" in strategy.signal_buffer # Signal should still be buffered
        print("TestCombinedStrategy.test_process_signal_no_confirmation_required PASSED")

    def test_process_signal_confirmation_required(self, event_bus_mock):
        """If require_confirmation is True, detector signal is buffered, not immediately passed."""
        strategy = CombinedStrategy(event_bus=event_bus_mock, require_confirmation=True)
        detector_signal = Signal("BTC/USDT", Side.SELL, 1.0, 0.8, SignalSource.DETECTOR)

        processed = strategy.process_signal(detector_signal)

        assert len(processed) == 0 # Not passed immediately
        assert "BTC/USDT" in strategy.signal_buffer
        assert strategy.signal_buffer["BTC/USDT"] == detector_signal
        print("TestCombinedStrategy.test_process_signal_confirmation_required PASSED")

    def test_analyze_market_internal_signal_no_confirmation(self, event_bus_mock):
        """Internal analysis generates signal, no confirmation needed for internal signals themselves."""
        strategy = CombinedStrategy(event_bus=event_bus_mock, require_confirmation=False) # Doesn't affect purely internal
        market_data = get_mock_market_data(prices_btc=prices_bullish_trend)

        analyzed = strategy.analyze_market(market_data)

        assert len(analyzed) == 1
        assert analyzed[0].asset == "BTC/USDT"
        assert analyzed[0].side == Side.BUY
        assert analyzed[0].source == SignalSource.INTERNAL
        print("TestCombinedStrategy.test_analyze_market_internal_signal_no_confirmation PASSED")

    def test_analyze_market_confirmation_match(self, event_bus_mock):
        """Detector signal + matching internal signal = COMBINED signal."""
        strategy = CombinedStrategy(event_bus=event_bus_mock, require_confirmation=True)
        detector_signal = Signal("BTC/USDT", Side.BUY, 1.0, 0.9, SignalSource.DETECTOR)
        strategy.process_signal(detector_signal) # Buffer it

        market_data = get_mock_market_data(prices_btc=prices_bullish_trend) # Internal will also say BUY
        analyzed = strategy.analyze_market(market_data)

        assert len(analyzed) == 1
        combined_signal = analyzed[0]
        assert combined_signal.asset == "BTC/USDT"
        assert combined_signal.side == Side.BUY
        assert combined_signal.source == SignalSource.COMBINED
        assert "BTC/USDT" not in strategy.signal_buffer # Should be consumed after confirmation
        print("TestCombinedStrategy.test_analyze_market_confirmation_match PASSED")

    def test_analyze_market_confirmation_conflict(self, event_bus_mock):
        """Detector signal + conflicting internal signal = No COMBINED signal."""
        strategy = CombinedStrategy(event_bus=event_bus_mock, require_confirmation=True)
        detector_signal = Signal("BTC/USDT", Side.SELL, 1.0, 0.9, SignalSource.DETECTOR) # Detector: SELL
        strategy.process_signal(detector_signal)

        market_data = get_mock_market_data(prices_btc=prices_bullish_trend) # Internal: BUY
        analyzed = strategy.analyze_market(market_data)

        assert len(analyzed) == 0
        assert "BTC/USDT" in strategy.signal_buffer # Not consumed due to conflict
        print("TestCombinedStrategy.test_analyze_market_confirmation_conflict PASSED")

    def test_analyze_market_no_detector_signal_for_confirmation(self, event_bus_mock):
        """Internal signal, but no detector signal was buffered for confirmation."""
        strategy = CombinedStrategy(event_bus=event_bus_mock, require_confirmation=True)
        # No detector signal processed for BTC/USDT
        market_data = get_mock_market_data(prices_btc=prices_bullish_trend) # Internal: BUY for BTC

        analyzed = strategy.analyze_market(market_data)

        # Current logic: if require_confirmation is True, it needs a buffered detector signal to form a COMBINED one.
        # Purely internal signals are not emitted in this mode by analyze_market.
        assert len(analyzed) == 0
        print("TestCombinedStrategy.test_analyze_market_no_detector_signal_for_confirmation PASSED")

    def test_analyze_market_flat_internal_analysis(self, event_bus_mock):
        """Detector signal buffered, but internal analysis is flat (no signal)."""
        strategy = CombinedStrategy(event_bus=event_bus_mock, require_confirmation=True)
        detector_signal = Signal("BTC/USDT", Side.BUY, 1.0, 0.9, SignalSource.DETECTOR)
        strategy.process_signal(detector_signal)

        market_data = get_mock_market_data(prices_btc=prices_flat_trend) # Internal: No signal
        analyzed = strategy.analyze_market(market_data)

        assert len(analyzed) == 0
        assert "BTC/USDT" in strategy.signal_buffer # Not consumed
        print("TestCombinedStrategy.test_analyze_market_flat_internal_analysis PASSED")

    @pytest.mark.asyncio # If using asyncio.sleep in strategy, though current one doesn't
    async def test_signal_buffer_timeout(self, event_bus_mock):
        """Buffered detector signal should expire after timeout."""
        # Using a very short timeout for testing.
        # The strategy's _cleanup_signal_buffer is called at start of process_signal and analyze_market
        strategy = CombinedStrategy(event_bus=event_bus_mock, require_confirmation=True, buffer_timeout_minutes=0.0001) # Approx 6ms

        detector_signal = Signal("BTC/USDT", Side.BUY, 1.0, 0.9, SignalSource.DETECTOR)
        strategy.process_signal(detector_signal) # Buffer it
        assert "BTC/USDT" in strategy.signal_buffer

        await asyncio.sleep(0.01) # Wait longer than timeout

        market_data = get_mock_market_data(prices_btc=prices_bullish_trend) # Internal would confirm BUY
        # Calling analyze_market will first trigger _cleanup_signal_buffer
        analyzed = strategy.analyze_market(market_data)

        assert "BTC/USDT" not in strategy.signal_buffer # Should have been cleaned up
        assert len(analyzed) == 0 # No COMBINED signal as detector signal expired
        print("TestCombinedStrategy.test_signal_buffer_timeout PASSED")

# To run these tests using pytest:
# Ensure pytest and pytest-asyncio (for async tests) are installed.
# Navigate to the directory containing this file (or its parent) and run:
# pytest
# or specifically:
# pytest path/to/test_strategy.py

# This will execute all functions starting with "test_" in classes starting with "Test".
if __name__ == "__main__":
    # This block allows running tests with `python path/to/test_strategy.py`
    # For more comprehensive test discovery and reporting, use `pytest`.

    # Running DetectorDrivenStrategy tests
    print("--- Running DetectorDrivenStrategy Tests ---")
    test_detector = TestDetectorDrivenStrategy()
    test_detector.test_process_signal_above_confidence_threshold()
    test_detector.test_process_signal_below_confidence_threshold()
    test_detector.test_process_signal_at_confidence_threshold()
    test_detector.test_process_signal_default_confidence()
    test_detector.test_analyze_market_returns_empty_list()
    print("--- DetectorDrivenStrategy Tests Completed ---\n")

    # Running CombinedStrategy tests
    # Note: Async tests like test_signal_buffer_timeout need an event loop.
    # Pytest handles this with pytest-asyncio. Running directly needs asyncio.run().
    print("--- Running CombinedStrategy Tests ---")
    test_combined = TestCombinedStrategy()
    bus = EventBus() # Create a bus for tests run this way

    test_combined.test_initialization_signal_buffer(bus)
    test_combined.test_process_signal_no_confirmation_required(bus)
    test_combined.test_process_signal_confirmation_required(bus)
    test_combined.test_analyze_market_internal_signal_no_confirmation(bus)
    test_combined.test_analyze_market_confirmation_match(bus)
    test_combined.test_analyze_market_confirmation_conflict(bus)
    test_combined.test_analyze_market_no_detector_signal_for_confirmation(bus)
    test_combined.test_analyze_market_flat_internal_analysis(bus)

    # Manually run async test if this script is run directly
    import asyncio
    print("Running async test_signal_buffer_timeout...")
    asyncio.run(test_combined.test_signal_buffer_timeout(bus))

    print("--- CombinedStrategy Tests Completed ---")
    print("\nConsider running tests with `pytest` for better output and features.")
