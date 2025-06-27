from typing import List, Optional, Dict, Any # Added Dict, Any
import numpy as np
from datetime import datetime, timedelta # For signal buffer cleanup

from trading_bot.core.types import Signal, Side, SignalSource
from trading_bot.core.events import EventBus
from .base import Strategy
# from loguru import logger # Optional: for logging

class CombinedStrategy(Strategy):
    """
    A strategy that combines signals from an external detector with its own internal market analysis.
    It can be configured to require confirmation from its internal analysis before acting on a detector signal.
    It also maintains a buffer of recent detector signals.
    """
    def __init__(self,
                 event_bus: EventBus = None,
                 require_confirmation: bool = True,
                 internal_analysis_weight: float = 0.5, # Example of a new param
                 buffer_timeout_minutes: int = 5, # How long to keep signals in buffer
                 **kwargs):
        """
        Initializes the CombinedStrategy.

        Args:
            event_bus: An optional EventBus instance.
            require_confirmation: If True, detector signals will only be processed if
                                  internal analysis also confirms a similar signal.
            internal_analysis_weight: A factor determining the influence of internal analysis (example).
            buffer_timeout_minutes: Duration in minutes to keep signals in the buffer.
            **kwargs: Additional parameters from configuration.
        """
        super().__init__(event_bus=event_bus, **kwargs)
        self.require_confirmation = require_confirmation
        self.internal_analysis_weight = internal_analysis_weight # Store for potential use

        # Fix for item 2 in "things to check.txt": Initialize signal_buffer
        self.signal_buffer: Dict[str, Signal] = {}
        self.buffer_timeout = timedelta(minutes=buffer_timeout_minutes)

        # logger.info(
        #    f"CombinedStrategy initialized. Require Confirmation: {self.require_confirmation}, "
        #    f"Internal Weight: {self.internal_analysis_weight}, Buffer Timeout: {self.buffer_timeout}"
        # )

    def _cleanup_signal_buffer(self):
        """Removes outdated signals from the buffer."""
        now = datetime.utcnow()
        expired_keys = [
            asset for asset, signal in self.signal_buffer.items()
            if now - signal.timestamp > self.buffer_timeout
        ]
        for key in expired_keys:
            del self.signal_buffer[key]
            # logger.debug(f"Removed expired signal for {key} from buffer.")

    def process_signal(self, signal: Signal) -> List[Signal]:
        """
        Processes an incoming signal from the detector.
        The signal is stored in a buffer. If `require_confirmation` is False,
        the signal is immediately returned for execution. Otherwise, it waits for
        `analyze_market` to potentially confirm it.

        Args:
            signal: The incoming Signal object, presumably from a detector.

        Returns:
            A list of signals. If `require_confirmation` is False, this list contains
            the input signal. Otherwise, it's an empty list, as confirmation happens
            during `analyze_market`.
        """
        self._cleanup_signal_buffer()

        if signal.source != SignalSource.DETECTOR:
            # logger.warning(f"CombinedStrategy.process_signal received a non-detector signal: {signal.source}. Ignoring.")
            return []

        # logger.debug(f"Processing detector signal for {signal.asset}: {signal.side}, Conf: {signal.confidence}")
        self.signal_buffer[signal.asset] = signal

        if not self.require_confirmation:
            # logger.info(f"Confirmation not required. Forwarding detector signal for {signal.asset}.")
            return [signal]
        else:
            # logger.debug(f"Confirmation required for {signal.asset}. Signal buffered. Waiting for market analysis.")
            # The actual combination logic might happen in analyze_market or a dedicated method
            # For now, if confirmation is required, process_signal itself doesn't emit.
            # Emission will be handled by analyze_market if a match is found.
            return []

    def analyze_market(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Performs internal market analysis and combines results with buffered detector signals
        if `require_confirmation` is True.

        Args:
            market_data: A dictionary where keys are asset symbols (e.g., "BTC/USDT")
                         and values are data for that asset (e.g., klines, order book info).
                         Example: `{'BTC/USDT': {'prices': [1,2,3,...], 'volume': ...}}`

        Returns:
            A list of Signal objects to be acted upon.
        """
        self._cleanup_signal_buffer()
        generated_signals: List[Signal] = []

        for asset, data in market_data.items():
            internal_signal = self._analyze_single_asset(asset, data)

            if self.require_confirmation:
                buffered_detector_signal = self.signal_buffer.get(asset)
                if internal_signal and buffered_detector_signal:
                    # Example confirmation: if both signals exist and have the same side
                    if internal_signal.side == buffered_detector_signal.side:
                        # Create a new combined signal or modify one of them
                        combined_signal = Signal(
                            asset=asset,
                            side=internal_signal.side, # Or detector_signal.side
                            size=min(internal_signal.size, buffered_detector_signal.size), # Example: take min size
                            confidence= (internal_signal.confidence + buffered_detector_signal.confidence) / 2,
                            source=SignalSource.COMBINED,
                            metadata={
                                "internal_confidence": internal_signal.confidence,
                                "detector_confidence": buffered_detector_signal.confidence,
                                "original_detector_signal_ts": buffered_detector_signal.timestamp.isoformat()
                            }
                        )
                        generated_signals.append(combined_signal)
                        # logger.info(f"Confirmed signal for {asset} via internal analysis and detector signal. Emitting combined signal.")
                        # Remove from buffer as it's now processed
                        del self.signal_buffer[asset]
                    else:
                        # logger.debug(f"Internal signal for {asset} ({internal_signal.side}) contradicts buffered detector signal ({buffered_detector_signal.side}). No combined signal.")
                        pass
                elif internal_signal:
                    # logger.debug(f"Internal signal for {asset} generated, but no matching buffered detector signal for confirmation.")
                    # Optionally, emit purely internal signals if desired, even if require_confirmation is true for detector signals
                    # generated_signals.append(internal_signal) # If we want to trade on pure internal signals too
                    pass

            elif internal_signal: # Not require_confirmation, so emit any internal signals
                generated_signals.append(internal_signal)
                # logger.info(f"Generated internal signal for {asset} (confirmation not required).")

        # If not requiring confirmation, we might have already emitted detector signals in process_signal.
        # This part ensures that any remaining buffered signals (if logic changes) or purely internal signals are handled.
        # However, with current `process_signal` logic, if `require_confirmation` is false, detector signals are already out.
        # So, if `require_confirmation` is false, `generated_signals` here will only contain purely internal signals.

        # logger.debug(f"CombinedStrategy.analyze_market completed. Generated {len(generated_signals)} signals.")
        return generated_signals

    def _analyze_single_asset(self, asset: str, asset_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Performs a simple internal analysis for a single asset.
        Example: Generates a BUY signal if a short-term moving average crosses
        above a long-term moving average.

        Args:
            asset: The asset symbol (e.g., "BTC/USDT").
            asset_data: Data for the asset, expected to contain 'prices'.
                        Example: `{'prices': [float, float, ...]}`

        Returns:
            A Signal object if a trading opportunity is identified, otherwise None.
        """
        prices = asset_data.get('prices')
        if not prices or not isinstance(prices, list) or len(prices) < 50:
            # logger.debug(f"Not enough price data for {asset} to perform internal analysis (need 50, got {len(prices) if prices else 0}).")
            return None

        try:
            # Ensure prices are numbers
            prices = [float(p) for p in prices]
        except (ValueError, TypeError):
            # logger.error(f"Invalid price data for {asset}. Contains non-numeric values.")
            return None

        # Simple Moving Average Crossover Example
        ma_short_period = 20
        ma_long_period = 50

        # Calculate MAs for the most recent point
        current_ma_short = np.mean(prices[-ma_short_period:])
        current_ma_long = np.mean(prices[-ma_long_period:])

        # Calculate MAs for the previous point to check for crossover
        prev_ma_short = np.mean(prices[-ma_short_period-1:-1])
        prev_ma_long = np.mean(prices[-ma_long_period-1:-1])

        signal_confidence = 0.6 # Base confidence for internal signals

        # Bullish crossover: short MA was below long MA, now it's above
        if prev_ma_short <= prev_ma_long and current_ma_short > current_ma_long:
            # logger.info(f"Internal Analysis for {asset}: Bullish MA crossover detected. Short MA: {current_ma_short:.2f}, Long MA: {current_ma_long:.2f}")
            return Signal(
                asset=asset,
                side=Side.BUY,
                size=1.0,  # Example size, could be dynamic
                confidence=signal_confidence,
                source=SignalSource.INTERNAL,
                metadata={'analysis_type': 'MA_crossover_bullish'}
            )
        # Bearish crossover: short MA was above long MA, now it's below
        elif prev_ma_short >= prev_ma_long and current_ma_short < current_ma_long:
            # logger.info(f"Internal Analysis for {asset}: Bearish MA crossover detected. Short MA: {current_ma_short:.2f}, Long MA: {current_ma_long:.2f}")
            return Signal(
                asset=asset,
                side=Side.SELL,
                size=1.0,  # Example size
                confidence=signal_confidence,
                source=SignalSource.INTERNAL,
                metadata={'analysis_type': 'MA_crossover_bearish'}
            )

        # logger.debug(f"No significant MA crossover for {asset}. Short MA: {current_ma_short:.2f}, Long MA: {current_ma_long:.2f}")
        return None


if __name__ == '__main__':
    # Example Usage
    print("--- CombinedStrategy Tests ---")
    prices_bullish = list(np.linspace(100, 90, 30)) + list(np.linspace(91, 110, 20)) # Ends with upward trend
    prices_bearish = list(np.linspace(90, 100, 30)) + list(np.linspace(99, 80, 20)) # Ends with downward trend
    prices_flat = list(np.linspace(100, 100, 50))

    mock_market_data_bullish = {"BTC/USDT": {"prices": prices_bullish}}
    mock_market_data_bearish = {"ETH/USDT": {"prices": prices_bearish}}
    mock_market_data_flat = {"LTC/USDT": {"prices": prices_flat}}

    detector_signal_btc_buy = Signal("BTC/USDT", Side.BUY, 1.0, 0.8, SignalSource.DETECTOR)
    detector_signal_btc_sell = Signal("BTC/USDT", Side.SELL, 1.0, 0.8, SignalSource.DETECTOR)
    detector_signal_eth_sell = Signal("ETH/USDT", Side.SELL, 1.0, 0.8, SignalSource.DETECTOR)

    # Test 1: require_confirmation = False
    print("\nTest 1: require_confirmation = False")
    strategy_no_confirm = CombinedStrategy(require_confirmation=False, buffer_timeout_minutes=1)

    # Detector signal should pass through process_signal
    processed = strategy_no_confirm.process_signal(detector_signal_btc_buy)
    assert len(processed) == 1 and processed[0] == detector_signal_btc_buy
    print(f"  Detector signal passed process_signal: {processed[0].asset} {processed[0].side}")

    # Internal analysis should also generate signals independently
    analyzed = strategy_no_confirm.analyze_market(mock_market_data_bullish)
    assert len(analyzed) == 1 and analyzed[0].source == SignalSource.INTERNAL and analyzed[0].side == Side.BUY
    print(f"  Internal analysis generated signal: {analyzed[0].asset} {analyzed[0].side}")

    # Test 2: require_confirmation = True, matching signals
    print("\nTest 2: require_confirmation = True, matching signals")
    strategy_confirm_match = CombinedStrategy(require_confirmation=True, buffer_timeout_minutes=1)

    processed_buffered = strategy_confirm_match.process_signal(detector_signal_btc_buy) # Buffer the detector signal
    assert len(processed_buffered) == 0 # Should not emit yet
    print(f"  Detector signal buffered for BTC/USDT BUY.")

    analyzed_confirmed = strategy_confirm_match.analyze_market(mock_market_data_bullish) # Internal analysis confirms BUY
    assert len(analyzed_confirmed) == 1
    assert analyzed_confirmed[0].source == SignalSource.COMBINED
    assert analyzed_confirmed[0].side == Side.BUY
    assert analyzed_confirmed[0].asset == "BTC/USDT"
    print(f"  Combined signal generated after confirmation: {analyzed_confirmed[0].asset} {analyzed_confirmed[0].side}")

    # Test 3: require_confirmation = True, conflicting signals
    print("\nTest 3: require_confirmation = True, conflicting signals")
    strategy_confirm_conflict = CombinedStrategy(require_confirmation=True, buffer_timeout_minutes=1)

    strategy_confirm_conflict.process_signal(detector_signal_btc_sell) # Buffer DETECTOR SELL for BTC
    print(f"  Detector signal buffered for BTC/USDT SELL.")

    analyzed_conflicting = strategy_confirm_conflict.analyze_market(mock_market_data_bullish) # Internal analysis suggests BUY for BTC
    assert len(analyzed_conflicting) == 0 # No combined signal due to conflict
    print(f"  No combined signal generated due to conflict (Detector SELL, Internal BUY).")

    # Test 4: require_confirmation = True, only internal signal (no detector signal buffered)
    print("\nTest 4: require_confirmation = True, only internal signal")
    strategy_confirm_internal_only = CombinedStrategy(require_confirmation=True, buffer_timeout_minutes=1)
    # No detector signal processed for ETH/USDT
    analyzed_internal_eth = strategy_confirm_internal_only.analyze_market(mock_market_data_bearish) # Internal suggests SELL for ETH
    # Depending on strictness, this might or might not generate a signal.
    # Current logic: if require_confirmation, it needs a buffered detector signal. So, 0.
    assert len(analyzed_internal_eth) == 0
    print(f"  No signal for ETH/USDT as no detector signal was buffered for confirmation (though internal analysis suggested SELL).")

    # Test 5: require_confirmation = True, only detector signal (internal analysis is flat)
    print("\nTest 5: require_confirmation = True, only detector signal (internal analysis flat)")
    strategy_confirm_detector_only = CombinedStrategy(require_confirmation=True, buffer_timeout_minutes=1)
    strategy_confirm_detector_only.process_signal(Signal("LTC/USDT", Side.BUY, 1.0, 0.7, SignalSource.DETECTOR))
    print(f"  Detector signal buffered for LTC/USDT BUY.")
    analyzed_flat_ltc = strategy_confirm_detector_only.analyze_market(mock_market_data_flat) # Internal is flat for LTC
    assert len(analyzed_flat_ltc) == 0
    print(f"  No combined signal for LTC/USDT as internal analysis was flat.")

    # Test 6: Signal buffer timeout
    print("\nTest 6: Signal buffer timeout")
    strategy_buffer_test = CombinedStrategy(require_confirmation=True, buffer_timeout_minutes=0.001) # Very short timeout
    strategy_buffer_test.process_signal(detector_signal_btc_buy)
    print(f"  Detector signal for BTC/USDT BUY buffered.")
    import time
    time.sleep(0.1) # Wait for buffer to time out
    analyzed_after_timeout = strategy_buffer_test.analyze_market(mock_market_data_bullish)
    assert len(analyzed_after_timeout) == 0 # Buffered signal should have expired
    print(f"  No combined signal for BTC/USDT as buffered detector signal expired.")

    print("\n--- CombinedStrategy Tests Completed ---")
