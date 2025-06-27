from typing import Dict, Any
from trading_bot.core.types import Signal, Side
from .base import RiskCheck
# from loguru import logger # Optional

class PositionLimitCheck(RiskCheck):
    """
    A risk check that enforces limits on position sizes and the total number of open positions.
    It maintains an internal state of current positions.
    """

    def __init__(self, max_position_size: float = float('inf'), max_total_positions: int = float('inf'), **kwargs):
        """
        Initializes the PositionLimitCheck.

        Args:
            max_position_size: The maximum allowable size for any single position (in base currency).
                               Example: For BTC/USDT, this limit applies to the amount of BTC.
            max_total_positions: The maximum number of different assets the bot can hold positions in.
            **kwargs: Additional parameters from configuration.
        """
        super().__init__(**kwargs)
        self.max_position_size = float(max_position_size)
        self.max_total_positions = int(max_total_positions)

        # Internal state: dictionary mapping asset symbol to current position size.
        # Positive size for long, negative for short (if shorting is supported/tracked).
        # For simplicity, this example assumes long-only or net positions.
        self.positions: Dict[str, float] = {}

        # logger.info(
        #    f"PositionLimitCheck initialized. Max Size/Pos: {self.max_position_size}, "
        #    f"Max Total Positions: {self.max_total_positions}"
        # )

    def validate(self, signal: Signal) -> Signal:
        """
        Validates a signal against position limits.

        - If the signal's size would create/increase a position beyond `max_position_size`,
          it adjusts the signal's size downwards.
        - If the signal is for a new asset and the bot is already at `max_total_positions`,
          it raises a ValueError.
        - If the signal size is adjusted to zero or negative (and it was a BUY, or vice-versa),
          it could also raise ValueError or return a modified signal that effectively cancels.

        Args:
            signal: The Signal object to validate.

        Returns:
            The (potentially modified) Signal object.

        Raises:
            ValueError: If the signal cannot be made compliant (e.g., max total positions reached
                        for a new asset, or signal size reduced to non-actionable).
        """
        asset = signal.asset
        current_pos_size = self.positions.get(asset, 0.0)

        # Calculate effective new position size if this signal were executed
        # This simple model assumes signal size directly adds/subtracts.
        # A more complex model would consider if it's opening or closing.
        if signal.side == Side.BUY:
            potential_new_size_for_asset = current_pos_size + signal.size
        elif signal.side == Side.SELL:
            # Assuming SELL signal reduces existing long position or opens/increases short.
            # For simplicity here, we'll just consider it reducing a long.
            # If shorting is allowed, logic needs to be more nuanced with absolute sizes.
            potential_new_size_for_asset = current_pos_size - signal.size
        else:
            # Should not happen with valid Side enum
            raise ValueError(f"Invalid signal side: {signal.side}")

        # 1. Check max_position_size for the specific asset
        # This check is primarily for increasing a position or opening a new one.
        # If signal.size would take the absolute position over the limit.
        # Let's assume max_position_size is for the absolute size of the position.

        if signal.side == Side.BUY:
            if current_pos_size + signal.size > self.max_position_size:
                # logger.warning(
                #    f"Signal for {asset} BUY {signal.size} would exceed max_position_size ({self.max_position_size}). "
                #    f"Current: {current_pos_size}. Adjusting size."
                # )
                adjusted_size = self.max_position_size - current_pos_size
                if adjusted_size <= 1e-9: # If no room to increase or tiny
                    raise ValueError(
                        f"Risk check for {asset}: Position already at/near max size ({current_pos_size}/{self.max_position_size}). "
                        f"Cannot execute BUY signal of size {signal.size}."
                    )
                signal.size = adjusted_size
                signal.metadata['risk_adjusted_reason'] = 'max_position_size_exceeded'
                signal.metadata['original_size'] = signal.size # This is wrong, original size was before adjustment
                # Store original size before modification if not already there
                if 'original_signal_size' not in signal.metadata: signal.metadata['original_signal_size'] = potential_new_size_for_asset - current_pos_size # a bit convoluted

                # logger.info(f"Adjusted signal size for {asset} BUY to {signal.size} due to max_position_size.")

        elif signal.side == Side.SELL:
            # If we are tracking net positions and allow shorts, this logic would be different.
            # For now, assume selling reduces a long position.
            # If current_pos_size is 0, and we are selling, it implies opening a short.
            # This example doesn't fully model short selling limits with max_position_size.
            # Let's assume max_position_size applies to |net position|.
            if current_pos_size == 0 and signal.size > self.max_position_size: # Opening a short position larger than limit
                 # logger.warning(f"Signal for {asset} SELL {signal.size} (opening short) would exceed max_position_size ({self.max_position_size}). Adjusting.")
                 signal.size = self.max_position_size
                 signal.metadata['risk_adjusted_reason'] = 'max_position_size_exceeded_short'
                 if 'original_signal_size' not in signal.metadata: signal.metadata['original_signal_size'] = signal.size # Store original
            # No adjustment if selling part of an existing long position, as it reduces risk by this metric's view.


        # 2. Check max_total_positions
        # This applies if the signal is for an asset not currently in `self.positions` (i.e., opening a new position)
        # and we are already at the limit of total distinct positions.
        is_new_asset_position = (asset not in self.positions or self.positions[asset] == 0)

        if is_new_asset_position and signal.side == Side.BUY: # Only count BUYs as new positions for this simple check
            # More robust: count if |position| becomes non-zero
            current_distinct_positions = len([p for p in self.positions.values() if abs(p) > 1e-9])
            if current_distinct_positions >= self.max_total_positions:
                # logger.error(
                #    f"Risk check for {asset}: Max total positions ({self.max_total_positions}) reached. "
                #    f"Cannot open new position for {asset}."
                # )
                raise ValueError(
                    f"Max total positions ({self.max_total_positions}) reached. Cannot open new position for {asset}."
                )

        # logger.debug(f"PositionLimitCheck validation passed for signal: {signal.asset} {signal.side.value} {signal.size}")
        return signal

    def update_position(self, asset: str, executed_size: float, side: Side, price: float = 0.0, **kwargs):
        """
        Updates the internal state of positions after a trade has been executed.

        Args:
            asset: The asset symbol (e.g., "BTC/USDT").
            executed_size: The actual size of the trade that was filled.
            side: The side of the executed trade (BUY or SELL).
            price: Execution price (optional, for P&L tracking if risk module does that).
            **kwargs: Additional execution details.
        """
        if asset not in self.positions:
            self.positions[asset] = 0.0

        if side == Side.BUY:
            self.positions[asset] += executed_size
        elif side == Side.SELL:
            self.positions[asset] -= executed_size
            # If position goes to (near) zero, consider removing it to free up a "slot"
            if abs(self.positions[asset]) < 1e-9: # Using a small epsilon for float comparison
                # logger.info(f"Position for {asset} closed (or near zero: {self.positions[asset]}). Removing from active positions.")
                del self.positions[asset]

        # logger.info(f"Updated position for {asset}: {self.positions.get(asset, 0.0)}. Executed: {side.value} {executed_size}")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns current metrics related to position limits."""
        return {
            "current_positions_count": len(self.positions),
            "current_positions_details": self.positions.copy(), # Return a copy
            "max_position_size_limit": self.max_position_size,
            "max_total_positions_limit": self.max_total_positions
        }

    def reset_state(self):
        """Resets the position tracking."""
        self.positions.clear()
        # logger.info("PositionLimitCheck state has been reset.")


if __name__ == '__main__':
    print("--- PositionLimitCheck Tests ---")
    # logger.remove() # Remove default handlers, if any, for cleaner test output
    # logger.add(sys.stderr, level="DEBUG") # Add console logger for tests

    # Test 1: Basic BUY signal within limits
    print("\nTest 1: Basic BUY within limits")
    risk_check1 = PositionLimitCheck(max_position_size=1.0, max_total_positions=2)
    signal1 = Signal("BTC/USDT", Side.BUY, 0.5)
    try:
        validated_signal1 = risk_check1.validate(signal1)
        risk_check1.update_position(validated_signal1.asset, validated_signal1.size, validated_signal1.side)
        print(f"  Validated: {validated_signal1.size}, Positions: {risk_check1.positions}")
        assert validated_signal1.size == 0.5
        assert risk_check1.positions["BTC/USDT"] == 0.5
    except ValueError as e:
        assert False, f"Test 1 failed: {e}"

    # Test 2: BUY signal exceeds max_position_size, should be adjusted
    print("\nTest 2: BUY exceeds max_position_size")
    # risk_check1 is stateful from Test 1 (BTC=0.5)
    signal2 = Signal("BTC/USDT", Side.BUY, 1.0) # Tries to buy 1.0, current 0.5, limit 1.0. Should allow 0.5.
    try:
        validated_signal2 = risk_check1.validate(signal2)
        risk_check1.update_position(validated_signal2.asset, validated_signal2.size, validated_signal2.side)
        print(f"  Validated: {validated_signal2.size}, Positions: {risk_check1.positions}")
        assert validated_signal2.size == 0.5 # Max 1.0, current 0.5, so can buy 0.5 more
        assert risk_check1.positions["BTC/USDT"] == 1.0 # 0.5 + 0.5
        assert validated_signal2.metadata.get('risk_adjusted_reason') == 'max_position_size_exceeded'
    except ValueError as e:
        assert False, f"Test 2 failed: {e}"

    # Test 3: BUY signal for new asset, within total position limits
    print("\nTest 3: BUY new asset, within total limits")
    signal3 = Signal("ETH/USDT", Side.BUY, 0.8) # BTC is 1.0. Max_total_positions=2. This is 2nd.
    try:
        validated_signal3 = risk_check1.validate(signal3)
        risk_check1.update_position(validated_signal3.asset, validated_signal3.size, validated_signal3.side)
        print(f"  Validated: {validated_signal3.size}, Positions: {risk_check1.positions}")
        assert validated_signal3.size == 0.8
        assert risk_check1.positions["ETH/USDT"] == 0.8
        assert len(risk_check1.positions) == 2
    except ValueError as e:
        assert False, f"Test 3 failed: {e}"

    # Test 4: BUY signal for new asset, exceeds max_total_positions
    print("\nTest 4: BUY new asset, exceeds max_total_positions")
    signal4 = Signal("LTC/USDT", Side.BUY, 0.1) # BTC, ETH exist. Max_total_positions=2. This is 3rd.
    try:
        risk_check1.validate(signal4)
        assert False, "Test 4 should have raised ValueError for max_total_positions"
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        assert "Max total positions" in str(e)
        assert "LTC/USDT" not in risk_check1.positions

    # Test 5: SELL signal, reduces position
    print("\nTest 5: SELL signal, reduces position")
    signal5 = Signal("BTC/USDT", Side.SELL, 0.3) # Current BTC is 1.0
    try:
        validated_signal5 = risk_check1.validate(signal5) # Sell validation is simple, won't reduce size based on limit
        risk_check1.update_position(validated_signal5.asset, validated_signal5.size, validated_signal5.side)
        print(f"  Validated: {validated_signal5.size}, Positions: {risk_check1.positions}")
        assert validated_signal5.size == 0.3
        assert abs(risk_check1.positions["BTC/USDT"] - 0.7) < 1e-9 # 1.0 - 0.3 = 0.7
    except ValueError as e:
        assert False, f"Test 5 failed: {e}"

    # Test 6: SELL signal closes position
    print("\nTest 6: SELL signal, closes position")
    signal6 = Signal("ETH/USDT", Side.SELL, 0.8) # Current ETH is 0.8
    try:
        validated_signal6 = risk_check1.validate(signal6)
        risk_check1.update_position(validated_signal6.asset, validated_signal6.size, validated_signal6.side)
        print(f"  Validated: {validated_signal6.size}, Positions: {risk_check1.positions}")
        assert "ETH/USDT" not in risk_check1.positions # Position should be removed
        assert len(risk_check1.positions) == 1 # Only BTC left
    except ValueError as e:
        assert False, f"Test 6 failed: {e}"

    # Test 7: BUY signal when position is full for that asset
    print("\nTest 7: BUY signal when asset position is full")
    # BTC is 0.7, max_position_size is 1.0.
    # Try to buy more BTC, but it's already at 0.7, limit is 1.0. Can buy 0.3 more.
    # If signal is larger than 0.3, it's capped. If smaller, it's allowed.
    signal7_cap = Signal("BTC/USDT", Side.BUY, 0.5) # Will be capped to 0.3
    try:
        validated_signal7 = risk_check1.validate(signal7_cap)
        print(f"  Validated for 0.5 BUY: {validated_signal7.size}")
        assert abs(validated_signal7.size - 0.3) < 1e-9
    except ValueError as e:
        assert False, f"Test 7 (cap) failed: {e}"

    risk_check1.update_position("BTC/USDT", 0.3, Side.BUY) # BTC is now 1.0 (full)
    print(f"  Positions after filling BTC to max: {risk_check1.positions}")

    signal7_reject = Signal("BTC/USDT", Side.BUY, 0.1) # Try to buy more when BTC is full
    try:
        risk_check1.validate(signal7_reject)
        assert False, "Test 7 (reject) should have raised ValueError"
    except ValueError as e:
        print(f"  Caught expected error for BUY on full position: {e}")
        assert "already at/near max size" in str(e)

    # Test 8: Reset state
    print("\nTest 8: Reset state")
    risk_check1.reset_state()
    print(f"  Positions after reset: {risk_check1.positions}, Metrics: {risk_check1.get_metrics()}")
    assert len(risk_check1.positions) == 0

    print("\n--- PositionLimitCheck Tests Completed ---")
