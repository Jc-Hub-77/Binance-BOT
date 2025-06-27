from trading_bot.core.types import Signal, Order, OrderType, ExecutionResult
from trading_bot.modules.exchange.base import Exchange # Assuming Exchange base class
from .base import ExecutionStrategy
# from loguru import logger # Optional

class MarketOrderExecution(ExecutionStrategy):
    """
    An execution strategy that places simple market orders.
    It converts a Signal directly into a market Order and sends it to the exchange.
    """

    def __init__(self, exchange: Exchange, **kwargs):
        """
        Initializes the MarketOrderExecution strategy.

        Args:
            exchange: An instance of an Exchange interface.
            **kwargs: Additional parameters from configuration.
        """
        super().__init__(exchange=exchange, **kwargs)
        # logger.info(f"MarketOrderExecution strategy initialized for exchange: {exchange.__class__.__name__}")

    async def execute(self, signal: Signal) -> ExecutionResult:
        """
        Executes the trading signal by placing a market order on the exchange.

        Args:
            signal: The Signal object to be executed.

        Returns:
            An ExecutionResult object detailing the outcome of the order placement.
        """
        # logger.debug(f"Executing market order for signal: Asset={signal.asset}, Side={signal.side.value}, Size={signal.size}")

        order_to_place = Order(
            symbol=signal.asset,
            side=signal.side,
            order_type=OrderType.MARKET,
            amount=signal.size,
            metadata=signal.metadata.copy() if signal.metadata else {} # Ensure metadata is copied
        )

        try:
            # logger.info(f"Placing market order: {order_to_place.symbol}, {order_to_place.side.value}, {order_to_place.amount}")
            execution_result = await self.exchange.place_order(order_to_place)
            # logger.info(f"Market order execution result: {execution_result}")
            return execution_result
        except Exception as e:
            # logger.error(f"Exception during market order execution for {signal.asset}: {e}", exc_info=True)
            return ExecutionResult(
                order_id="", # No order ID if placement failed before exchange interaction
                success=False,
                error=f"MarketOrderExecution failed: {str(e)}"
                # filled_amount, average_price, fees will be default 0.0
            )

if __name__ == '__main__':
    import asyncio

    # Mocking necessary components for testing
    class MockExchange(Exchange):
        def __init__(self, params: dict):
            super().__init__(params)
            self.order_id_counter = 1

        async def place_order(self, order: Order) -> ExecutionResult:
            # logger.debug(f"MockExchange: Received order: {order}")
            if order.symbol == "FAIL/USDT":
                # logger.warn("MockExchange: Simulating placement failure for FAIL/USDT")
                return ExecutionResult(order_id="", success=False, error="Simulated exchange error")

            order_id = f"mock_order_{self.order_id_counter}"
            self.order_id_counter += 1
            # Simulate partial fill for testing, or full fill
            filled_amount = order.amount * 0.9 # Simulate 90% fill
            avg_price = 50000.0 if "BTC" in order.symbol else 2000.0
            fees = filled_amount * avg_price * 0.001 # Simulate 0.1% fee

            # logger.info(f"MockExchange: Simulating successful order placement for {order.symbol}, ID: {order_id}")
            return ExecutionResult(
                order_id=order_id,
                success=True,
                filled_amount=filled_amount,
                average_price=avg_price,
                fees=fees
            )

        async def cancel_order(self, order_id: str, symbol: str = None) -> bool: return True
        async def get_balance(self, asset: str = None) -> dict: return {"USDT": 10000.0}
        async def get_order_status(self, order_id: str, symbol: str = None) -> ExecutionResult: return None # type: ignore
        async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', since: int = None, limit: int = None) -> list: return []


    async def main():
        print("--- MarketOrderExecution Tests ---")
        mock_exchange = MockExchange(params={'name': 'mock'})
        market_executor = MarketOrderExecution(exchange=mock_exchange)

        # Test 1: Successful market order
        print("\nTest 1: Successful market order")
        buy_signal_btc = Signal(asset="BTC/USDT", side=Side.BUY, size=0.1, confidence=0.9)
        result_btc = await market_executor.execute(buy_signal_btc)
        print(f"  Result for BTC BUY: {result_btc}")
        assert result_btc.success is True
        assert result_btc.order_id.startswith("mock_order_")
        assert result_btc.filled_amount > 0

        # Test 2: Market order that simulates an exchange error
        print("\nTest 2: Failed market order (simulated error)")
        sell_signal_fail = Signal(asset="FAIL/USDT", side=Side.SELL, size=10.0, confidence=0.7)
        result_fail = await market_executor.execute(sell_signal_fail)
        print(f"  Result for FAIL SELL: {result_fail}")
        assert result_fail.success is False
        assert result_fail.error == "Simulated exchange error" # This depends on how MockExchange returns error
        assert result_fail.order_id == ""


        # Test 3: Market order with metadata
        print("\nTest 3: Market order with metadata")
        buy_signal_eth_meta = Signal(
            asset="ETH/USDT", side=Side.BUY, size=0.5, confidence=0.95,
            metadata={"source_strategy": "TestCombined", "slippage_tolerance": 0.005}
        )
        result_eth_meta = await market_executor.execute(buy_signal_eth_meta)
        print(f"  Result for ETH BUY with metadata: {result_eth_meta}")
        assert result_eth_meta.success is True
        # In a real scenario, you might check if metadata was passed or used by the exchange connector

        print("\n--- MarketOrderExecution Tests Completed ---")

    # To run the test (if this file is executed directly)
    # from trading_bot.core.types import Side # If not already imported at top level
    # asyncio.run(main())
    pass # Add pass here as asyncio.run will not be called by the agent directly.
         # The main() function is for local testing if someone runs this file.

# If running this file directly for testing:
if __name__ == '__main__':
    # Need to re-import Side if it's not at the top level of this script for the main() to run
    from trading_bot.core.types import Side
    # Setup basic logging for the test
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    # logger.remove()
    # logger.add(sys.stderr, level="DEBUG")

    asyncio.run(main())
