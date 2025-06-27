import asyncio # For asyncio.sleep
import time # Fallback for sync, but we aim for async
from typing import List, Tuple # Added Tuple for return type hint of _calculate_chunks

from trading_bot.core.types import Signal, Order, OrderType, ExecutionResult, Side
from trading_bot.modules.exchange.base import Exchange
from .base import ExecutionStrategy
# from loguru import logger # Optional

class IcebergExecution(ExecutionStrategy):
    """
    An execution strategy that breaks a large order (signal) into smaller chunks (child orders)
    and executes them sequentially, typically with a delay between chunks.
    This is done to minimize market impact or to avoid showing large order sizes on the order book.
    """

    def __init__(self, exchange: Exchange, chunk_size_pct: float = 0.1, interval_seconds: int = 5, **kwargs):
        """
        Initializes the IcebergExecution strategy.

        Args:
            exchange: An instance of an Exchange interface.
            chunk_size_pct: The maximum size of each chunk as a percentage of the total signal size.
                            E.g., 0.1 means each chunk is at most 10% of the total.
            interval_seconds: The delay in seconds between placing each chunk.
            **kwargs: Additional parameters from configuration.
        """
        super().__init__(exchange=exchange, **kwargs)
        if not (0 < chunk_size_pct <= 1):
            raise ValueError("chunk_size_pct must be between 0 (exclusive) and 1 (inclusive)")
        self.chunk_size_pct = chunk_size_pct
        self.interval_seconds = int(interval_seconds)
        # logger.info(
        #    f"IcebergExecution strategy initialized. Chunk Size: {self.chunk_size_pct*100}%, "
        #    f"Interval: {self.interval_seconds}s"
        # )

    def _calculate_chunks(self, total_size: float) -> List[float]:
        """
        Calculates the sizes of the individual chunks for the iceberg order.

        Args:
            total_size: The total size of the order to be executed.

        Returns:
            A list of floats, where each float is the size of a chunk.
            The sum of chunk sizes will equal `total_size`.
        """
        if total_size <= 0:
            return []

        chunks: List[float] = []
        remaining_size = total_size
        max_chunk_abs = total_size * self.chunk_size_pct

        while remaining_size > 1e-9: # Use a small epsilon for float comparison
            # Ensure max_chunk_abs is not zero if total_size is very small but positive
            current_max_chunk = max(max_chunk_abs, 1e-8) if max_chunk_abs > 0 else remaining_size

            chunk = min(remaining_size, current_max_chunk)
            # TODO: Consider exchange precision rules for amount (e.g. min order size, step size)
            # For now, we assume the exchange handles or rounds this.
            # A more robust solution would fetch market data for precision.
            chunks.append(chunk)
            remaining_size -= chunk

            # Safety break if remaining_size isn't decreasing enough
            if len(chunks) > 1000 and remaining_size > 1e-9 : # Arbitrary large number of chunks
                 # logger.warning(f"Iceberg _calculate_chunks potentially stuck. Remaining: {remaining_size}. Chunks: {len(chunks)}")
                 # Add remaining as last chunk to avoid infinite loop with very small chunk_size_pct / total_size
                 if remaining_size > 1e-9: chunks.append(remaining_size)
                 break

        # logger.debug(f"Calculated chunks for total size {total_size}: {chunks}")
        return chunks

    async def execute(self, signal: Signal) -> ExecutionResult:
        """
        Executes the trading signal as an iceberg order.
        It breaks the signal into chunks and places market orders for each chunk sequentially.

        Args:
            signal: The Signal object to be executed.

        Returns:
            An ExecutionResult object summarizing the outcome of all chunk executions.
            If any chunk fails critically, the process may stop early.
        """
        # logger.info(f"Starting Iceberg execution for signal: {signal.asset} {signal.side.value} {signal.size}")
        chunks = self._calculate_chunks(signal.size)
        if not chunks:
            # logger.warning(f"No chunks calculated for signal size {signal.size}. Aborting iceberg execution.")
            return ExecutionResult(order_id="iceberg_no_chunks", success=False, error="No chunks to execute.")

        total_filled_amount = 0.0
        total_fees = 0.0
        # To calculate weighted average price: sum(price * amount) / sum(amount)
        sum_of_price_times_amount = 0.0

        cumulative_results: List[ExecutionResult] = []
        overall_success = True # Assume success until a failure

        for i, chunk_amount in enumerate(chunks):
            # logger.info(f"Executing chunk {i+1}/{len(chunks)}: Size={chunk_amount} for {signal.asset}")
            chunk_order = Order(
                symbol=signal.asset,
                side=signal.side,
                order_type=OrderType.MARKET, # Iceberg typically uses market orders for chunks
                amount=chunk_amount,
                metadata={
                    **(signal.metadata or {}),
                    "iceberg_parent_signal_ts": signal.timestamp.isoformat(),
                    "iceberg_chunk_num": i + 1,
                    "iceberg_total_chunks": len(chunks)
                }
            )

            try:
                chunk_result = await self.exchange.place_order(chunk_order)
                # logger.debug(f"Chunk {i+1} result: {chunk_result}")
                cumulative_results.append(chunk_result)

                if chunk_result.success and chunk_result.filled_amount > 0:
                    total_filled_amount += chunk_result.filled_amount
                    total_fees += chunk_result.fees
                    sum_of_price_times_amount += chunk_result.average_price * chunk_result.filled_amount
                    # If a chunk is partially filled but considered success by exchange, we continue.
                    # If filled_amount is 0 but success is true, it's odd, but we'd continue based on success.
                elif not chunk_result.success:
                    overall_success = False
                    # logger.error(f"Chunk {i+1} execution failed: {chunk_result.error}. Stopping iceberg execution.")
                    # Decide if we should stop or continue. For now, stop on any failure.
                    break
                # else: success is true but filled_amount is 0. This is unusual for market orders. Log it.
                    # logger.warning(f"Chunk {i+1} for {signal.asset} reported success but filled_amount is 0.")


            except Exception as e:
                # logger.error(f"Exception during Iceberg chunk {i+1} execution for {signal.asset}: {e}", exc_info=True)
                overall_success = False
                # Create a failed ExecutionResult for this chunk to add to cumulative if needed, or just break.
                cumulative_results.append(ExecutionResult(
                    order_id=f"iceberg_chunk_{i+1}_exception",
                    success=False,
                    error=str(e)
                ))
                break # Stop on critical exception

            # Delay before the next chunk, unless it's the last one
            if i < len(chunks) - 1:
                # logger.debug(f"Waiting {self.interval_seconds}s before next chunk...")
                # Addressing item 3 in "things to check.txt": Use asyncio.sleep
                await asyncio.sleep(self.interval_seconds)

        final_average_price = (sum_of_price_times_amount / total_filled_amount) if total_filled_amount > 0 else 0.0

        # Consolidate order IDs if possible, or use a synthetic one.
        # For simplicity, we'll use a synthetic ID here. A real system might store all child order IDs.
        final_order_id = f"iceberg_agg_{signal.asset.replace('/', '')}_{time.time()}"
        if cumulative_results: # If at least one chunk attempted
            final_order_id = cumulative_results[0].order_id + f"_iceberg_parent" # Use first chunk's ID as base for parent

        # Determine final error message if not successful
        final_error = None
        if not overall_success:
            # Get error from the last failed result
            for res in reversed(cumulative_results):
                if not res.success and res.error:
                    final_error = f"Iceberg failed on chunk: {res.error}"
                    break
            if not final_error: # Generic if no specific error found
                 final_error = "One or more iceberg chunks failed to execute."


        # logger.info(
        #    f"Iceberg execution completed for {signal.asset}. Overall Success: {overall_success}. "
        #    f"Total Filled: {total_filled_amount:.8f} (Target: {signal.size:.8f}). Avg Price: {final_average_price:.2f}. Total Fees: {total_fees:.8f}"
        # )

        return ExecutionResult(
            order_id=final_order_id, # This could be an aggregation of child order IDs or a new parent ID
            success=overall_success, # True if all chunks (or as many as possible) succeeded
            filled_amount=total_filled_amount,
            average_price=final_average_price,
            fees=total_fees,
            error=final_error # Error message if overall_success is False
            # metadata could include list of child_order_ids and their results
        )


if __name__ == '__main__':
    # Mocking necessary components for testing
    class MockExchange(Exchange):
        def __init__(self, params: dict):
            super().__init__(params)
            self.order_id_counter = 1
            self.fail_after_n_orders = -1 # -1 means never fail, 0 means fail first, 1 means fail second etc.
            self.orders_processed = 0

        async def place_order(self, order: Order) -> ExecutionResult:
            self.orders_processed += 1
            if self.fail_after_n_orders != -1 and self.orders_processed > self.fail_after_n_orders:
                # logger.warn(f"MockExchange: Simulating failure for order {self.orders_processed} (chunk).")
                return ExecutionResult(order_id=f"mock_chunk_fail_{self.order_id_counter}", success=False, error="Simulated chunk failure")

            order_id = f"mock_chunk_{self.order_id_counter}"
            self.order_id_counter += 1

            # Simulate full fill for chunks
            filled_amount = order.amount
            avg_price = 50000.0 if "BTC" in order.symbol else 2000.0
            fees = filled_amount * avg_price * 0.001

            # logger.info(f"MockExchange: Simulating successful chunk placement for {order.symbol}, ID: {order_id}, Amount: {filled_amount}")
            return ExecutionResult(
                order_id=order_id,
                success=True,
                filled_amount=filled_amount,
                average_price=avg_price,
                fees=fees
            )
        async def cancel_order(self, order_id: str, symbol: str = None) -> bool: return True
        async def get_balance(self, asset: str = None) -> dict: return {"USDT": 100000.0} # Enough balance
        async def get_order_status(self, order_id: str, symbol: str = None) -> ExecutionResult: return None # type: ignore
        async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', since: int = None, limit: int = None) -> list: return []


    async def main_iceberg_tests():
        print("--- IcebergExecution Tests ---")

        # Test _calculate_chunks
        print("\nTest _calculate_chunks:")
        temp_iceberg = IcebergExecution(None, chunk_size_pct=0.25, interval_seconds=1) # type: ignore
        chunks1 = temp_iceberg._calculate_chunks(10) # 10 / 0.25 = 4 chunks of 2.5
        print(f"  Chunks for 10, 25%: {chunks1}, Sum: {sum(chunks1)}")
        assert chunks1 == [2.5, 2.5, 2.5, 2.5]

        chunks2 = temp_iceberg._calculate_chunks(10.5) # 10.5 * 0.25 = 2.625. Chunks: 2.625, 2.625, 2.625, 2.625
        print(f"  Chunks for 10.5, 25%: {chunks2}, Sum: {sum(chunks2)}")
        assert sum(chunks2) == 10.5 and len(chunks2) == 4

        chunks3 = temp_iceberg._calculate_chunks(0.00000001) # Very small
        print(f"  Chunks for 0.00000001, 25%: {chunks3}, Sum: {sum(chunks3)}")
        assert sum(chunks3) == 0.00000001

        chunks4 = IcebergExecution(None, chunk_size_pct=0.33, interval_seconds=1)._calculate_chunks(10) # type: ignore
        # 10 * 0.33 = 3.3. Chunks: 3.3, 3.3, 3.3, 0.1 (approx)
        print(f"  Chunks for 10, 33%: {chunks4}, Sum: {sum(chunks4)}")
        assert abs(sum(chunks4) - 10) < 1e-9 and len(chunks4) == 4


        # Test full execution
        print("\nTest 1: Successful Iceberg execution (all chunks succeed)")
        mock_exchange_success = MockExchange(params={'name': 'mock_success'})
        iceberg_executor_success = IcebergExecution(mock_exchange_success, chunk_size_pct=0.4, interval_seconds=0) # Interval 0 for faster test

        buy_signal_large_btc = Signal(asset="BTC/USDT", side=Side.BUY, size=1.0, confidence=0.9) # 1.0 * 0.4 => 0.4, 0.4, 0.2
        result_large_btc = await iceberg_executor_success.execute(buy_signal_large_btc)
        print(f"  Result for BTC BUY (1.0 size, 40% chunks): {result_large_btc}")
        assert result_large_btc.success is True
        assert abs(result_large_btc.filled_amount - 1.0) < 1e-9
        assert result_large_btc.average_price > 0 # Should be calculated from chunks
        assert mock_exchange_success.orders_processed == 3 # 0.4, 0.4, 0.2

        # Test 2: Iceberg execution where a chunk fails
        print("\nTest 2: Iceberg execution with a failing chunk")
        mock_exchange_fail_chunk = MockExchange(params={'name': 'mock_fail_chunk'})
        mock_exchange_fail_chunk.fail_after_n_orders = 1 # First chunk ok, second fails
        iceberg_executor_fail = IcebergExecution(mock_exchange_fail_chunk, chunk_size_pct=0.4, interval_seconds=0)

        sell_signal_eth = Signal(asset="ETH/USDT", side=Side.SELL, size=2.0, confidence=0.8) # Chunks: 0.8, 0.8, 0.4
        result_fail_eth = await iceberg_executor_fail.execute(sell_signal_eth)
        print(f"  Result for ETH SELL (2.0 size, 40% chunks, 2nd chunk fails): {result_fail_eth}")
        assert result_fail_eth.success is False
        assert abs(result_fail_eth.filled_amount - 0.8) < 1e-9 # Only first chunk filled
        assert result_fail_eth.error == "Iceberg failed on chunk: Simulated chunk failure"
        assert mock_exchange_fail_chunk.orders_processed == 2 # First succeeded, second attempted and failed

        # Test 3: Signal size smaller than effective first chunk (should just be one chunk)
        print("\nTest 3: Signal size leads to a single chunk")
        mock_exchange_single = MockExchange(params={'name': 'mock_single'})
        iceberg_executor_single = IcebergExecution(mock_exchange_single, chunk_size_pct=0.5, interval_seconds=0)
        buy_signal_small_btc = Signal(asset="BTC/USDT", side=Side.BUY, size=0.1, confidence=0.9) # 0.1 * 0.5 = 0.05. Total size 0.1. So one chunk of 0.1
        result_small_btc = await iceberg_executor_single.execute(buy_signal_small_btc)
        print(f"  Result for BTC BUY (0.1 size, 50% chunks): {result_small_btc}")
        assert result_small_btc.success is True
        assert abs(result_small_btc.filled_amount - 0.1) < 1e-9
        assert mock_exchange_single.orders_processed == 1

        print("\n--- IcebergExecution Tests Completed ---")

    pass # For the agent

if __name__ == '__main__':
    from trading_bot.core.types import Side # Re-import for direct script execution
    # import sys
    # logger.remove()
    # logger.add(sys.stderr, level="DEBUG")
    asyncio.run(main_iceberg_tests())
