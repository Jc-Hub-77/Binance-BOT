import asyncio # For async operations if any complex simulation logic added
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import random
# from loguru import logger

from trading_bot.core.types import Order, ExecutionResult, OrderType, Side
from .base import Exchange

class SimulatedExchange(Exchange):
    """
    A simulated exchange environment for backtesting or paper trading.
    It mimics basic exchange functionalities like order placement, balance management,
    and price updates (simplified).
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the SimulatedExchange.

        Args:
            params: Configuration parameters:
                - 'initial_balances': Dict[str, float], e.g., {"USDT": 10000, "BTC": 1}.
                - 'default_fees': Dict containing 'maker' and 'taker' fee rates (e.g., 0.001 for 0.1%).
                - 'simulated_latency_ms': Optional int, latency in milliseconds to simulate.
                - 'price_feed_config': Optional dict, config for how prices are updated (e.g., CSV path).
        """
        super().__init__(params)
        self.initial_balances = params.get('initial_balances', {"USDT": 10000.0})
        self.balances: Dict[str, float] = defaultdict(float)
        self.balances.update(self.initial_balances.copy())

        default_fees = params.get('default_fees', {'maker': 0.001, 'taker': 0.001})
        self.maker_fee = default_fees.get('maker', 0.001)
        self.taker_fee = default_fees.get('taker', 0.001)

        self.simulated_latency_ms = params.get('simulated_latency_ms', 0)

        # Internal state for orders and price data
        self.orders: Dict[str, Order] = {} # Store placed orders if needed for status checks
        self.order_id_counter: int = 1
        self.trade_history: List[ExecutionResult] = []

        # Simplified price mechanism: Store current prices, allow manual updates or from a feed
        self.current_prices: Dict[str, float] = params.get('initial_prices', {}) # e.g. {"BTC/USDT": 50000.0}
        self._price_data_source = params.get('price_data_source') # Could be a DataFrame, CSV path, etc.
        self._current_time: datetime = datetime.utcnow() # Internal clock for simulation

        # Load initial prices if a CSV path is provided (example)
        if isinstance(self._price_data_source, str) and self._price_data_source.endswith(".csv"):
            self._load_prices_from_csv(self.params.get('price_data_symbol_column', 'symbol'),
                                       self.params.get('price_data_price_column', 'price'))

        # logger.info(
        #    f"SimulatedExchange initialized. Initial Balances: {self.initial_balances}, "
        #    f"Fees (T/M): {self.taker_fee}/{self.maker_fee}"
        # )
        # logger.debug(f"Initial prices for simulation: {self.current_prices}")


    def _load_prices_from_csv(self, symbol_col: str, price_col: str):
        """ Placeholder for loading initial prices from a CSV. Requires pandas. """
        try:
            import pandas as pd
            df = pd.read_csv(self._price_data_source)
            for _, row in df.iterrows():
                symbol = row[symbol_col]
                price = float(row[price_col])
                self.current_prices[symbol] = price
            # logger.info(f"Loaded initial prices from CSV: {self._price_data_source}. {len(self.current_prices)} symbols priced.")
        except ImportError:
            # logger.warning("Pandas not installed. Cannot load prices from CSV for SimulatedExchange.")
            pass
        except Exception as e:
            # logger.error(f"Error loading prices from CSV {self._price_data_source}: {e}")
            pass

    async def _simulate_latency(self):
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Gets the current simulated price for a symbol.
        Addresses item 5 in "things to check.txt" - this is where price lookup happens.
        The mechanism to *update* these prices over time is separate (e.g., `update_market_data`).
        """
        price = self.current_prices.get(symbol)
        if price is None:
            # logger.warning(f"No price data available in simulation for symbol {symbol}. Using default or failing.")
            # Fallback for common pairs if not explicitly set, for basic testing
            if "BTC/USDT" in symbol: return 50000.0 + random.uniform(-100, 100)
            if "ETH/USDT" in symbol: return 2000.0 + random.uniform(-10, 10)
            return None # Or raise error: raise ExchangeError(f"No price available for {symbol} in simulation")
        return price + (price * random.uniform(-0.0005, 0.0005)) # Add tiny random slippage/fluctuation

    def update_market_data(self, prices: Optional[Dict[str, float]] = None, timestamp: Optional[datetime] = None):
        """
        Allows external updates to the simulated market prices and current time.
        This is crucial for backtesting where a historical data feed drives the simulation.

        Args:
            prices: A dictionary of symbol -> price to update.
            timestamp: The new current time for the simulation.
        """
        if prices:
            self.current_prices.update(prices)
            # logger.debug(f"Simulated prices updated: {prices}")
        if timestamp:
            self._current_time = timestamp
            # logger.debug(f"Simulated time updated: {self._current_time.isoformat()}")

    async def place_order(self, order: Order) -> ExecutionResult:
        await self._simulate_latency()

        order_id = f"SIM_ORDER_{self.order_id_counter}"
        self.order_id_counter += 1
        self.orders[order_id] = order # Store the order

        # logger.info(f"SimulatedExchange: Processing order {order_id} for {order.symbol} {order.side.value} {order.amount} @ {order.order_type.value}")

        base_asset, quote_asset = order.symbol.split('/')
        execution_price = self._get_current_price(order.symbol)

        if execution_price is None:
            # logger.error(f"Order {order_id} for {order.symbol} failed: No price available in simulation.")
            return ExecutionResult(order_id=order_id, success=False, error="No price available for symbol in simulation")

        # For LIMIT orders, check if price is met (simplified)
        if order.order_type == OrderType.LIMIT:
            if order.price is None:
                return ExecutionResult(order_id=order_id, success=False, error="Limit order requires a price.")
            if order.side == Side.BUY and execution_price > order.price:
                # logger.info(f"Limit BUY order {order_id} for {order.symbol} not filled: Market price {execution_price} > Limit price {order.price}")
                return ExecutionResult(order_id=order_id, success=False, error="Limit price not met (BUY)") # Or treat as open
            if order.side == Side.SELL and execution_price < order.price:
                # logger.info(f"Limit SELL order {order_id} for {order.symbol} not filled: Market price {execution_price} < Limit price {order.price}")
                return ExecutionResult(order_id=order_id, success=False, error="Limit price not met (SELL)") # Or treat as open
            # If limit price is met, use it for execution, otherwise use market price (or stricter: only limit price)
            execution_price = order.price # Assume limit order fills at the specified limit price if met

        # Calculate cost/proceeds and fees
        cost_or_proceeds_before_fees = order.amount * execution_price
        fee_rate = self.taker_fee # Assume all simulated orders are taker for simplicity
        fees = cost_or_proceeds_before_fees * fee_rate

        # Check balances
        if order.side == Side.BUY:
            if self.balances[quote_asset] < cost_or_proceeds_before_fees + fees: # Check if enough quote for cost + fee
                # logger.warning(f"Order {order_id} for {order.symbol} failed: Insufficient {quote_asset} balance. Need {cost_or_proceeds_before_fees + fees}, Have {self.balances[quote_asset]}")
                return ExecutionResult(order_id=order_id, success=False, error=f"Insufficient {quote_asset} balance")
            self.balances[base_asset] += order.amount
            self.balances[quote_asset] -= (cost_or_proceeds_before_fees + fees)
        elif order.side == Side.SELL:
            if self.balances[base_asset] < order.amount:
                # logger.warning(f"Order {order_id} for {order.symbol} failed: Insufficient {base_asset} balance. Need {order.amount}, Have {self.balances[base_asset]}")
                return ExecutionResult(order_id=order_id, success=False, error=f"Insufficient {base_asset} balance")
            self.balances[base_asset] -= order.amount
            self.balances[quote_asset] += (cost_or_proceeds_before_fees - fees) # Fees deducted from proceeds

        filled_amount = order.amount # Assume full fill for market orders and met limit orders

        result = ExecutionResult(
            order_id=order_id,
            success=True,
            filled_amount=filled_amount,
            average_price=execution_price,
            fees=fees,
            # metadata={'simulated_fill_time': self._current_time.isoformat()}
        )
        self.trade_history.append(result)
        # logger.info(f"Order {order_id} for {order.symbol} executed successfully. Result: {result}. New Balances: {self.balances}")
        return result

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        await self._simulate_latency()
        if order_id in self.orders:
            # In a more complex simulation, you'd check if it's cancellable (e.g. not fully filled limit order)
            # For now, just remove it.
            del self.orders[order_id]
            # logger.info(f"SimulatedExchange: Order {order_id} cancelled.")
            return True
        # logger.warning(f"SimulatedExchange: Order {order_id} not found for cancellation.")
        return False

    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, float]:
        await self._simulate_latency()
        if asset:
            return {asset: self.balances.get(asset, 0.0)}
        return self.balances.copy() # Return a copy

    async def get_order_status(self, order_id: str, symbol: Optional[str] = None) -> Optional[ExecutionResult]:
        await self._simulate_latency()
        # Find in trade history if it was filled
        for trade in self.trade_history:
            if trade.order_id == order_id:
                # logger.debug(f"SimulatedExchange: Order {order_id} found in trade history (filled). Status: {trade}")
                return trade

        # If not in trade history, it might be an open (unfilled limit) or non-existent order
        # This basic simulation doesn't explicitly track "open" limit orders that haven't filled.
        # It assumes limit orders either fill immediately if price matches or "fail" (don't execute).
        if order_id in self.orders:
             # logger.debug(f"SimulatedExchange: Order {order_id} found in self.orders (implies it was placed but maybe not filled). This sim treats it as not filled if not in history.")
             # This means it was a limit order that didn't meet price conditions at placement.
             # Could return a synthetic "open" status or just None.
             original_order = self.orders[order_id]
             return ExecutionResult(order_id=order_id, success=False, error="Order placed but not filled (e.g. limit price not met)",
                                    # filled_amount=0, average_price=0, fees=0,
                                    # metadata={'status': 'open', 'original_order': original_order}
                                    )
        # logger.warning(f"SimulatedExchange: Order {order_id} not found.")
        return None

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', since: Optional[int] = None, limit: Optional[int] = None) -> List[List[Any]]:
        await self._simulate_latency()
        # This requires a proper historical data feed for the simulation.
        # For now, return empty or a very simple mock.
        # logger.debug(f"SimulatedExchange: fetch_ohlcv called for {symbol}. Basic sim returns canned data or empty.")

        # Example: generate some dummy data based on current price
        current_price = self._get_current_price(symbol)
        if current_price is None: return []

        limit = limit or 2 # Default to a few candles
        ohlcv_data = []
        # Simulate backwards from current time
        timestamp_ms = int(self._current_time.timestamp() * 1000)

        # Timeframe to seconds (very simplified)
        tf_seconds = {'1m': 60, '5m': 300, '1h': 3600, '1d': 86400}.get(timeframe, 3600)

        for i in range(limit):
            ts = timestamp_ms - (i * tf_seconds * 1000)
            open_p = current_price * (1 - 0.01 * i * random.uniform(0.1, 0.5))
            high_p = open_p * (1 + 0.005 * random.uniform(0.1, 1))
            low_p = open_p * (1 - 0.005 * random.uniform(0.1, 1))
            close_p = (open_p + high_p + low_p) / 3 # Arbitrary close
            volume = random.uniform(1, 100)
            ohlcv_data.append([ts, open_p, high_p, low_p, close_p, volume])

        return list(reversed(ohlcv_data)) # CCXT expects chronological order

    def reset_simulation(self, initial_balances: Optional[Dict[str, float]] = None, initial_prices: Optional[Dict[str, float]] = None):
        """Resets the simulation to its initial state or new provided states."""
        self.balances = defaultdict(float)
        self.balances.update(initial_balances or self.initial_balances.copy())
        self.current_prices = initial_prices or self.params.get('initial_prices', {}).copy()
        self.orders.clear()
        self.order_id_counter = 1
        self.trade_history.clear()
        self._current_time = datetime.utcnow() # Reset clock
        if isinstance(self._price_data_source, str) and self._price_data_source.endswith(".csv"):
            self._load_prices_from_csv(self.params.get('price_data_symbol_column', 'symbol'),
                                       self.params.get('price_data_price_column', 'price'))
        # logger.info("SimulatedExchange state has been reset.")

    async def close(self):
        # logger.info("SimulatedExchange closed (no specific resources to release).")
        pass # No specific resources like network connections to close for a sim

async def _simulated_exchange_test_main():
    print("--- SimulatedExchange Test ---")
    sim_params = {
        'initial_balances': {"USDT": 10000, "BTC": 0},
        'initial_prices': {"BTC/USDT": 20000.0, "ETH/USDT": 600.0},
        'default_fees': {'taker': 0.002, 'maker': 0.001} # 0.2% taker fee
    }
    sim_exchange = SimulatedExchange(sim_params)

    # Test get_balance
    print("\nTesting get_balance...")
    balances = await sim_exchange.get_balance()
    print(f"  Initial balances: {balances}")
    assert balances["USDT"] == 10000
    assert balances.get("BTC", 0) == 0

    # Test place_order: BUY BTC
    print("\nTesting place_order (BUY BTC)...")
    buy_order_btc = Order(symbol="BTC/USDT", side=Side.BUY, order_type=OrderType.MARKET, amount=0.1)
    result_buy_btc = await sim_exchange.place_order(buy_order_btc)
    print(f"  BUY BTC result: {result_buy_btc}")
    assert result_buy_btc.success is True
    assert result_buy_btc.filled_amount == 0.1
    assert result_buy_btc.average_price == sim_exchange.current_prices["BTC/USDT"] # Approx, due to internal fluctuation

    expected_cost_btc = 0.1 * sim_exchange.current_prices["BTC/USDT"] # Price might fluctuate slightly
    expected_fees_btc = expected_cost_btc * sim_params['default_fees']['taker']
    # print(f"  Expected cost: {expected_cost_btc}, Expected fees: {expected_fees_btc}")

    new_balances_after_buy = await sim_exchange.get_balance()
    print(f"  Balances after BUY BTC: {new_balances_after_buy}")
    assert abs(new_balances_after_buy["BTC"] - 0.1) < 1e-9
    # Initial USDT 10000 - (0.1 * 20000) - (0.1 * 20000 * 0.002) = 10000 - 2000 - 4 = 7996
    assert abs(new_balances_after_buy["USDT"] - (10000 - result_buy_btc.average_price * 0.1 - result_buy_btc.fees)) < 1e-9


    # Test place_order: SELL BTC
    print("\nTesting place_order (SELL BTC)...")
    # Update price for sell scenario
    sim_exchange.update_market_data(prices={"BTC/USDT": 21000.0})
    sell_order_btc = Order(symbol="BTC/USDT", side=Side.SELL, order_type=OrderType.MARKET, amount=0.05)
    result_sell_btc = await sim_exchange.place_order(sell_order_btc)
    print(f"  SELL BTC result: {result_sell_btc}")
    assert result_sell_btc.success is True
    assert result_sell_btc.filled_amount == 0.05

    new_balances_after_sell = await sim_exchange.get_balance()
    print(f"  Balances after SELL BTC: {new_balances_after_sell}")
    # BTC: 0.1 - 0.05 = 0.05
    assert abs(new_balances_after_sell["BTC"] - 0.05) < 1e-9
    # USDT: 7996 + (0.05 * 21000) - (0.05 * 21000 * 0.002) = 7996 + 1050 - 2.1 = 9043.9
    assert abs(new_balances_after_sell["USDT"] - ( (10000 - result_buy_btc.average_price * 0.1 - result_buy_btc.fees) + result_sell_btc.average_price * 0.05 - result_sell_btc.fees)) < 1e-9


    # Test place_order: Insufficient balance
    print("\nTesting place_order (Insufficient BTC)...")
    sell_too_much_btc = Order(symbol="BTC/USDT", side=Side.SELL, order_type=OrderType.MARKET, amount=1.0) # Have 0.05 BTC
    result_sell_too_much = await sim_exchange.place_order(sell_too_much_btc)
    print(f"  SELL too much BTC result: {result_sell_too_much}")
    assert result_sell_too_much.success is False
    assert "Insufficient BTC balance" in result_sell_too_much.error if result_sell_too_much.error else False

    # Test fetch_ohlcv (basic mock)
    print("\nTesting fetch_ohlcv...")
    ohlcv = await sim_exchange.fetch_ohlcv("ETH/USDT", timeframe="1h", limit=3)
    print(f"  Fetched OHLCV for ETH/USDT: {ohlcv}")
    assert len(ohlcv) == 3
    assert len(ohlcv[0]) == 6 # ts, o, h, l, c, v

    # Test get_order_status
    print("\nTesting get_order_status...")
    status_filled = await sim_exchange.get_order_status(result_buy_btc.order_id)
    print(f"  Status for filled order ({result_buy_btc.order_id}): {status_filled}")
    assert status_filled is not None and status_filled.success is True

    status_nonexistent = await sim_exchange.get_order_status("NONEXISTENT_ID")
    print(f"  Status for non-existent order: {status_nonexistent}")
    assert status_nonexistent is None


    # Test Limit order - Fill
    print("\nTesting Limit Order (BUY, should fill)...")
    sim_exchange.update_market_data(prices={"ETH/USDT": 590.0}) # Market price is 590
    limit_buy_eth_fill = Order(symbol="ETH/USDT", side=Side.BUY, order_type=OrderType.LIMIT, amount=1.0, price=595.0) # Limit buy at 595
    result_limit_fill = await sim_exchange.place_order(limit_buy_eth_fill)
    print(f"  Limit BUY ETH (Price: 595, Market: 590) result: {result_limit_fill}")
    assert result_limit_fill.success is True
    assert result_limit_fill.average_price == 595.0 # Filled at limit price

    # Test Limit order - No Fill
    print("\nTesting Limit Order (BUY, should not fill)...")
    sim_exchange.update_market_data(prices={"ETH/USDT": 610.0}) # Market price is 610
    limit_buy_eth_nofill = Order(symbol="ETH/USDT", side=Side.BUY, order_type=OrderType.LIMIT, amount=1.0, price=605.0) # Limit buy at 605
    result_limit_nofill = await sim_exchange.place_order(limit_buy_eth_nofill)
    print(f"  Limit BUY ETH (Price: 605, Market: 610) result: {result_limit_nofill}")
    assert result_limit_nofill.success is False
    assert "Limit price not met" in result_limit_nofill.error if result_limit_nofill.error else False


    # Test Reset
    print("\nTesting reset_simulation...")
    sim_exchange.reset_simulation(initial_balances={"USDT": 500, "XRP": 1000}, initial_prices={"XRP/USDT": 0.5})
    reset_balances = await sim_exchange.get_balance()
    print(f"  Balances after reset: {reset_balances}")
    assert reset_balances["USDT"] == 500 and reset_balances["XRP"] == 1000
    assert sim_exchange.current_prices["XRP/USDT"] == 0.5

    print("\n--- SimulatedExchange Test Finished ---")

if __name__ == '__main__':
    # from trading_bot.core.types import Side, OrderType, Order # Ensure these are available
    # import sys
    # logger.remove()
    # logger.add(sys.stderr, level="DEBUG")
    asyncio.run(_simulated_exchange_test_main())
