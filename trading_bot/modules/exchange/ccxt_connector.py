import ccxt.async_support as ccxt # Import the async version of ccxt
from typing import Dict, Any, List, Optional
# from loguru import logger

from trading_bot.core.types import Order, ExecutionResult, OrderType, Side
from .base import Exchange
from trading_bot.utils.resilience import with_retry, with_circuit_breaker, exchange_breaker # Assuming these are defined

# It's good practice to define a specific exception for exchange related errors
class ExchangeError(Exception):
    pass

class CCXTConnector(Exchange):
    """
    An exchange connector that uses the CCXT library to interact with various
    cryptocurrency exchanges. This version uses ccxt.async_support for asyncio compatibility.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the CCXTConnector.

        Args:
            params: Configuration parameters:
                - 'exchange_id': The CCXT id of the exchange (e.g., 'binance', 'coinbasepro').
                - 'api_key': Optional API key.
                - 'secret': Optional API secret.
                - 'password': Optional API password (for some exchanges like KuCoin).
                - 'sandbox_mode': Optional boolean to use exchange's sandbox/testnet.
                - 'ccxt_options': Optional dictionary of additional options for CCXT.
        """
        super().__init__(params)
        self.exchange_id = params.get('exchange_id')
        if not self.exchange_id:
            raise ValueError("CCXTConnector requires 'exchange_id' in params.")

        self.api_key = params.get('api_key')
        self.secret = params.get('secret')
        self.password = params.get('password')
        self.sandbox_mode = params.get('sandbox_mode', False)
        self.ccxt_options = params.get('ccxt_options', {})

        self.exchange: Optional[ccxt.Exchange] = None # Will be initialized in _init_client
        self.markets: Optional[Dict[str, Any]] = None
        self._initialized = False # To ensure _init_client is called

        # logger.info(f"CCXTConnector initialized for exchange '{self.exchange_id}'. Sandbox: {self.sandbox_mode}")

    async def _init_client(self):
        """Initializes the CCXT exchange client and loads markets."""
        if self._initialized:
            return

        if not hasattr(ccxt, self.exchange_id):
            # logger.error(f"Exchange ID '{self.exchange_id}' not found in CCXT.")
            raise ExchangeError(f"Unsupported exchange ID: {self.exchange_id}")

        exchange_class = getattr(ccxt, self.exchange_id)

        config = {
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True, # Recommended by CCXT
            **self.ccxt_options # Spread any additional user-defined CCXT options
        }
        if self.password:
            config['password'] = self.password

        self.exchange = exchange_class(config)

        if self.sandbox_mode:
            if self.exchange.has.get('test'): # Check if sandbox mode is supported
                self.exchange.set_sandbox_mode(True)
                # logger.info(f"Sandbox mode enabled for {self.exchange_id}.")
            else:
                # logger.warning(f"Exchange {self.exchange_id} does not support sandbox mode via CCXT, or it's not listed in `has`.")
                # Depending on strictness, one might raise an error here.
                pass

        try:
            # logger.debug(f"Loading markets for {self.exchange_id}...")
            await self.load_markets(reload=True) # Load markets on initialization
            self._initialized = True
            # logger.info(f"CCXT client for {self.exchange_id} initialized successfully. Markets loaded: {len(self.markets) if self.markets else 0}")
        except ccxt.NetworkError as e:
            # logger.error(f"Network error while initializing CCXT client for {self.exchange_id}: {e}", exc_info=True)
            await self.close() # Clean up client if init fails
            raise ExchangeError(f"Failed to connect to {self.exchange_id}: {e}")
        except ccxt.ExchangeError as e: # Catch other CCXT specific errors
            # logger.error(f"CCXT ExchangeError while initializing client for {self.exchange_id}: {e}", exc_info=True)
            await self.close()
            raise ExchangeError(f"Exchange error for {self.exchange_id}: {e}")
        except Exception as e: # Catch any other unexpected errors
            # logger.error(f"Unexpected error initializing CCXT client for {self.exchange_id}: {e}", exc_info=True)
            await self.close()
            raise ExchangeError(f"Unexpected error initializing {self.exchange_id}: {e}")


    @with_retry(stop_attempts=3, wait_min=1, wait_max=5)
    @with_circuit_breaker(exchange_breaker)
    async def place_order(self, order: Order) -> ExecutionResult:
        await self._init_client()
        if not self.exchange: raise ExchangeError("Exchange client not initialized.")

        # logger.info(f"Placing order on {self.exchange_id}: {order.symbol}, {order.side.value}, {order.order_type.value}, Amount: {order.amount}, Price: {order.price}")

        ccxt_order_type = order.order_type.value.lower()
        ccxt_side = order.side.value.lower()
        params = order.metadata.get('ccxt_params', {}) # Allow passing extra params to CCXT

        try:
            # Some exchanges require price for market orders (e.g. to specify quoteOrderQty)
            # This logic can be enhanced based on exchange capabilities
            price_param = order.price if order.order_type == OrderType.LIMIT else None
            if order.order_type == OrderType.MARKET and self.exchange.id == 'binance' and 'quoteOrderQty' in (self.markets.get(order.symbol, {}).get('info', {}) if self.markets else {}):
                 # Example: if using quoteOrderQty for Binance market orders
                 # This is a simplification; proper handling requires checking market capabilities
                 # if order.side == Side.BUY and order.price: # Assuming price holds quote quantity for this special case
                 #    params['quoteOrderQty'] = order.price
                 #    price_param = None # Amount is base, quoteOrderQty is quote
                 pass


            raw_order_result = await self.exchange.create_order(
                symbol=order.symbol,
                type=ccxt_order_type,
                side=ccxt_side,
                amount=order.amount,
                price=price_param,
                params=params
            )
            # logger.debug(f"Raw order result from {self.exchange_id} for {order.symbol}: {raw_order_result}")

            # Normalize the result (CCXT results can vary slightly)
            return ExecutionResult(
                order_id=str(raw_order_result['id']),
                success=True, # If create_order doesn't raise, it's usually accepted by exchange
                filled_amount=float(raw_order_result.get('filled', 0.0) or 0.0),
                average_price=float(raw_order_result.get('average', 0.0) or 0.0),
                # Fees can be complex; 'fee' object might contain cost, currency, rate
                fees=self._parse_fees(raw_order_result.get('fee') or raw_order_result.get('fees')),
                # metadata={'raw_result': raw_order_result} # Optionally include raw result
            )
        except ccxt.InsufficientFunds as e:
            # logger.error(f"Insufficient funds on {self.exchange_id} for order {order.symbol}: {e}", exc_info=True)
            return ExecutionResult(order_id="", success=False, error=f"InsufficientFunds: {e}")
        except ccxt.OrderNotFound as e: # Should not happen on create, but good to handle
            # logger.error(f"OrderNotFound on {self.exchange_id} (unexpected during create): {e}", exc_info=True)
            return ExecutionResult(order_id="", success=False, error=f"OrderNotFound: {e}")
        except ccxt.InvalidOrder as e:
            # logger.error(f"InvalidOrder on {self.exchange_id} for {order.symbol}: {e}", exc_info=True)
            return ExecutionResult(order_id="", success=False, error=f"InvalidOrder: {e}")
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            # logger.error(f"Network/ExchangeNotAvailable/Timeout error on {self.exchange_id} placing order for {order.symbol}: {e}", exc_info=True)
            # This will be caught by @with_retry, but if retries fail, it comes here.
            raise ExchangeError(f"Network/Timeout: {e}") # Re-raise as ExchangeError to be handled by circuit breaker/caller
        except ccxt.ExchangeError as e: # Catch-all for other CCXT exchange-specific errors
            # logger.error(f"Generic ExchangeError on {self.exchange_id} for {order.symbol}: {e}", exc_info=True)
            return ExecutionResult(order_id="", success=False, error=f"ExchangeError: {e}")
        except Exception as e: # Non-CCXT errors
            # logger.error(f"Unexpected exception placing order on {self.exchange_id} for {order.symbol}: {e}", exc_info=True)
            return ExecutionResult(order_id="", success=False, error=f"Unexpected error: {str(e)}")

    def _parse_fees(self, fee_data: Any) -> float:
        """Helper to parse fee information from CCXT order result."""
        if not fee_data:
            return 0.0
        if isinstance(fee_data, dict): # Common structure: {'cost': 0.1, 'currency': 'USDT', 'rate': 0.001}
            return float(fee_data.get('cost', 0.0) or 0.0)
        if isinstance(fee_data, list) and fee_data: # Sometimes it's a list of fee objects
             return sum(float(f.get('cost', 0.0) or 0.0) for f in fee_data if isinstance(f, dict))
        try: # If it's just a number (less common for create_order)
            return float(fee_data)
        except (ValueError, TypeError):
            # logger.warning(f"Could not parse fee data: {fee_data}")
            return 0.0

    @with_retry(stop_attempts=3, wait_min=1, wait_max=3)
    @with_circuit_breaker(exchange_breaker)
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        await self._init_client()
        if not self.exchange: raise ExchangeError("Exchange client not initialized.")

        # logger.info(f"Attempting to cancel order {order_id} on {self.exchange_id} for symbol {symbol}")
        try:
            # Some exchanges require symbol for cancellation, CCXT handles this if `has['cancelOrder']` is not 'emulated'
            if not symbol and self.exchange.has.get('cancelOrderRequiresSymbol', False):
                 # logger.error(f"Cancel order {order_id} on {self.exchange_id} requires a symbol, but none provided.")
                 raise ExchangeError(f"Exchange {self.exchange_id} requires symbol to cancel order {order_id}.")

            await self.exchange.cancel_order(order_id, symbol)
            # logger.info(f"Order {order_id} (Symbol: {symbol}) cancelled successfully on {self.exchange_id}.")
            return True
        except ccxt.OrderNotFound:
            # logger.warning(f"Order {order_id} (Symbol: {symbol}) not found on {self.exchange_id} during cancellation (already cancelled/filled?).")
            return False # Or True if "not found" implies it's no longer open, thus effectively "cancelled" from our POV
        except ccxt.NetworkError as e:
            # logger.error(f"Network error cancelling order {order_id} on {self.exchange_id}: {e}", exc_info=True)
            raise ExchangeError(f"Network error cancelling order: {e}")
        except ccxt.ExchangeError as e:
            # logger.error(f"Exchange error cancelling order {order_id} on {self.exchange_id}: {e}", exc_info=True)
            return False # Failed to cancel
        except Exception as e:
            # logger.error(f"Unexpected error cancelling order {order_id} on {self.exchange_id}: {e}", exc_info=True)
            return False

    @with_retry(stop_attempts=2, wait_min=1, wait_max=2)
    @with_circuit_breaker(exchange_breaker)
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, float]:
        await self._init_client()
        if not self.exchange: raise ExchangeError("Exchange client not initialized.")

        # logger.debug(f"Fetching balance for asset '{asset if asset else 'all'}' from {self.exchange_id}")
        try:
            balances_raw = await self.exchange.fetch_balance()
            # Structure of balances_raw: {'free': {}, 'used': {}, 'total': {}, 'info': ...}
            # We are interested in 'free' balances.

            free_balances = balances_raw.get('free', {})
            if not free_balances: # Some exchanges might return balances directly, or under a different key
                 # Fallback or more specific parsing might be needed if 'free' is not standard for an exchange
                 # For now, assume 'free' is the primary way
                 # logger.warning(f"No 'free' balances found in fetch_balance response from {self.exchange_id}. Response: {balances_raw}")
                 free_balances = {k: v for k, v in balances_raw.items() if isinstance(v, (int, float))}


            if asset:
                return {asset: float(free_balances.get(asset, 0.0) or 0.0)}
            else:
                return {k: float(v or 0.0) for k, v in free_balances.items() if v is not None} # Filter out None values
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            # logger.error(f"Error fetching balances from {self.exchange_id}: {e}", exc_info=True)
            raise ExchangeError(f"Error fetching balances: {e}")
        except Exception as e:
            # logger.error(f"Unexpected error fetching balances from {self.exchange_id}: {e}", exc_info=True)
            raise ExchangeError(f"Unexpected error fetching balances: {str(e)}")


    @with_retry(stop_attempts=3, wait_min=1, wait_max=3)
    @with_circuit_breaker(exchange_breaker)
    async def get_order_status(self, order_id: str, symbol: Optional[str] = None) -> Optional[ExecutionResult]:
        await self._init_client()
        if not self.exchange: raise ExchangeError("Exchange client not initialized.")

        # logger.debug(f"Fetching order status for ID {order_id}, Symbol {symbol} from {self.exchange_id}")
        try:
            if not self.exchange.has['fetchOrder']:
                # logger.warning(f"Exchange {self.exchange_id} does not support fetchOrder.")
                return None # Cannot fetch status

            # Some exchanges require symbol for fetchOrder
            if not symbol and self.exchange.has.get('fetchOrderRequiresSymbol', False):
                 # logger.error(f"Fetch order {order_id} on {self.exchange_id} requires a symbol, but none provided.")
                 raise ExchangeError(f"Exchange {self.exchange_id} requires symbol to fetch order {order_id}.")

            raw_order_data = await self.exchange.fetch_order(order_id, symbol)
            # logger.debug(f"Raw status for order {order_id} from {self.exchange_id}: {raw_order_data}")

            # Normalize CCXT order data to ExecutionResult
            # Status can be 'open', 'closed', 'canceled', 'expired', 'rejected'
            # We map this to ExecutionResult.success (True if closed/filled, False if rejected/error, ongoing if open)
            # This mapping might need refinement based on how 'success' is defined for a status check.
            # For now, let's say success=True implies fully filled and closed.
            status = raw_order_data.get('status')
            is_successful_fill = status == 'closed' and (float(raw_order_data.get('filled', 0.0) or 0.0) > 0)

            return ExecutionResult(
                order_id=str(raw_order_data['id']),
                success=is_successful_fill, # This is a simplification
                filled_amount=float(raw_order_data.get('filled', 0.0) or 0.0),
                average_price=float(raw_order_data.get('average', 0.0) or 0.0),
                fees=self._parse_fees(raw_order_data.get('fee') or raw_order_data.get('fees')),
                error=f"Status: {status}" if not is_successful_fill and status else None,
                # metadata={'raw_order_data': raw_order_data, 'status': status}
            )
        except ccxt.OrderNotFound:
            # logger.warning(f"Order {order_id} (Symbol: {symbol}) not found on {self.exchange_id} when fetching status.")
            return None # Or an ExecutionResult with error="OrderNotFound"
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            # logger.error(f"Error fetching order status for {order_id} from {self.exchange_id}: {e}", exc_info=True)
            raise ExchangeError(f"Error fetching order status: {e}")
        except Exception as e:
            # logger.error(f"Unexpected error fetching order status for {order_id} from {self.exchange_id}: {e}", exc_info=True)
            raise ExchangeError(f"Unexpected error fetching order status: {str(e)}")


    @with_retry(stop_attempts=2, wait_min=1, wait_max=2)
    @with_circuit_breaker(exchange_breaker)
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', since: Optional[int] = None, limit: Optional[int] = None) -> List[List[Any]]:
        await self._init_client()
        if not self.exchange: raise ExchangeError("Exchange client not initialized.")

        if not self.exchange.has['fetchOHLCV']:
            # logger.warning(f"Exchange {self.exchange_id} does not support fetchOHLCV.")
            return []

        # logger.debug(f"Fetching OHLCV for {symbol}, Timeframe: {timeframe}, Since: {since}, Limit: {limit} from {self.exchange_id}")
        try:
            # CCXT returns: [timestamp, open, high, low, close, volume]
            ohlcv_data = await self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            return ohlcv_data if ohlcv_data else []
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            # logger.error(f"Error fetching OHLCV for {symbol} from {self.exchange_id}: {e}", exc_info=True)
            raise ExchangeError(f"Error fetching OHLCV: {e}")
        except Exception as e: # Catch any other error
            # logger.error(f"Unexpected error fetching OHLCV for {symbol} from {self.exchange_id}: {e}", exc_info=True)
            raise ExchangeError(f"Unexpected error fetching OHLCV: {str(e)}")

    async def load_markets(self, reload: bool = False) -> Dict[str, Any]:
        # No explicit _init_client here as it might be called by other methods first,
        # or this could be the first call. If self.exchange is None, it needs init.
        if not self.exchange: # If client not created yet (e.g. direct call to load_markets)
            if not hasattr(ccxt, self.exchange_id): raise ExchangeError(f"Exchange ID '{self.exchange_id}' not found.")
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({'apiKey': self.api_key, 'secret': self.secret, 'enableRateLimit': True, **self.ccxt_options})
            if self.sandbox_mode and self.exchange.has.get('test'): self.exchange.set_sandbox_mode(True)
            # Not setting self._initialized here, as full init involves loading markets too.

        if not self.exchange: raise ExchangeError("Exchange client could not be setup for load_markets.")

        try:
            # logger.debug(f"Loading markets for {self.exchange_id} (Reload: {reload})...")
            self.markets = await self.exchange.load_markets(reload)
            # logger.info(f"Markets loaded for {self.exchange_id}: {len(self.markets) if self.markets else 0} markets found.")
            return self.markets if self.markets else {}
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            # logger.error(f"Error loading markets for {self.exchange_id}: {e}", exc_info=True)
            raise ExchangeError(f"Error loading markets: {e}")
        except Exception as e:
            # logger.error(f"Unexpected error loading markets for {self.exchange_id}: {e}", exc_info=True)
            raise ExchangeError(f"Unexpected error loading markets: {str(e)}")

    async def close(self):
        """Closes the CCXT exchange client connection."""
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                # logger.info(f"Closing CCXT client for {self.exchange_id}...")
                await self.exchange.close()
                # logger.info(f"CCXT client for {self.exchange_id} closed.")
            except Exception as e:
                # logger.error(f"Error closing CCXT client for {self.exchange_id}: {e}", exc_info=True)
                pass # Don't let close fail critically
        self.exchange = None
        self._initialized = False


# Example usage (for testing, not run by agent directly)
async def _ccxt_connector_test_main():
    # This is a very basic test and requires environment variables for a real exchange or a mock.
    # For a real test, you'd set:
    # os.environ['TEST_EXCHANGE_ID'] = 'binance' # or another exchange
    # os.environ['TEST_API_KEY'] = 'your_key'
    # os.environ['TEST_SECRET'] = 'your_secret'
    # And potentially use sandbox mode.

    # Using a mock exchange for this example as we can't make live calls here easily.
    # However, CCXT doesn't have a generic "mock" exchange.
    # We'll test with 'kraken' as it's a known public exchange, but calls will likely fail without auth for private endpoints.
    # Public endpoints like fetchOHLCV might work.

    print("--- CCXTConnector Test ---")
    params = {
        'exchange_id': 'kraken', # A public exchange, some calls might work without API keys
        # 'sandbox_mode': True, # Kraken doesn't support sandbox via set_sandbox_mode easily this way
    }
    connector = CCXTConnector(params)

    try:
        # Test loading markets (public call)
        print("\nTesting load_markets...")
        markets = await connector.load_markets(reload=True)
        assert markets is not None and len(markets) > 0
        print(f"  Loaded {len(markets)} markets from {connector.exchange_id}.")
        # print(f"  BTC/USD market info (example): {markets.get('BTC/USD') or markets.get('XBT/USD')}") # Kraken uses XBT

        # Test fetching OHLCV (public call)
        print("\nTesting fetch_ohlcv (BTC/USD)...")
        ohlcv = await connector.fetch_ohlcv('BTC/USD', timeframe='1h', limit=5) # Kraken uses XBT/USD or BTC/USD
        if not ohlcv and 'XBT/USD' in markets: # Try XBT if BTC fails
            ohlcv = await connector.fetch_ohlcv('XBT/USD', timeframe='1h', limit=5)

        assert ohlcv is not None and len(ohlcv) == 5
        print(f"  Fetched {len(ohlcv)} candles for BTC/USD (or XBT/USD). First candle: {ohlcv[0] if ohlcv else 'N/A'}")

        # Test get_balance (will likely fail or return empty without API keys)
        print("\nTesting get_balance (expected to be empty or fail gracefully)...")
        try:
            balance_btc = await connector.get_balance('BTC')
            print(f"  Balance for BTC: {balance_btc}")
            # No assertion here as it depends on whether public balance check is possible or if keys are set
        except ExchangeError as e:
            print(f"  Caught expected ExchangeError for get_balance: {e}")
        except Exception as e:
            print(f"  Unexpected error during get_balance: {e}")


        # Test place_order (will fail without API keys)
        print("\nTesting place_order (expected to fail gracefully)...")
        test_order = Order(symbol='BTC/USD', side=Side.BUY, order_type=OrderType.MARKET, amount=0.001)
        try:
            result = await connector.place_order(test_order)
            print(f"  Place order result: {result}")
            assert not result.success # Should fail without keys
        except ExchangeError as e: # This might be raised by retry/circuit breaker
            print(f"  Caught expected ExchangeError for place_order: {e}")
        except Exception as e: # Catch any other exception
            print(f"  Unexpected error during place_order: {e}")


    except ExchangeError as e:
        print(f"CCXTConnector Test Error: {e}")
    except Exception as e:
        print(f"Unexpected CCXTConnector Test Error: {e}")
    finally:
        await connector.close()
        print("\n--- CCXTConnector Test Finished ---")

if __name__ == '__main__':
    import asyncio
    # import os
    # from trading_bot.core.types import Side, OrderType, Order # Make sure these are importable
    # # Basic logging for test
    # import sys
    # logger.remove()
    # logger.add(sys.stderr, level="DEBUG")

    # To run this test, you might need to set environment variables for a test exchange
    # e.g., TEST_EXCHANGE_ID, TEST_API_KEY, TEST_SECRET
    # For now, it uses 'kraken' which has public endpoints that might work for OHLCV.
    asyncio.run(_ccxt_connector_test_main())
