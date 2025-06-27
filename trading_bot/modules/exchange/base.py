from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from trading_bot.core.types import Order, ExecutionResult, Signal # Added Signal for context

class Exchange(ABC):
    """
    Abstract Base Class for all exchange connectors.
    This class defines the standard interface for interacting with different
    cryptocurrency exchanges, whether real or simulated.
    """

    def __init__(self, params: Dict[str, Any]): # Changed from config to params for consistency
        """
        Initializes the Exchange connector.

        Args:
            params: A dictionary of parameters specific to this exchange,
                    loaded from the configuration file (e.g., API keys, exchange name).
        """
        self.params = params
        self.client = None # Placeholder for the actual exchange client (e.g., ccxt instance)
        # logger.debug(f"Exchange '{self.__class__.__name__}' initialized with params: {params}")

    @abstractmethod
    async def place_order(self, order: Order) -> ExecutionResult: # Changed to async
        """
        Place an order on the exchange.

        Args:
            order: An Order object containing the details of the order to be placed.

        Returns:
            An ExecutionResult object detailing the outcome of the order placement.
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool: # Changed to async, made symbol optional
        """
        Cancel an existing order on the exchange.

        Args:
            order_id: The unique identifier of the order to be cancelled.
            symbol: The trading symbol (e.g., "BTC/USDT"). Some exchanges require this.

        Returns:
            True if the order was successfully cancelled, False otherwise.
        """
        pass

    @abstractmethod
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, float]: # Changed to async, asset optional for all balances
        """
        Fetch the balance for a specific asset or all assets from the exchange.

        Args:
            asset: The ticker symbol of the asset (e.g., "BTC", "USDT").
                   If None, should return all available balances.

        Returns:
            A dictionary where keys are asset symbols and values are their free balances.
            Example: {'BTC': 0.5, 'USDT': 1000.0}
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: Optional[str] = None) -> Optional[ExecutionResult]: # Changed to async
        """
        Fetch the current status of a specific order.

        Args:
            order_id: The unique identifier of the order.
            symbol: The trading symbol. Some exchanges require this.

        Returns:
            An ExecutionResult object representing the current state of the order,
            or None if the order is not found or an error occurs.
        """
        pass

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', since: Optional[int] = None, limit: Optional[int] = None) -> List[List[Any]]:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").
            timeframe: The length of time each candle represents (e.g., '1m', '5m', '1h', '1d').
            since: Timestamp in milliseconds to fetch OHLCV data from.
            limit: The maximum number of candles to fetch.

        Returns:
            A list of lists, where each inner list represents a candle:
            [timestamp, open, high, low, close, volume]
        """
        pass

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetch the current market price for a given symbol.
        This can be implemented using fetch_ticker or by fetching recent trades.
        Default implementation fetches the last price from OHLCV data (not ideal for live trading).
        Subclasses should provide a more robust implementation.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").

        Returns:
            The current price as a float, or None if unable to fetch.
        """
        try:
            # A more robust implementation would use fetch_ticker or fetch_trades
            ohlcv = await self.fetch_ohlcv(symbol, timeframe='1m', limit=1)
            if ohlcv and len(ohlcv) > 0:
                return float(ohlcv[0][4]) # Close price of the last 1m candle
            return None
        except Exception:
            # logger.error(f"Could not fetch current price for {symbol} via default get_current_price: {e}")
            return None

    async def load_markets(self, reload: bool = False) -> Dict[str, Any]:
        """
        Load market data from the exchange. This is often needed to understand
        trading pairs, precision, limits, etc.
        This is a common CCXT method, so it's good to have in the interface.

        Args:
            reload: Whether to force a reload of market data from the exchange.

        Returns:
            A dictionary of market data.
        """
        # Default implementation, subclasses (like CCXTConnector) will override.
        return {}

    async def close(self):
        """
        Optional method to clean up resources, like closing network connections.
        Important for graceful shutdown, especially for async clients (e.g., aiohttp sessions).
        """
        # Default implementation does nothing.
        pass

    # --- Methods from "order bot guide v2.txt" that could be part of the interface ---
    # These are more specific and might not be universally applicable to all exchange types
    # or might be part of specific execution strategies rather than the raw exchange interface.

    # async def set_leverage(self, symbol: str, leverage: int): # Typically for futures
    #     """ Sets leverage for a symbol. """
    #     raise NotImplementedError

    # async def make_trade_dex(self, input_token: str, output_token: str, qty: float): # Specific to DEX
    #     """ Executes a trade on a DEX. """
    #     raise NotImplementedError
