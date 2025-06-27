from abc import ABC, abstractmethod
from trading_bot.core.types import Signal, ExecutionResult, Order # Added Order
from trading_bot.modules.exchange.base import Exchange # Assuming Exchange base class is defined

class ExecutionStrategy(ABC):
    """
    Abstract Base Class for all order execution strategies.
    Execution strategies define HOW an order is placed on an exchange
    (e.g., market order, limit order, iceberg order).
    """
    def __init__(self, exchange: Exchange, **kwargs): # Added **kwargs
        """
        Initializes the ExecutionStrategy.

        Args:
            exchange: An instance of an Exchange interface, used to interact with the trading venue.
            **kwargs: Allows for additional parameters from configuration to be accepted.
        """
        self.exchange = exchange
        # Example: self.default_timeout = kwargs.get('timeout', 30) # seconds
        # logger.debug(f"ExecutionStrategy '{self.__class__.__name__}' initialized with kwargs: {kwargs}")


    @abstractmethod
    async def execute(self, signal: Signal) -> ExecutionResult: # Changed to async
        """
        Execute a trading signal by placing an order on the exchange.

        This method should translate a Signal into one or more Orders and manage their execution.
        It must be implemented as an asynchronous method if it involves I/O operations
        like network calls to an exchange, especially if using asyncio.sleep for strategies
        like Iceberg.

        Args:
            signal: The Signal object to be executed.

        Returns:
            An ExecutionResult object detailing the outcome of the order placement.
        """
        pass

    # Optional: Helper method to create an Order from a Signal, could be common logic
    # def _create_order_from_signal(self, signal: Signal, order_type: OrderType) -> Order:
    #     """
    #     Creates an Order object from a Signal.
    #     """
    #     return Order(
    #         symbol=signal.asset,
    #         side=signal.side,
    #         order_type=order_type,
    #         amount=signal.size,
    #         metadata=signal.metadata.copy() # Ensure metadata is copied
    #     )

    # Optional: A cancel method if execution strategies manage outstanding orders directly
    # @abstractmethod
    # async def cancel_order(self, order_id: str, symbol: str) -> bool:
    #     """
    #     Cancel a previously placed order.
    #     Args:
    #         order_id: The ID of the order to cancel.
    #         symbol: The trading symbol of the order.
    #     Returns:
    #         True if cancellation was successful, False otherwise.
    #     """
    #     pass
