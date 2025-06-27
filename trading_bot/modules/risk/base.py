from abc import ABC, abstractmethod
from typing import Any, Dict # For metrics method
from trading_bot.core.types import Signal, Side # Added Side for update_position

class RiskCheck(ABC):
    """
    Abstract Base Class for all risk management checks.
    Risk checks are applied to signals before they are executed to ensure
    they comply with predefined risk parameters.
    """
    def __init__(self, **kwargs): # Added **kwargs
        """
        Initializes the RiskCheck.
        Args:
            **kwargs: Allows for additional parameters from configuration to be accepted.
        """
        # Example: self.check_name = kwargs.get('name', self.__class__.__name__)
        # logger.debug(f"RiskCheck '{self.check_name}' initialized with kwargs: {kwargs}")
        pass

    @abstractmethod
    def validate(self, signal: Signal) -> Signal:
        """
        Validate a trading signal against this risk check's criteria.

        If the signal violates the risk criteria, this method should either:
        1. Modify the signal to be compliant (e.g., reduce order size).
        2. Raise a ValueError to prevent the trade.

        Args:
            signal: The Signal object to validate.

        Returns:
            The (potentially modified) Signal object if it's compliant.

        Raises:
            ValueError: If the signal violates risk criteria and cannot be made compliant.
        """
        pass

    def update_position(self, asset: str, size: float, side: Side, price: float = 0.0, **kwargs):
        """
        Optional method for risk checks that need to maintain state about current positions.
        This method would be called after a trade is successfully executed.
        Not all risk checks need to implement this.

        Args:
            asset: The asset symbol (e.g., "BTC/USDT").
            size: The size of the executed trade.
            side: The side of the trade (BUY or SELL).
            price: The execution price of the trade.
            **kwargs: Additional information about the trade.
        """
        # Default implementation does nothing. Subclasses can override.
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        Optional method to return current metrics related to this risk check.
        Useful for monitoring and debugging.

        Returns:
            A dictionary of metrics.
        """
        # Default implementation returns an empty dictionary.
        return {}

    def reset_state(self):
        """
        Optional method to reset any internal state of the risk check.
        Useful for backtesting or restarting the bot.
        """
        # Default implementation does nothing.
        pass
