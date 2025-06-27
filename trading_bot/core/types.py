from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime

class Side(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    # STOP = "stop" # From guide, can be added if BracketOrderExecution is used
    # More order types can be added as needed

class SignalSource(Enum):
    DETECTOR = "detector"
    INTERNAL = "internal"
    COMBINED = "combined"

@dataclass
class Signal:
    asset: str
    side: Side
    size: float
    confidence: float = 1.0
    source: SignalSource = SignalSource.DETECTOR
    metadata: Dict[str, Any] = field(default_factory=dict) # Ensure mutable default is handled correctly
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Removed __post_init__ as default_factory handles these cases more cleanly
    # def __post_init__(self):
    #     if self.timestamp is None: # Redundant with default_factory
    #         self.timestamp = datetime.utcnow()
    #     if self.metadata is None: # Redundant with default_factory
    #         self.metadata = {}

@dataclass
class Order:
    symbol: str
    side: Side
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # order_id: Optional[str] = None # Typically assigned by exchange, not part of initial order dataclass

@dataclass
class ExecutionResult:
    order_id: str
    success: bool
    filled_amount: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    error: Optional[str] = None
    # timestamp: datetime = field(default_factory=datetime.utcnow) # Can be useful to record when the result was processed
    # original_order: Optional[Order] = None # Could be useful for linking back
