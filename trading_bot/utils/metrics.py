from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
from prometheus_client.exposition import generate_latest
# from loguru import logger
from typing import Optional, Dict, Any

from trading_bot.config.config_loader import MonitoringConfig # For type hinting config

# --- Prometheus Metrics Definitions ---
# It's good practice to define them globally or in a way that they are singletons.
# Using a prefix for all metrics specific to this application.
METRIC_PREFIX = "trading_bot_"

# Order Tracking
ORDERS_PLACED_TOTAL = Counter(
    METRIC_PREFIX + 'orders_placed_total',
    'Total number of orders placed.',
    ['exchange', 'symbol', 'side', 'order_type', 'strategy'] # Added order_type and strategy
)
ORDERS_SUCCEEDED_TOTAL = Counter(
    METRIC_PREFIX + 'orders_succeeded_total',
    'Total number of orders that executed successfully (fully or partially).',
    ['exchange', 'symbol', 'side', 'order_type', 'strategy']
)
ORDERS_FAILED_TOTAL = Counter(
    METRIC_PREFIX + 'orders_failed_total',
    'Total number of orders that failed.',
    ['exchange', 'symbol', 'side', 'order_type', 'strategy', 'reason']
)
ORDER_LATENCY_SECONDS = Histogram(
    METRIC_PREFIX + 'order_latency_seconds',
    'Latency of order placement and confirmation from the exchange.',
    ['exchange', 'symbol', 'order_type', 'strategy'],
    buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")] # Example buckets
)
ORDER_FILL_AMOUNT = Histogram( # New: Track fill amounts (e.g. for partial fills)
    METRIC_PREFIX + 'order_fill_amount_total', # Using _total as it's a sum over time, though Histogram is more complex
    'Amount filled for orders (in base currency).',
    ['exchange', 'symbol', 'side', 'order_type', 'strategy'],
    # Buckets could represent typical order sizes or relative fill % if normalized
)

# Portfolio / Balance Tracking
PORTFOLIO_ASSET_BALANCE = Gauge(
    METRIC_PREFIX + 'portfolio_asset_balance',
    'Current balance of a specific asset.',
    ['exchange', 'asset_symbol']
)
PORTFOLIO_TOTAL_VALUE_USD = Gauge( # Assuming USD as the quote currency for total value
    METRIC_PREFIX + 'portfolio_total_value_usd',
    'Estimated total value of the portfolio in USD.',
    ['exchange']
)

# Signal Processing
SIGNALS_RECEIVED_TOTAL = Counter(
    METRIC_PREFIX + 'signals_received_total',
    'Total number of trading signals received.',
    ['source', 'asset', 'side'] # Source could be 'detector', 'internal', 'combined'
)
SIGNALS_PROCESSED_TOTAL = Counter(
    METRIC_PREFIX + 'signals_processed_total',
    'Total number of signals processed by strategies.',
    ['strategy_name', 'asset', 'side', 'outcome'] # Outcome: 'accepted', 'rejected', 'modified'
)

# Risk Management
RISK_CHECKS_TRIGGERED_TOTAL = Counter(
    METRIC_PREFIX + 'risk_checks_triggered_total',
    'Total number of times a specific risk check was triggered.',
    ['risk_check_name', 'asset', 'outcome'] # Outcome: 'passed', 'adjusted', 'blocked'
)

# Application Health
UPTIME_SECONDS = Gauge( # This would need to be updated periodically by the main loop
    METRIC_PREFIX + 'uptime_seconds',
    'Bot uptime in seconds.'
)
EVENT_BUS_QUEUE_SIZE = Gauge( # If event bus has a queue
    METRIC_PREFIX + 'event_bus_queue_size',
    'Number of events currently in the event bus queue.'
)


class MetricsCollector:
    """
    A centralized class for managing and exposing Prometheus metrics.
    It provides methods to update various metrics related to trading bot operations.
    """
    _instance = None
    _server_started = False

    def __new__(cls, monitoring_config: Optional[MonitoringConfig] = None):
        # Singleton pattern to ensure only one MetricsCollector and one Prometheus server
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance.monitoring_config = monitoring_config
            # logger.info("MetricsCollector instance created.")
        return cls._instance

    def __init__(self, monitoring_config: Optional[MonitoringConfig] = None):
        # The __init__ will be called every time MetricsCollector() is invoked,
        # but the actual instance will be the same due to __new__.
        # We only need to assign config if it hasn't been or is being updated.
        if not hasattr(self, 'initialized'): # Ensure this part runs only once per instance
            self.monitoring_config = monitoring_config or getattr(self, 'monitoring_config', None)
            self.initialized = True

    def start_server(self):
        """
        Starts the Prometheus HTTP server if a port is configured and server not already started.
        """
        if MetricsCollector._server_started:
            # logger.info("Prometheus metrics server already running.")
            return

        if self.monitoring_config and self.monitoring_config.prometheus_port is not None:
            # Fix for item 6 in "things to check.txt": Access Pydantic model correctly
            port = self.monitoring_config.prometheus_port
            try:
                start_http_server(port)
                MetricsCollector._server_started = True
                # logger.info(f"Prometheus metrics server started on port {port}.")
            except OSError as e: # Handle cases like port already in use
                # logger.error(f"Failed to start Prometheus metrics server on port {port}: {e}", exc_info=True)
                # Potentially try another port or disable metrics server
                MetricsCollector._server_started = False # Ensure it's marked as not started
            except Exception as e:
                # logger.error(f"An unexpected error occurred while starting Prometheus server: {e}", exc_info=True)
                MetricsCollector._server_started = False
        else:
            # logger.warning("Prometheus port not configured in MonitoringConfig. Metrics server not started.")
            pass

    # --- Methods to update metrics ---

    @staticmethod
    def track_signal_received(source: str, asset: str, side: str):
        SIGNALS_RECEIVED_TOTAL.labels(source=source, asset=asset, side=side).inc()

    @staticmethod
    def track_signal_processed(strategy_name: str, asset: str, side: str, outcome: str):
        SIGNALS_PROCESSED_TOTAL.labels(strategy_name=strategy_name, asset=asset, side=side, outcome=outcome).inc()

    @staticmethod
    def track_order_placed(exchange: str, symbol: str, side: str, order_type: str, strategy: str):
        ORDERS_PLACED_TOTAL.labels(exchange=exchange, symbol=symbol, side=side, order_type=order_type, strategy=strategy).inc()

    @staticmethod
    def track_order_result(
        exchange: str, symbol: str, side: str, order_type: str, strategy: str,
        success: bool, duration_seconds: Optional[float] = None,
        filled_amount: Optional[float] = None, reason_failed: Optional[str] = "unknown"
    ):
        if success:
            ORDERS_SUCCEEDED_TOTAL.labels(exchange=exchange, symbol=symbol, side=side, order_type=order_type, strategy=strategy).inc()
            if filled_amount is not None and filled_amount > 0:
                 ORDER_FILL_AMOUNT.labels(exchange=exchange, symbol=symbol, side=side, order_type=order_type, strategy=strategy).observe(filled_amount)
        else:
            ORDERS_FAILED_TOTAL.labels(exchange=exchange, symbol=symbol, side=side, order_type=order_type, strategy=strategy, reason=reason_failed).inc()

        if duration_seconds is not None:
            ORDER_LATENCY_SECONDS.labels(exchange=exchange, symbol=symbol, order_type=order_type, strategy=strategy).observe(duration_seconds)

    @staticmethod
    def track_risk_check(risk_check_name: str, asset: str, outcome: str):
        RISK_CHECKS_TRIGGERED_TOTAL.labels(risk_check_name=risk_check_name, asset=asset, outcome=outcome).inc()

    @staticmethod
    def update_asset_balance(exchange: str, asset_symbol: str, balance: float):
        PORTFOLIO_ASSET_BALANCE.labels(exchange=exchange, asset_symbol=asset_symbol).set(balance)

    @staticmethod
    def update_portfolio_total_value_usd(exchange: str, total_value_usd: float):
        PORTFOLIO_TOTAL_VALUE_USD.labels(exchange=exchange).set(total_value_usd)

    @staticmethod
    def update_uptime(uptime_val: float):
        UPTIME_SECONDS.set(uptime_val)

    @staticmethod
    def update_event_bus_queue_size(size: int):
        EVENT_BUS_QUEUE_SIZE.set(size)

    @staticmethod
    def get_metrics_exposition() -> str:
        """Returns the metrics in Prometheus exposition format."""
        return generate_latest(REGISTRY).decode('utf-8')


if __name__ == '__main__':
    # Example Usage:
    # logger.remove() # Clear default loggers for test
    # logger.add(sys.stderr, level="DEBUG")

    print("--- MetricsCollector Test ---")

    # Mock MonitoringConfig
    class MockMonitoringConfig(MonitoringConfig):
        prometheus_port: Optional[int] = 8001 # Use a different port for testing

    test_monitoring_config = MockMonitoringConfig()

    # Initialize collector (singleton)
    collector1 = MetricsCollector(test_monitoring_config)
    collector2 = MetricsCollector() # Should be the same instance

    assert collector1 is collector2, "MetricsCollector should be a singleton."
    print(f"Collector instance ID: {id(collector1)}")

    # Start the server (only once)
    collector1.start_server() # First call starts
    collector1.start_server() # Second call should notice it's already started

    # Simulate some activities
    print("\nSimulating bot activities to update metrics...")
    MetricsCollector.track_signal_received(source="detector", asset="BTC/USDT", side="buy")
    MetricsCollector.track_signal_processed(strategy_name="CombinedStrategy", asset="BTC/USDT", side="buy", outcome="accepted")

    MetricsCollector.track_order_placed(exchange="binance", symbol="BTC/USDT", side="buy", order_type="market", strategy="CombinedStrategy")
    MetricsCollector.track_order_result(
        exchange="binance", symbol="BTC/USDT", side="buy", order_type="market", strategy="CombinedStrategy",
        success=True, duration_seconds=0.55, filled_amount=0.1
    )

    MetricsCollector.track_order_placed(exchange="kraken", symbol="ETH/USD", side="sell", order_type="limit", strategy="Arbitrage")
    MetricsCollector.track_order_result(
        exchange="kraken", symbol="ETH/USD", side="sell", order_type="limit", strategy="Arbitrage",
        success=False, duration_seconds=0.20, reason_failed="insufficient_funds"
    )

    MetricsCollector.track_risk_check(risk_check_name="PositionLimitCheck", asset="BTC/USDT", outcome="passed")
    MetricsCollector.track_risk_check(risk_check_name="DrawdownProtection", asset="PORTFOLIO", outcome="triggered")

    MetricsCollector.update_asset_balance(exchange="binance", asset_symbol="BTC", balance=0.5)
    MetricsCollector.update_asset_balance(exchange="binance", asset_symbol="USDT", balance=10000)
    MetricsCollector.update_portfolio_total_value_usd(exchange="binance", total_value_usd=25000)
    MetricsCollector.update_uptime(3600) # 1 hour

    print("\nMetrics updated. Check Prometheus server at http://localhost:8001/metrics")

    # Test getting exposition format
    exposition_data = MetricsCollector.get_metrics_exposition()
    assert "trading_bot_orders_placed_total" in exposition_data
    assert "trading_bot_portfolio_asset_balance" in exposition_data
    assert 'trading_bot_orders_failed_total{exchange="kraken",order_type="limit",reason="insufficient_funds",side="sell",strategy="Arbitrage",symbol="ETH/USD"} 1.0' in exposition_data
    print("\nSuccessfully retrieved metrics exposition data.")
    # print("\n--- Sample Exposition Data ---")
    # print(exposition_data[:1000] + "\n...") # Print a sample

    # Note: The Prometheus server runs in a separate thread.
    # To stop it for cleanup in a real test suite, you might need more complex handling
    # or just let the process exit. For this example, we'll let it run.
    print("\nTo stop the test, manually terminate the script (Ctrl+C).")

    # Keep the main thread alive for a bit to allow checking the /metrics endpoint
    try:
        import time
        time.sleep(15) # Keep alive for 15 seconds
    except KeyboardInterrupt:
        print("\nExiting test.")
    finally:
        # In a real app, you might want a way to shut down the Prometheus server thread.
        # `start_http_server` doesn't return a server object to call `shutdown()` on directly.
        # One common way is to run it in a daemon thread that exits with the main program.
        # Or, for tests, you might use a context manager if a library provides one.
        # For now, it will stop when the script ends.
        # logger.info("MetricsCollector test finished.")
        pass
