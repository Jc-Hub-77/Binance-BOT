import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import time

from trading_bot.config.config_loader import load_config, AppConfig, RiskConfig, ExchangeConfig, StrategyConfig, ExecutionConfig, LoggingConfig
from trading_bot.core.types import Signal, Side, ExecutionResult, SignalSource, OrderType
from trading_bot.main import TradingBot # Assuming main.py is structured to allow TradingBot import
from trading_bot.modules.risk.position_limits import PositionLimitCheck # For testing risk updates
from trading_bot.modules.execution.iceberg import IcebergExecution # For time.sleep check (now asyncio.sleep)
from trading_bot.modules.exchange.simulated import SimulatedExchange # For testing simulated exchange integration

# --- Fixtures ---

@pytest.fixture
def minimal_config_dict():
    """Provides a minimal, valid configuration dictionary."""
    return {
        "mode": "paper",
        "logging": {"level": "DEBUG", "file": "logs/test_bot.log"},
        "exchange": {"class": "trading_bot.modules.exchange.simulated.SimulatedExchange", "params": {
            "initial_balances": {"USDT": 10000, "BTC": 1},
            "initial_prices": {"BTC/USDT": 50000.0}
        }},
        "strategy": {"class": "trading_bot.modules.strategy.detector_driven.DetectorDrivenStrategy", "params": {"min_confidence": 0.5}},
        "execution": {"class": "trading_bot.modules.execution.market_order.MarketOrderExecution", "params": {}},
        "risk": {
            "enabled": True,
            "checks": [
                {"class": "trading_bot.modules.risk.position_limits.PositionLimitCheck", "enabled": True, "params": {"max_position_size": 10, "max_total_positions": 5}}
            ]
        },
        "monitoring": {"prometheus_port": 8002} # Use a different port for testing
    }

@pytest.fixture
def minimal_app_config(minimal_config_dict):
    """Provides an AppConfig instance from the minimal dictionary."""
    return AppConfig(**minimal_config_dict)


@pytest.fixture
async def trading_bot(minimal_app_config):
    """Fixture to create and cleanup a TradingBot instance."""
    bot = TradingBot(minimal_app_config)
    # Perform async initialization if needed (e.g., for exchange client)
    if hasattr(bot.exchange, '_init_client'): # CCXTConnector has this
         await bot.exchange._init_client() # type: ignore
    yield bot
    # Cleanup: stop the bot if it was started
    if bot.running:
        await bot.stop()
    # Ensure exchange client is closed if it has a close method (like CCXTConnector)
    if bot.exchange and hasattr(bot.exchange, 'close'):
        await bot.exchange.close()


# --- Tests for issues from "things to check.txt" ---

# Item 1: Module Path / Typo Mismatch (Covered by ensuring config loads correct paths)
# This is implicitly tested if the bot initializes. If paths are wrong, _load_class will fail.

# Item 2: CombinedStrategy Initialization Bug (Covered in test_strategy.py for CombinedStrategy itself)

# Item 3: Mixing Sync & Async (IcebergExecution uses asyncio.sleep)
@pytest.mark.asyncio
async def test_iceberg_execution_uses_asyncio_sleep(minimal_app_config):
    """Verify IcebergExecution uses asyncio.sleep (mocked) instead of time.sleep."""
    # Modify config to use IcebergExecution
    iceberg_config_dict = minimal_app_config.model_dump() # dict() is deprecated
    iceberg_config_dict["execution"]["class"] = "trading_bot.modules.execution.iceberg.IcebergExecution"
    iceberg_config_dict["execution"]["params"] = {"chunk_size_pct": 0.5, "interval_seconds": 0.01} # Fast interval

    iceberg_app_config = AppConfig(**iceberg_config_dict)
    bot = TradingBot(iceberg_app_config)
    await bot._initialize_async_components() # Initialize exchange for executor

    assert isinstance(bot.executor, IcebergExecution)

    signal_to_execute = Signal("BTC/USDT", Side.BUY, 0.1, source=SignalSource.DETECTOR, confidence=0.9)

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_async_sleep:
        with patch('time.sleep') as mock_time_sleep: # Ensure time.sleep is NOT called
            await bot.executor.execute(signal_to_execute) # Iceberg with 2 chunks: 0.05, 0.05. 1 sleep interval.

            assert mock_async_sleep.called, "asyncio.sleep should have been called by IcebergExecution"
            # For 2 chunks, sleep is called once between them.
            if signal_to_execute.size * bot.executor.chunk_size_pct < signal_to_execute.size : # if more than 1 chunk
                 assert mock_async_sleep.call_count >= 1

            assert not mock_time_sleep.called, "time.sleep should NOT have been called"
    print("test_iceberg_execution_uses_asyncio_sleep PASSED")
    await bot.stop()


# Item 4: Risk Module Doesn't Update State
@pytest.mark.asyncio
async def test_risk_module_state_updates_after_trade(trading_bot: TradingBot):
    """Test that PositionLimitCheck's update_position is called after a successful trade."""
    # Ensure PositionLimitCheck is loaded and get an instance of it
    position_limit_check = None
    for rc in trading_bot.risk_checks:
        if isinstance(rc, PositionLimitCheck):
            position_limit_check = rc
            break
    assert position_limit_check is not None, "PositionLimitCheck not found in bot's risk_checks"

    # Mock its update_position method
    position_limit_check.update_position = MagicMock() # type: ignore

    # Simulate a signal event that leads to a successful trade
    test_signal = Signal("BTC/USDT", Side.BUY, 0.01, source=SignalSource.DETECTOR, confidence=0.9)

    # Mock exchange's place_order to return a successful result
    mock_exchange_result = ExecutionResult(order_id="sim_123", success=True, filled_amount=0.01, average_price=50000)

    # If using SimulatedExchange, it will handle this. If CCXT, need to mock.
    if isinstance(trading_bot.exchange, SimulatedExchange): # Already configured with SimulatedExchange
        pass # SimulatedExchange will provide a result
    else: # If testing with a different exchange, mock its place_order
        trading_bot.exchange.place_order = AsyncMock(return_value=mock_exchange_result) # type: ignore

    await trading_bot._handle_signal_event(MagicMock(signal=test_signal)) # Simulate event processing

    # Assert that update_position was called
    position_limit_check.update_position.assert_called_once()
    call_args = position_limit_check.update_position.call_args[0] # Get positional arguments
    assert call_args[0] == test_signal.asset
    assert call_args[1] == mock_exchange_result.filled_amount
    assert call_args[2] == test_signal.side
    print("test_risk_module_state_updates_after_trade PASSED")


# Item 5: Missing "SimulatedExchange" Implementation Gaps (Price Population)
@pytest.mark.asyncio
async def test_simulated_exchange_price_lookup(minimal_config_dict):
    """Test that SimulatedExchange uses prices provided or defaults them."""
    # Config with specific initial prices
    config_dict_prices = minimal_config_dict.copy()
    config_dict_prices["exchange"]["params"]["initial_prices"] = {"ETH/USDT": 2000.0, "LTC/USDT": 150.0}
    app_config_prices = AppConfig(**config_dict_prices)

    bot_prices = TradingBot(app_config_prices)
    assert isinstance(bot_prices.exchange, SimulatedExchange)

    # Test lookup for configured price
    eth_price = bot_prices.exchange._get_current_price("ETH/USDT") # type: ignore
    assert eth_price is not None and abs(eth_price - 2000.0) < 100 # Allowing for small random fluctuation

    # Test lookup for a non-configured but common symbol (BTC/USDT should have default from fixture)
    btc_price = bot_prices.exchange._get_current_price("BTC/USDT") # type: ignore
    assert btc_price is not None and abs(btc_price - 50000.0) < 100 # Default from minimal_config_dict

    # Test lookup for a completely unknown symbol (should return None or default, depending on impl)
    xyz_price = bot_prices.exchange._get_current_price("XYZ/USDT") # type: ignore
    assert xyz_price is None or isinstance(xyz_price, float) # Current SimulatedExchange returns None if not in common fallbacks
    print("test_simulated_exchange_price_lookup PASSED")
    await bot_prices.stop()


# Item 6: Metrics Module Assumes Dict API on Pydantic Config (Accessing prometheus_port)
# This is tested by MetricsCollector.start_server() if it correctly reads from MonitoringConfig.
# The MetricsCollector test in metrics.py itself should cover this.
# Here, we can ensure the bot passes the MonitoringConfig object.
def test_metrics_collector_receives_pydantic_config(minimal_app_config):
    bot = TradingBot(minimal_app_config)
    assert bot.metrics_collector.monitoring_config == minimal_app_config.monitoring
    # Further test that start_server uses it correctly (mock start_http_server)
    with patch('prometheus_client.start_http_server') as mock_start_http:
        bot.metrics_collector.start_server()
        if minimal_app_config.monitoring and minimal_app_config.monitoring.prometheus_port:
            mock_start_http.assert_called_with(minimal_app_config.monitoring.prometheus_port)
        else:
            assert not mock_start_http.called # Should not be called if port is None
    print("test_metrics_collector_receives_pydantic_config PASSED")


# Item 7: Missing Dependency on click (Handled by adding to pyproject.toml)
# This is an environment/setup issue, not directly testable in unit tests here,
# but CLI invocation would fail if 'click' is missing.

# Item 8: ExecutionResult Error Handling (Checking result.success in main.py)
@pytest.mark.asyncio
async def test_main_app_checks_execution_result_success(trading_bot: TradingBot, caplog):
    """Test that main._handle_signal_event checks ExecutionResult.success."""
    caplog.set_level("ERROR") # Capture ERROR logs

    test_signal = Signal("BTC/USDT", Side.BUY, 0.01, source=SignalSource.DETECTOR, confidence=0.9)

    # Mock executor to return a FAILED ExecutionResult
    failed_result = ExecutionResult(order_id="fail_123", success=False, error="Simulated execution failure")
    trading_bot.executor.execute = AsyncMock(return_value=failed_result) # type: ignore

    await trading_bot._handle_signal_event(MagicMock(signal=test_signal))

    # Check logs for error message indicating failure
    assert any("Order execution failed" in record.message and "Simulated execution failure" in record.message for record in caplog.records)
    print("test_main_app_checks_execution_result_success PASSED")

# Item 9: Graceful Shutdown Race (Signal handlers registered early in CLI)
# This is hard to test in unit tests as it involves OS signals and asyncio loop management.
# It's more of an integration or manual test for the CLI entry point.
# We can test if _signal_handler schedules bot.stop() correctly if loop is running.
@pytest.mark.asyncio
async def test_signal_handler_schedules_stop(trading_bot: TradingBot, monkeypatch):
    # Make _bot_instance global point to our test bot
    monkeypatch.setattr('trading_bot.main._bot_instance', trading_bot)

    mock_stop_task = None
    original_create_task = asyncio.create_task

    def mock_create_task(coro):
        nonlocal mock_stop_task
        if "stop" in coro.__name__: # Check if it's the stop coroutine
            mock_stop_task = coro # Capture the coroutine
            # return original_create_task(asyncio.sleep(0)) # Return a dummy completed task
            return MagicMock() # Return a mock task
        return original_create_task(coro)

    monkeypatch.setattr('asyncio.create_task', mock_create_task)

    # Simulate loop is running
    # loop = asyncio.get_event_loop() # This is not the main loop the bot runs in test typically
    # For this test, we'll assume the check `loop.is_running()` passes.
    # We can mock `asyncio.get_event_loop().is_running()` if needed.

    with patch('asyncio.get_event_loop') as mock_get_loop:
        mock_get_loop.return_value.is_running.return_value = True
        trading_bot.main._signal_handler(2, None) # Simulate SIGINT

    assert mock_stop_task is not None, "_signal_handler should have scheduled bot.stop()"
    # Further checks: assert mock_stop_task is trading_bot.stop coroutine. Difficult due to decorator wrapping.
    # Just checking if *a* task involving "stop" was created is a good indicator.
    print("test_signal_handler_schedules_stop PASSED")


# Item 10: Config Validation & Defaults (Optional RiskConfig)
def test_config_risk_section_optional():
    """Test that AppConfig allows 'risk' section to be omitted."""
    config_dict_no_risk = {
        "mode": "paper",
        "logging": {"level": "INFO", "file": "logs/app.log"},
        "exchange": {"class": "some.Exchange", "params": {}},
        "strategy": {"class": "some.Strategy", "params": {}},
        "execution": {"class": "some.Execution", "params": {}}
        # 'risk' section is omitted
    }
    try:
        app_config = AppConfig(**config_dict_no_risk)
        assert app_config.risk is None, "RiskConfig should be None when omitted"
    except Exception as e:
        assert False, f"AppConfig validation failed when 'risk' was omitted: {e}"
    print("test_config_risk_section_optional PASSED")

    # Test that if risk is provided but empty, it's also handled (e.g., risk: null in YAML)
    config_dict_null_risk = config_dict_no_risk.copy()
    config_dict_null_risk['risk'] = None
    try:
        app_config_null_risk = AppConfig(**config_dict_null_risk)
        assert app_config_null_risk.risk is None
    except Exception as e:
        assert False, f"AppConfig validation failed for risk: null : {e}"
    print("test_config_risk_section_optional (with null risk) PASSED")

    # Test with an empty risk block (e.g. risk: {} in YAML or risk: enabled: false without checks)
    config_dict_empty_risk_block = config_dict_no_risk.copy()
    config_dict_empty_risk_block['risk'] = {"enabled": False, "checks": []} # valid empty risk config
    try:
        app_config_empty_risk = AppConfig(**config_dict_empty_risk_block)
        assert app_config_empty_risk.risk is not None
        assert app_config_empty_risk.risk.enabled is False
        assert len(app_config_empty_risk.risk.checks) == 0
    except Exception as e:
        assert False, f"AppConfig validation failed for empty risk block: {e}"
    print("test_config_risk_section_optional (with empty but valid risk block) PASSED")

if __name__ == "__main__":
    pytest.main([__file__])
