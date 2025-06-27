import asyncio
import signal as os_signal # Renamed to avoid conflict with core.types.Signal
import importlib
import time # For order duration tracking
from typing import Optional, Any, Type # Added Type for _load_class

import click
from loguru import logger # Logger will be configured by setup_logger

# Configuration and Core Components
from trading_bot.config.config_loader import AppConfig, load_config
from trading_bot.utils.logger import setup_logger
from trading_bot.utils.metrics import MetricsCollector
from trading_bot.core.events import EventBus, SignalEvent, OrderEvent # Assuming OrderEvent might be useful
from trading_bot.core.types import Signal as CoreSignal, ExecutionResult, Side # Aliased Signal to CoreSignal

# Module Base Classes (for type hinting)
from trading_bot.modules.exchange.base import Exchange
from trading_bot.modules.strategy.base import Strategy as BaseStrategy # Aliased Strategy to BaseStrategy
from trading_bot.modules.execution.base import ExecutionStrategy as BaseExecutionStrategy # Aliased
from trading_bot.modules.risk.base import RiskCheck

class TradingBot:
    """
    The main application class for the trading bot.
    It initializes and manages all components like exchange connectors,
    strategies, execution engines, risk management, and event handling.
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.app_logger = setup_logger(config.logging) # setup_logger returns the logger instance
        logger.info(f"TradingBot initializing with mode: {config.mode}")

        self.event_bus = EventBus()
        self.metrics_collector = MetricsCollector(config.monitoring) # Pass monitoring part of config

        self.exchange: Optional[Exchange] = None
        self.strategy: Optional[BaseStrategy] = None
        self.executor: Optional[BaseExecutionStrategy] = None
        self.risk_checks: List[RiskCheck] = []

        self._init_modules()
        self._setup_event_handlers()

        self.running = False
        self._main_task: Optional[asyncio.Task] = None
        self._start_time = time.time()


    def _load_class(self, class_path: str) -> Type[Any]:
        """Dynamically loads a class from a given module path."""
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load class {class_path}: {e}", exc_info=True)
            raise # Reraise to stop bot initialization if a critical module fails

    def _init_modules(self):
        """Initializes all pluggable modules based on the configuration."""
        logger.info("Initializing bot modules...")

        # Exchange
        exchange_cfg = self.config.exchange
        logger.debug(f"Loading exchange module: {exchange_cfg.class_path} with params: {exchange_cfg.params}")
        exchange_class = self._load_class(exchange_cfg.class_path)
        self.exchange = exchange_class(exchange_cfg.params)
        # Exchange initialization (like loading markets) might be async, handle that in an async setup method if needed.

        # Strategy
        # Addressing item 1 in "things to check.txt" - module path typo
        # The config itself should be correct. If a default config had a typo, it's fixed there.
        # This loader just uses what's in AppConfig.
        strategy_cfg = self.config.strategy
        logger.debug(f"Loading strategy module: {strategy_cfg.class_path} with params: {strategy_cfg.params}")
        strategy_class = self._load_class(strategy_cfg.class_path)
        self.strategy = strategy_class(event_bus=self.event_bus, **strategy_cfg.params)

        # Execution
        exec_cfg = self.config.execution
        logger.debug(f"Loading execution module: {exec_cfg.class_path} with params: {exec_cfg.params}")
        exec_class = self._load_class(exec_cfg.class_path)
        if not self.exchange: raise ValueError("Exchange must be initialized before execution module.")
        self.executor = exec_class(exchange=self.exchange, **exec_cfg.params)

        # Risk Checks
        if self.config.risk and self.config.risk.enabled:
            logger.info(f"Initializing {len(self.config.risk.checks)} risk checks...")
            for risk_cfg in self.config.risk.checks:
                if risk_cfg.enabled: # Check if individual risk check is enabled
                    logger.debug(f"Loading risk check: {risk_cfg.class_path} with params: {risk_cfg.params}")
                    risk_class = self._load_class(risk_cfg.class_path)
                    risk_check_instance = risk_class(**risk_cfg.params)
                    self.risk_checks.append(risk_check_instance)
                else:
                    logger.info(f"Skipping disabled risk check: {risk_cfg.class_path}")
        else:
            logger.info("Risk management is disabled or no checks configured.")

        logger.info("All core modules initialized.")


    def _setup_event_handlers(self):
        """Subscribes methods to relevant events on the event bus."""
        self.event_bus.subscribe(SignalEvent, self._handle_signal_event)
        # Potentially subscribe to OrderEvents if further action is needed after execution reporting
        # self.event_bus.subscribe(OrderEvent, self._handle_order_event)
        logger.info("Event handlers set up.")

    async def _handle_signal_event(self, event: SignalEvent):
        """Handles incoming SignalEvents from the event bus."""
        signal = event.signal # This is CoreSignal type
        logger.info(f"Received signal event for {signal.asset} {signal.side.value} from {signal.source.value}")

        # Metrics for received signal
        self.metrics_collector.track_signal_received(
            source=signal.source.value, asset=signal.asset, side=signal.side.value
        )

        if not self.strategy or not self.executor or not self.exchange:
            logger.error("Strategy, Executor, or Exchange not initialized. Cannot process signal.")
            return

        try:
            # 1. Process signal through strategy
            # Assuming strategy.process_signal might be async if it does internal I/O
            # For now, assuming it's sync as per base.py, but can be changed.
            # If it becomes async: processed_signals = await self.strategy.process_signal(signal)
            processed_signals = self.strategy.process_signal(signal)
            if not processed_signals:
                logger.debug(f"Strategy {self.strategy.__class__.__name__} produced no actionable signals from input.")
                self.metrics_collector.track_signal_processed(
                     strategy_name=self.strategy.__class__.__name__, asset=signal.asset, side=signal.side.value, outcome="rejected_by_strategy"
                )
                return

            for proc_signal in processed_signals:
                logger.debug(f"Strategy output signal: {proc_signal.asset} {proc_signal.side.value} Size: {proc_signal.size}")

                # 2. Apply risk checks
                validated_signal = self._apply_risk_checks(proc_signal)
                if not validated_signal:
                    # Risk check blocked or modified to non-actionable
                    self.metrics_collector.track_signal_processed(
                        strategy_name=self.strategy.__class__.__name__, asset=proc_signal.asset, side=proc_signal.side.value, outcome="blocked_by_risk"
                    )
                    continue # Move to next processed signal if any

                self.metrics_collector.track_signal_processed(
                    strategy_name=self.strategy.__class__.__name__, asset=validated_signal.asset, side=validated_signal.side.value,
                    outcome="accepted_for_execution" if validated_signal.size == proc_signal.size else "modified_by_risk"
                )

                # 3. Execute trade
                logger.info(f"Executing validated signal: {validated_signal.asset} {validated_signal.side.value} {validated_signal.size}")

                self.metrics_collector.track_order_placed(
                    exchange=self.exchange.__class__.__name__, symbol=validated_signal.asset,
                    side=validated_signal.side.value, order_type="unknown", # Infer order type from executor if possible
                    strategy=self.strategy.__class__.__name__
                )

                start_time = time.monotonic()
                # Executor.execute should be async
                execution_result: ExecutionResult = await self.executor.execute(validated_signal)
                duration_seconds = time.monotonic() - start_time

                logger.info(f"Execution result for {validated_signal.asset}: {execution_result}")

                # Metrics for order result
                self.metrics_collector.track_order_result(
                    exchange=self.exchange.__class__.__name__, symbol=validated_signal.asset,
                    side=validated_signal.side.value, order_type=getattr(execution_result, 'order_type', 'unknown'), # Get order type if available
                    strategy=self.strategy.__class__.__name__,
                    success=execution_result.success,
                    duration_seconds=duration_seconds,
                    filled_amount=execution_result.filled_amount,
                    reason_failed=execution_result.error if not execution_result.success else None
                )

                # Addressing item 8 from "things to check.txt": Check result.success
                if not execution_result.success:
                    logger.error(f"Order execution failed for {validated_signal.asset}: {execution_result.error}")
                    # Potentially publish a failure event or take other actions
                else:
                    logger.info(f"Order for {validated_signal.asset} executed. ID: {execution_result.order_id}, Filled: {execution_result.filled_amount} @ {execution_result.average_price}")
                    # Addressing item 4 from "things to check.txt": Update risk module state
                    self._update_risk_module_positions(validated_signal, execution_result)

                # Publish OrderEvent (optional, if other components need to know about order outcomes)
                # order_event = OrderEvent(order=None, result=execution_result) # Need original order object for OrderEvent
                # self.event_bus.publish(order_event)


        except Exception as e:
            logger.error(f"Error processing signal event for {signal.asset}: {e}", exc_info=True)
            self.metrics_collector.track_signal_processed(
                strategy_name=self.strategy.__class__.__name__ if self.strategy else "UnknownStrategy",
                asset=signal.asset, side=signal.side.value, outcome="processing_error"
            )


    def _apply_risk_checks(self, signal: CoreSignal) -> Optional[CoreSignal]:
        """Applies all configured risk checks to a signal."""
        current_signal = signal
        for risk_check in self.risk_checks:
            try:
                logger.debug(f"Applying risk check '{risk_check.__class__.__name__}' to signal for {current_signal.asset}")
                current_signal = risk_check.validate(current_signal)
                if not current_signal: # Should not happen if validate raises or returns modified signal
                    logger.warning(f"Risk check {risk_check.__class__.__name__} returned None. Blocking signal for {signal.asset}.")
                    self.metrics_collector.track_risk_check(risk_check_name=risk_check.__class__.__name__, asset=signal.asset, outcome="blocked_null_signal")
                    return None

                # Check if signal size was modified (indicates adjustment)
                original_size = signal.metadata.get('original_signal_size', signal.size) # Assuming original_signal_size is stored if modified
                if current_signal.size < original_size :
                     self.metrics_collector.track_risk_check(risk_check_name=risk_check.__class__.__name__, asset=signal.asset, outcome="adjusted")
                else: # Passed without adjustment by this specific check
                     self.metrics_collector.track_risk_check(risk_check_name=risk_check.__class__.__name__, asset=signal.asset, outcome="passed")


            except ValueError as e: # Risk check explicitly blocked the signal
                logger.warning(f"Risk check {risk_check.__class__.__name__} blocked signal for {signal.asset}: {e}")
                self.metrics_collector.track_risk_check(risk_check_name=risk_check.__class__.__name__, asset=signal.asset, outcome="blocked_exception")
                return None # Signal is blocked
            except Exception as e:
                logger.error(f"Unexpected error in risk check {risk_check.__class__.__name__} for {signal.asset}: {e}", exc_info=True)
                self.metrics_collector.track_risk_check(risk_check_name=risk_check.__class__.__name__, asset=signal.asset, outcome="error")
                return None # Block signal on unexpected error in risk check
        return current_signal

    def _update_risk_module_positions(self, signal_executed: CoreSignal, exec_result: ExecutionResult):
        """Updates position information in risk modules after a successful trade."""
        if exec_result.success and exec_result.filled_amount > 0:
            for rc in self.risk_checks:
                if hasattr(rc, "update_position"):
                    try:
                        logger.debug(f"Updating position in risk check '{rc.__class__.__name__}' for {signal_executed.asset}")
                        rc.update_position(
                            asset=signal_executed.asset,
                            size=exec_result.filled_amount, # Use actual filled amount
                            side=signal_executed.side, # Original intended side
                            price=exec_result.average_price
                        )
                    except Exception as e:
                        logger.error(f"Error updating position in risk check {rc.__class__.__name__}: {e}", exc_info=True)

    async def _periodic_tasks(self):
        """Placeholder for tasks that run periodically, like updating metrics or market data analysis."""
        if not self.strategy or not self.exchange: return

        while self.running:
            try:
                # 1. Market Analysis by Strategy (if strategy supports it)
                # This part needs a way to get market data.
                # For now, let's assume self.exchange can provide some basic form or it's mocked.
                # market_data_sample = {"BTC/USDT": {"prices": [await self.exchange.get_current_price("BTC/USDT")]}}
                # internal_signals = self.strategy.analyze_market(market_data_sample) # Assuming sync for now
                # for signal in internal_signals:
                #    self.event_bus.publish(SignalEvent(signal))

                # 2. Update portfolio metrics (example)
                if self.config.mode != 'backtest': # Don't fetch live balances in backtest
                    all_balances = await self.exchange.get_balance()
                    for asset, bal_val in all_balances.items():
                        self.metrics_collector.update_asset_balance(
                            exchange=self.exchange.__class__.__name__, asset_symbol=asset, balance=bal_val
                        )
                    # TODO: Calculate total portfolio value in USD for metrics_collector.update_portfolio_total_value_usd

                # 3. Update uptime metric
                self.metrics_collector.update_uptime(time.time() - self._start_time)

                await asyncio.sleep(self.config.monitoring.get('report_interval_seconds', 60)
                                    if self.config.monitoring else 60) # Default to 60s if not in config
            except asyncio.CancelledError:
                logger.info("Periodic tasks cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in periodic tasks: {e}", exc_info=True)
                await asyncio.sleep(60) # Wait before retrying periodic tasks on error


    async def _initialize_async_components(self):
        """Initialize components that require async operations, like CCXT client."""
        if self.exchange and hasattr(self.exchange, '_init_client'):
            logger.info("Initializing async components for exchange...")
            await self.exchange._init_client() # type: ignore # Assuming _init_client is the async setup method
        logger.info("Async component initialization complete.")


    async def run(self):
        """Starts the trading bot's main event loop and periodic tasks."""
        if self.running:
            logger.warning("Bot is already running.")
            return

        self.running = True
        logger.info("Trading bot starting...")
        self.metrics_collector.start_server() # Start Prometheus server

        await self._initialize_async_components() # Initialize async parts of components

        # Start periodic tasks in the background
        self._periodic_task_handle = asyncio.create_task(self._periodic_tasks())

        logger.info("Trading bot is now running. Waiting for signals or events...")
        # The bot primarily reacts to events. The main loop here can just keep alive.
        # Or, if there are other primary loop activities, they go here.
        try:
            while self.running:
                # Main loop can be simple if event-driven.
                # For example, check health, or just sleep.
                await asyncio.sleep(1) # Keep alive, check self.running flag
        except asyncio.CancelledError:
            logger.info("Main bot run loop cancelled.")
        finally:
            logger.info("Bot run loop ended.")
            # Ensure periodic tasks are also stopped if main loop exits unexpectedly
            if self._periodic_task_handle and not self._periodic_task_handle.done():
                self._periodic_task_handle.cancel()
                try:
                    await self._periodic_task_handle
                except asyncio.CancelledError:
                    pass # Expected


    async def stop(self):
        """Stops the trading bot gracefully."""
        if not self.running:
            logger.info("Bot is not running or already stopping.")
            return

        logger.info("Trading bot stopping...")
        self.running = False # Signal loops to stop

        # Cancel main task if it's structured that way (not currently, but good practice)
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                logger.debug("Main task cancelled successfully.")

        # Cancel periodic tasks
        if hasattr(self, '_periodic_task_handle') and self._periodic_task_handle and not self._periodic_task_handle.done():
            self._periodic_task_handle.cancel()
            try:
                await self._periodic_task_handle
            except asyncio.CancelledError:
                logger.debug("Periodic tasks cancelled successfully.")

        # Close exchange connections if any
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                await self.exchange.close() # Assuming exchange has an async close
            except Exception as e:
                logger.error(f"Error closing exchange connection: {e}", exc_info=True)

        # Other cleanup tasks can go here

        logger.info("Trading bot stopped.")


# --- CLI Setup ---
# Addressing item 9 from "things to check.txt": Graceful shutdown race
# Signal handlers should be registered early.
_bot_instance: Optional[TradingBot] = None

def _signal_handler(sig, frame):
    logger.warning(f"Signal {sig} received. Initiating graceful shutdown...")
    if _bot_instance:
        # We cannot call await _bot_instance.stop() directly from a sync signal handler.
        # We need to schedule it on the event loop.
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_bot_instance.stop()) # Schedule stop() to run on the loop
        else:
            logger.error("Event loop not running. Cannot schedule bot stop. May require forceful exit.")
            # Fallback or just let it try to exit; this scenario is tricky.
    else:
        logger.warning("No bot instance to stop. Exiting.")
        # If bot instance isn't created yet, a simple exit might be okay,
        # or set a global flag that `cli` checks before starting the bot.
        # For now, this will allow the program to try to exit if bot not up.
        raise SystemExit(0) # Force exit if bot not running

@click.command()
@click.option('--config', '-c', 'config_path', default='config/config.yaml', help='Path to the configuration file.')
@click.option('--mode', '-m', 'override_mode', type=click.Choice(['live', 'paper', 'backtest']), help='Override operation mode from config.')
def cli(config_path: str, override_mode: Optional[str]):
    """
    Modular Cryptocurrency Trading Bot
    """
    global _bot_instance

    # Register signal handlers early (item 9 fix)
    os_signal.signal(os_signal.SIGINT, _signal_handler)
    os_signal.signal(os_signal.SIGTERM, _signal_handler)

    try:
        bot_config = load_config(config_path)
        if override_mode:
            logger.info(f"Overriding mode from config '{bot_config.mode}' to '{override_mode}' via CLI.")
            bot_config.mode = override_mode

        _bot_instance = TradingBot(bot_config)
        asyncio.run(_bot_instance.run())

    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}")
    except ValueError as e: # Handles Pydantic validation errors or other ValueErrors from load_config
        logger.error(f"Configuration or setup error: {e}", exc_info=True)
    except Exception as e: # Catch-all for other unexpected errors during setup or runtime
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
    finally:
        logger.info("Bot CLI process finished.")
        # Ensure any final cleanup if bot didn't stop fully via signal
        if _bot_instance and _bot_instance.running:
             logger.warning("Bot was still marked as running at CLI exit. Forcing stop (if loop allows).")
             # This is a fallback, stop should ideally be handled by signals or main loop completion
             # If loop is stuck, this won't do much without more forceful measures.
             # asyncio.run(_bot_instance.stop()) # Careful with running new loop if old one crashed.


if __name__ == "__main__":
    # This allows running the bot directly using `python -m trading_bot.main`
    # Ensure logger is set up at least minimally if run this way without CLI immediately configuring it.
    # Basic setup if not already done by TradingBot instantiation through CLI.
    # setup_logger_minimal() # A hypothetical minimal setup if needed before config load.
    cli()
