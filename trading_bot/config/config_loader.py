import os
import yaml
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator, Extra
from dotenv import load_dotenv

# Load .env file variables into environment
load_dotenv()

class BaseSubConfig(BaseModel):
    class_path: str = Field(alias="class") # Allow 'class' as field name in YAML
    params: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True # Allows using 'class' in YAML and mapping to 'class_path'
        extra = Extra.forbid # Forbid extra fields in params if not defined by a more specific model

class ExchangeConfig(BaseSubConfig):
    # Add specific exchange params if needed for validation, e.g.
    # params: ExchangeParams
    pass

class StrategyConfig(BaseSubConfig):
    pass

class ExecutionConfig(BaseSubConfig):
    pass

class RiskCheckConfig(BaseSubConfig):
    enabled: bool = True # Individual checks can be disabled

class RiskConfig(BaseModel):
    enabled: bool = False # Default to False as per plan
    checks: List[RiskCheckConfig] = Field(default_factory=list)

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/trading_bot.log"
    rotation: str = "1 day"
    retention: str = "30 days"

    @validator('level')
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Logging level must be one of {allowed_levels}")
        return v.upper()

class MonitoringAlertsConfig(BaseSubConfig): # Inherits class_path and params
    enabled: bool = False
    # Example: slack_webhook: Optional[str] = None (handled by params dict)

class MonitoringAnalyticsConfig(BaseSubConfig): # Inherits class_path and params
    enabled: bool = False
    # Example: report_interval: int = 3600 (handled by params dict)

class MonitoringConfig(BaseModel):
    prometheus_port: Optional[int] = 8000 # Optional with a default
    alerts: Optional[MonitoringAlertsConfig] = None
    analytics: Optional[MonitoringAnalyticsConfig] = None

class TrackingConfig(BaseSubConfig): # Inherits class_path and params
    enabled: bool = False
    # Example: persistence_enabled: bool = False (handled by params dict)
    # reporting: Optional[Dict[str, Any]] = None # Or a more specific model

class AppConfig(BaseModel): # Renamed from 'Config' to avoid Pydantic internal conflicts
    mode: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    exchange: ExchangeConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    risk: Optional[RiskConfig] = None # Made RiskConfig optional as per things_to_check.txt (item 10)
    monitoring: Optional[MonitoringConfig] = None # Made MonitoringConfig optional
    tracking: Optional[TrackingConfig] = None # Made TrackingConfig optional

    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['live', 'paper', 'backtest']:
            raise ValueError("Mode must be 'live', 'paper', or 'backtest'")
        return v

    class Config:
        extra = Extra.forbid # Forbid any top-level unknown fields in config.yaml

def _substitute_env_vars_in_string(value: str) -> str:
    """Substitute environment variables in a string, e.g., ${VAR_NAME}."""
    env_pattern = re.compile(r'\$\{([^}]+)\}')

    def replacer(match):
        env_var_name = match.group(1)
        env_var_value = os.getenv(env_var_name)
        if env_var_value is None:
            # print(f"Warning: Environment variable '{env_var_name}' not set, using it as literal.")
            return match.group(0) # Return the original placeholder if not found
        return env_var_value

    return env_pattern.sub(replacer, value)

def _substitute_env_vars_recursive(item: Any) -> Any:
    """Recursively substitute environment variables in a nested structure."""
    if isinstance(item, str):
        return _substitute_env_vars_in_string(item)
    elif isinstance(item, dict):
        return {k: _substitute_env_vars_recursive(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [_substitute_env_vars_recursive(elem) for elem in item]
    return item

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """
    Load, substitute environment variables, and validate configuration.
    """
    path = Path(config_path)
    if not path.is_file(): # More specific check
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    try:
        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {path}: {e}")
    except IOError as e:
        raise IOError(f"Error reading config file {path}: {e}")


    if not isinstance(raw_config, dict):
        raise TypeError(f"Config file {path} does not contain a valid YAML dictionary.")

    # Substitute environment variables
    config_with_env_vars = _substitute_env_vars_recursive(raw_config)

    try:
        # Validate with Pydantic
        return AppConfig(**config_with_env_vars)
    except Exception as e: # Catch Pydantic's ValidationError specifically if possible, but general Exception for now
        raise ValueError(f"Configuration validation error: {e}")


if __name__ == '__main__':
    # Example of how to use the loader
    # Create a dummy .env file
    with open(".env", "w") as f:
        f.write("MY_API_KEY=test_key_from_env\n")
        f.write("LOG_LEVEL_ENV=DEBUG\n")

    # Create a dummy config.yaml
    dummy_config_content = """
mode: paper
logging:
  level: ${LOG_LEVEL_ENV} # Test env var substitution
  file: logs/bot.log
exchange:
  class: exchange.MyExchange
  params:
    apiKey: ${MY_API_KEY}
    secret: "hardcoded_secret"
strategy:
  class: strategy.MyStrategy
  params:
    threshold: 0.5
execution:
  class: execution.MyExecutor
  params: {}
risk: # This section is optional
  enabled: true
  checks:
    - class: risk.MyRiskCheck
      enabled: false
      params:
        max_drawdown: 0.1
monitoring: # This is also optional
  prometheus_port: 8001
  alerts:
    enabled: true
    class: monitoring.MyAlerter
    params:
      webhook: ${SLACK_HOOK_MISSING_EXAMPLE} # Test missing env var
"""
    dummy_config_path = Path("dummy_config.yaml")
    with open(dummy_config_path, "w") as f:
        f.write(dummy_config_content)

    try:
        print(f"Attempting to load config from: {dummy_config_path.resolve()}")
        config = load_config(str(dummy_config_path))
        print("Config loaded successfully!")
        print(f"Mode: {config.mode}")
        print(f"Log Level: {config.logging.level}")
        print(f"Exchange API Key: {config.exchange.params.get('apiKey')}")
        if config.risk and config.risk.checks:
            print(f"Risk Check 1 enabled: {config.risk.checks[0].enabled}")
            print(f"Risk Check 1 class: {config.risk.checks[0].class_path}")
        if config.monitoring:
            print(f"Prometheus Port: {config.monitoring.prometheus_port}")
            if config.monitoring.alerts:
                 print(f"Alerts Webhook: {config.monitoring.alerts.params.get('webhook')}")


    except (FileNotFoundError, ValueError, IOError, TypeError) as e:
        print(f"Error loading config: {e}")
    finally:
        # Clean up dummy files
        if dummy_config_path.exists():
            dummy_config_path.unlink()
        env_file = Path(".env")
        if env_file.exists():
            env_file.unlink()

    # Test loading the actual config file if it exists
    actual_config_path = Path("trading_bot/config/config.yaml") # Adjust if your actual config is elsewhere
    if actual_config_path.exists():
        try:
            print(f"\nAttempting to load actual config from: {actual_config_path.resolve()}")
            # Ensure necessary env vars are set for this test or handle missing ones
            os.environ['BINANCE_API_KEY'] = 'dummy_key_for_test'
            os.environ['BINANCE_SECRET'] = 'dummy_secret_for_test'
            actual_cfg = load_config(str(actual_config_path))
            print("Actual config loaded successfully!")
            print(f"Mode: {actual_cfg.mode}")
            # print(f"Log Level: {actual_cfg.logging.level}")
            # print(f"Exchange Class: {actual_cfg.exchange.class_path}")
            # if actual_cfg.risk:
            #     print(f"Risk enabled: {actual_cfg.risk.enabled}")

        except Exception as e:
            print(f"Error loading actual config: {e}")
    else:
        print(f"\nActual config file not found at {actual_config_path}, skipping load test.")
