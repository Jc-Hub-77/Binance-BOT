import asyncio
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError as TenacityRetryError
from pybreaker import CircuitBreaker, CircuitBreakerError
# from loguru import logger # Optional, for logging retry attempts or breaker state changes

# --- Default Circuit Breaker Instances ---
# You can define multiple breakers for different parts of the system if needed.
# Example: A general exchange interaction breaker.
exchange_breaker = CircuitBreaker(
    fail_max=5,       # Max number of failures before opening the circuit
    reset_timeout=60, # Seconds to wait before trying to close the circuit (HALF-OPEN state)
    # exclude=[ExpectedException1, ExpectedException2] # List of exceptions to ignore
    # listeners=[MyBreakerListener()] # For custom actions on state changes
    name="ExchangeAPIBreaker"
)

# Example: A more sensitive breaker for critical, fast operations
# critical_op_breaker = CircuitBreaker(fail_max=2, reset_timeout=30, name="CriticalOpBreaker")


# --- Custom Decorators ---

def with_retry(
    stop_attempts: int = 3,
    wait_min_seconds: int = 1,
    wait_max_seconds: int = 10,
    # Can add specific exception types to retry on, e.g., retry_on_exceptions=(IOError, TimeoutError)
):
    """
    A decorator that uses Tenacity to automatically retry a function call
    with exponential backoff if it fails.

    Args:
        stop_attempts: Maximum number of attempts before giving up.
        wait_min_seconds: Minimum time to wait before retrying (seconds).
        wait_max_seconds: Maximum time to wait before retrying (seconds).
    """
    def decorator(func):
        # Determine if the function is async or sync
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Configure Tenacity retry for async functions
                retry_decorator = retry(
                    stop=stop_after_attempt(stop_attempts),
                    wait=wait_exponential(multiplier=1, min=wait_min_seconds, max=wait_max_seconds), # multiplier=1 for linear increase of random part
                    # before_sleep=lambda rs: logger.warning( # Tenacity's before_sleep is sync
                    #    f"Retrying async {func.__name__} due to {rs.last_attempt.exception()}. "
                    #    f"Attempt {rs.attempt_number}/{stop_attempts}. Waiting {rs.next_action.sleep}s..."
                    # ),
                    reraise=True # Reraise the last exception if all retries fail
                )

                async def before_sleep_async(retry_state): # Custom async before_sleep
                    # logger.warning(
                    #    f"Retrying async {func.__name__} due to {retry_state.outcome.exception()}. "
                    #    f"Attempt {retry_state.attempt_number}/{stop_attempts}. Waiting {retry_state.next_action.sleep}s..."
                    # )
                    pass # Placeholder for actual async logging if needed

                decorated_func = retry_decorator(func)
                decorated_func.retry.before_sleep = before_sleep_async # type: ignore # Assign custom async callback

                try:
                    return await decorated_func(*args, **kwargs)
                except TenacityRetryError as e: # This is raised if reraise=True and retries exhausted
                    # logger.error(f"Async function {func.__name__} failed after {stop_attempts} retries: {e.last_attempt.exception()}")
                    raise e.last_attempt.exception() # type: ignore # Raise the original exception
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Configure Tenacity retry for sync functions
                retry_decorator = retry(
                    stop=stop_after_attempt(stop_attempts),
                    wait=wait_exponential(multiplier=1, min=wait_min_seconds, max=wait_max_seconds),
                    # before_sleep=lambda rs: logger.warning(
                    #    f"Retrying sync {func.__name__} due to {rs.last_attempt.exception()}. "
                    #    f"Attempt {rs.attempt_number}/{stop_attempts}. Waiting {rs.next_action.sleep}s..."
                    # ),
                    reraise=True
                )
                decorated_func = retry_decorator(func)
                try:
                    return decorated_func(*args, **kwargs)
                except TenacityRetryError as e:
                    # logger.error(f"Sync function {func.__name__} failed after {stop_attempts} retries: {e.last_attempt.exception()}")
                    raise e.last_attempt.exception() # type: ignore
            return sync_wrapper
    return decorator


def with_circuit_breaker(breaker: CircuitBreaker):
    """
    A decorator that wraps a function call with a PyBreaker circuit breaker.
    If the function fails repeatedly, the circuit opens, and subsequent calls
    will fail immediately (raising CircuitBreakerError) until the breaker resets.

    Args:
        breaker: An instance of pybreaker.CircuitBreaker.
    """
    def decorator(func):
        # PyBreaker's @breaker decorator handles both sync and async functions correctly
        # by inspecting the function it wraps.
        # If func is async, breaker(func) returns an async function.
        # If func is sync, breaker(func) returns a sync function.

        # We need to ensure our logging or any pre/post call logic also respects async/sync
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    # logger.debug(f"Calling {func.__name__} via async circuit breaker '{breaker.name}' (State: {breaker.current_state})")
                    return await breaker.call_async(func, *args, **kwargs) # Use call_async for coroutines
                except CircuitBreakerError as e:
                    # logger.error(f"Circuit breaker '{breaker.name}' is OPEN for {func.__name__}. Call rejected: {e}")
                    raise # Re-raise the CircuitBreakerError
                # except Exception as e:
                    # logger.error(f"Exception caught by circuit breaker wrapper for {func.__name__}: {e}")
                    # This exception will be counted by the breaker.
                    # raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    # logger.debug(f"Calling {func.__name__} via sync circuit breaker '{breaker.name}' (State: {breaker.current_state})")
                    return breaker.call(func, *args, **kwargs) # Use call for regular functions
                except CircuitBreakerError as e:
                    # logger.error(f"Circuit breaker '{breaker.name}' is OPEN for {func.__name__}. Call rejected: {e}")
                    raise
                # except Exception as e:
                    # logger.error(f"Exception caught by circuit breaker wrapper for {func.__name__}: {e}")
                    # raise
            return sync_wrapper

    return decorator


# --- Example Usage (for testing) ---
class NetworkError(Exception):
    pass

class CriticalServiceError(Exception):
    pass

# Example sync function
@with_retry(stop_attempts=2, wait_min_seconds=0.1, wait_max_seconds=0.2)
@with_circuit_breaker(exchange_breaker)
def fetch_data_sync(url: str, fail_count: list): # fail_count is a list with one int element
    # logger.info(f"Attempting to fetch data synchronously from {url}...")
    if fail_count[0] > 0:
        fail_count[0] -= 1
        # logger.warning(f"Simulating sync failure for {url}. Remaining failures to simulate: {fail_count[0]}")
        raise NetworkError(f"Failed to connect to {url} (sync)")
    # logger.success(f"Successfully fetched data from {url} (sync).")
    return f"Data from {url}"

# Example async function
@with_retry(stop_attempts=2, wait_min_seconds=0.1, wait_max_seconds=0.2)
@with_circuit_breaker(exchange_breaker)
async def fetch_data_async(url: str, fail_count: list):
    # logger.info(f"Attempting to fetch data asynchronously from {url}...")
    await asyncio.sleep(0.05) # Simulate async I/O
    if fail_count[0] > 0:
        fail_count[0] -= 1
        # logger.warning(f"Simulating async failure for {url}. Remaining failures to simulate: {fail_count[0]}")
        raise NetworkError(f"Failed to connect to {url} (async)")
    # logger.success(f"Successfully fetched data from {url} (async).")
    return f"Async data from {url}"


async def _resilience_test_main():
    print("--- Resilience Utilities Test ---")
    # logger.remove()
    # logger.add(sys.stderr, level="DEBUG")

    # Test sync function with retries and circuit breaker
    print("\nTesting synchronous function with failures...")
    sync_fail_counter = [1] # Will fail once, succeed on first retry
    try:
        result_sync = fetch_data_sync("http://sync-service.com", sync_fail_counter)
        print(f"  Sync result (after 1 fail, 1 retry): {result_sync}")
        assert result_sync == "Data from http://sync-service.com"
    except Exception as e:
        print(f"  Sync test failed unexpectedly: {e}")
        assert False

    print("\nTesting synchronous function that exhausts retries & opens circuit breaker...")
    # Reset breaker for this test part if needed, or use a fresh one
    # For simplicity, we continue with the global 'exchange_breaker'
    # It might already be half-open or closed depending on previous tests if not isolated.
    # Let's assume it's closed or we make it fail enough times.

    sync_fail_counter_breaker = [exchange_breaker._fail_max + 1] # Fail enough to open breaker
    for i in range(exchange_breaker._fail_max + 2): # Try enough times to open and then test open state
        try:
            print(f"  Attempt {i+1} to call failing sync function...")
            fetch_data_sync("http://sync-breaker-service.com", sync_fail_counter_breaker)
            if i < exchange_breaker._fail_max : # Should fail for these attempts
                 print(f"  Call {i+1} unexpectedly succeeded (expected failure).")
            # else: call might succeed if breaker half-closes and this call is the one that closes it.
        except NetworkError:
            print(f"  Call {i+1} failed with NetworkError (expected for first {exchange_breaker._fail_max} calls).")
        except CircuitBreakerError:
            print(f"  Call {i+1} correctly failed with CircuitBreakerError (breaker is OPEN).")
            assert i >= exchange_breaker._fail_max # Should only happen after fail_max failures
            break # Stop once breaker is confirmed open
        except Exception as e:
            print(f"  Call {i+1} failed with unexpected error: {e}")
    else: # Loop finished without break (i.e. CircuitBreakerError not caught)
        print("  Circuit breaker did not seem to open as expected for sync function.")
        # This might happen if fail_count logic or breaker state isn't perfectly aligned for this test structure.

    # Wait for breaker to potentially reset (optional, for testing half-open state)
    # print(f"\n  Waiting {exchange_breaker.reset_timeout + 1}s for breaker to potentially half-close...")
    # await asyncio.sleep(exchange_breaker.reset_timeout + 1)


    # Test async function
    print("\nTesting asynchronous function with failures...")
    async_fail_counter = [1] # Fail once, succeed on retry
    try:
        result_async = await fetch_data_async("http://async-service.com", async_fail_counter)
        print(f"  Async result (after 1 fail, 1 retry): {result_async}")
        assert result_async == "Async data from http://async-service.com"
    except Exception as e:
        print(f"  Async test failed unexpectedly: {e}")
        assert False

    # Test async function that exhausts retries
    print("\nTesting asynchronous function that exhausts retries (breaker state depends on previous tests)...")
    async_fail_counter_exhaust = [3] # stop_attempts is 2, so this will fail twice and then raise
    try:
        await fetch_data_async("http://async-retry-exhaust.com", async_fail_counter_exhaust)
        assert False, "Async function should have raised NetworkError after exhausting retries"
    except NetworkError:
        print(f"  Async function correctly failed with NetworkError after exhausting retries.")
    except Exception as e:
        print(f"  Async function failed with unexpected error after retries: {e}")
        assert False

    print("\n--- Resilience Utilities Test Finished ---")

if __name__ == '__main__':
    import sys
    # logger.remove() # Ensure clean logging for test output
    # logger.add(sys.stderr, level="DEBUG")
    asyncio.run(_resilience_test_main())
