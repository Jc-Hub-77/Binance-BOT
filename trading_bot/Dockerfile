# Stage 1: Builder stage to compile dependencies if needed and keep final image small
FROM python:3.10-slim as builder

WORKDIR /build

# Install build tools if any C extensions need compilation (e.g. numpy, pandas can sometimes)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ build-essential

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies to a user directory to avoid running as root
# Using --user can be problematic if the user doesn't exist in the next stage or PATH is not set.
# Alternative: Install to a specific path like /opt/venv and then copy to final stage.
# For simplicity with --user, ensure user exists and PATH is set in final stage.
# RUN pip install --user -r requirements.txt
# Switched to a virtual environment approach for cleaner separation and less PATH fiddling.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Final application stage
FROM python:3.10-slim

# Create a non-root user and group for security
RUN groupadd -r trader && useradd --no-log-init -r -g trader trader

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . .

# Ensure correct ownership of app files for the non-root user
# This might be broad; consider more specific chown if needed.
RUN chown -R trader:trader /app

# Set environment variables (PATH for venv, PYTHONPATH if needed)
ENV PATH="/opt/venv/bin:$PATH"
# ENV PYTHONPATH="/app" # Add if imports are relative to /app and not using -m

# Switch to the non-root user
USER trader

# Healthcheck (as defined in order bot v4.txt)
# Ensure prometheus_client is installed and server runs on 8000 by default
# The bot needs to be running for this healthcheck to pass.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1m \
  CMD curl -f http://localhost:8000/metrics || exit 1
# Note: Default PROMETHEUS_PORT is 8000 in config_loader.py for MonitoringConfig

# Command to run the application
# Using `python -m trading_bot.main` makes it runnable as a module.
# Default config path is 'config/config.yaml' in main.py's @click.option.
CMD ["python", "-m", "trading_bot.main"]

# Expose the Prometheus metrics port if it's meant to be accessible from outside the container
# This is often done via docker-compose.yml's port mapping.
# EXPOSE 8000 # Default Prometheus port (if not overridden by .env or config)
