# logger_setup.py

import logging
import os
import json

def setup_logging(config_path='config.json'):
    """
    Sets up logging for the application.

    Reads logging configuration, creates a run-specific log directory, and
    configures a dedicated application logger (not the root logger) to output
    to both the console and a log file. This prevents capturing verbose logs
    from third-party libraries like Numba.

    Data Contract:
    - Inputs: config_path (str) - Path to the configuration file.
    - Outputs: None
    - Side Effects:
        - Configures the "particle_sim" logger.
        - Creates directories for log files.
    - Invariants: Assumes the config file contains 'run_id' and a 'logging' dictionary
      with 'level' and 'format'.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    run_id = config['run_id']
    log_config = config['logging']

    # --- Get a dedicated logger for the application ---
    logger = logging.getLogger("particle_sim")
    logger.setLevel(log_config['level'])

    # --- Prevent logs from propagating to the root logger ---
    logger.propagate = False

    # --- Create directories for logs ---
    log_dir = os.path.join('runs', run_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'simulation.log')

    # --- Create formatter and handlers ---
    formatter = logging.Formatter(log_config['format'])

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # --- Add handlers to the logger ---
    # Clear existing handlers to avoid duplication if this function is called again
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logging initialized. Run ID: {run_id}. Log file: {log_file}")