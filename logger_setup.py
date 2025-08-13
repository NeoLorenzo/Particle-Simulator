# logger_setup.py

import logging
import os
import json

def setup_logging(config_path='config.json'):
    """
    Sets up logging for the application.

    Reads logging configuration from the provided JSON file, creates a run-specific
    log directory, and configures a logger to output to both the console and a
    log file.

    Data Contract:
    - Inputs: config_path (str) - Path to the configuration file.
    - Outputs: None
    - Side Effects:
        - Configures the root logger.
        - Creates directories for log files.
    - Invariants: Assumes the config file contains 'run_id' and a 'logging' dictionary
      with 'level' and 'format'.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    run_id = config['run_id']
    log_config = config['logging']

    # Create directories for logs
    log_dir = os.path.join('runs', run_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'simulation.log')

    # Configure logging
    logging.basicConfig(
        level=log_config['level'],
        format=log_config['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # To output to console
        ]
    )

    logging.info(f"Logging initialized. Run ID: {run_id}. Log file: {log_file}")