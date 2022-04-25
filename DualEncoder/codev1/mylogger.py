import os
import sys
import logging
import yaml


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    with open('./config.yml', 'r') as f:
        config = yaml.safe_load(f)

    rf_handler = logging.FileHandler(config['config']['log_file'])
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    rf_handler.setLevel(logging.DEBUG)
    logger.addHandler(rf_handler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)

    return logger

logger = get_logger()
