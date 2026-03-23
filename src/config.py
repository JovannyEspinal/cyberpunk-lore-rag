"""
config.py
---------
Shared configuration and logging setup for all modules.
"""

import os
import logging

import yaml
from dotenv import load_dotenv

load_dotenv()

with open("config.yaml") as f:
    config = yaml.safe_load(f)


def get_logger(name: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(name)
