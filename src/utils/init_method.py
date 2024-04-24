import os
import random
import socket
import sys

from loguru import logger as log


def init_dirs(config):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    samples_dir = os.path.join(config.output_dir, 'samples')
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    logs_dir = os.path.join(config.output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    tensorboard_dir = os.path.join(config.output_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    return samples_dir, logs_dir, tensorboard_dir


def init_logs(config):
    log.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{" \
                 "name}</cyan>:<cyan>{function}</cyan> <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{" \
                 "message}</level> "
    log.add(sys.stdout, format=log_format)
    filename = os.path.join(config.output_dir, 'logs', f'{socket.gethostname()}/info_{{time}}.log')
    log.add(filename, format=log_format, encoding='utf-8')


def init_seeds(config):
    if not hasattr(config, 'init_seeds') or config.init_seeds is None:
        seed = random.randint(0, 2 ** 15)
    else:
        seed = config.init_seeds
    log.info(f'Initializing seed: {seed}')
