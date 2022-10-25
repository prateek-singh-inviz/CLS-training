import argparse
import importlib
import inspect
import logging
import os

from datetime import datetime
import sys
from dotenv import load_dotenv
from loguru import logger
import boto3

from src.config.logging import Formatter, InterceptHandler
from src.config.parser import load_config
from src.train.cls_train_model import CLSTrainModelJob
from src.utils.utility_functions import log_time

load_dotenv()
LOGGING_LEVEL = logging.INFO

logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(
    handlers=[
        {"sink": sys.stdout, "level": LOGGING_LEVEL, "format": Formatter().format}
    ]
)

if __name__ == '__main__':
    # args = getResolvedOptions(sys.argv, ["TempDir", "JOB_NAME"])

    PARSER = argparse.ArgumentParser(description='Run a PySpark job')
    args = PARSER.parse_args().__dict__

    logger.info(args)

    s_time = datetime.now()
    # config = load_config('aws_keys')
    session = boto3.session.Session()
    s3_client = session.client('s3')

    start_time = datetime.now()
    config = load_config('ner_train_model')
    logger.info(' Starting job at {}'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    job = CLSTrainModelJob(args, config, s3_client)
    job.train(args, config, s3_client)

    log_time(s_time)

