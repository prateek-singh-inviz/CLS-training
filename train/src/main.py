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
    PARSER.add_argument('--JOB_NAME', type=str, required=False, dest="JOB_NAME",
                        default=os.environ.get("JOB_NAME", "cls_train_dataprep"),
                        help="The name of the job module you want to run. (ex: poc will run job on jobs.poc package)")

    args = PARSER.parse_args().__dict__

    logger.info(args)

    s_time = datetime.now()
    # config = load_config('aws_keys')
    session = boto3.session.Session()
    s3_client = session.client('s3')

    job_name = args['JOB_NAME']

    start_time = datetime.now()
    config = load_config(job_name)
    logger.info(' Starting job {} at {}'.format(job_name, start_time.strftime("%Y-%m-%d %H:%M:%S")))

    job_module = importlib.import_module('src.main.jobs.%s' % job_name)
    logger.info(inspect.signature(job_module.execute))
    job_module.execute(args, config, s3_client)

    log_time(s_time)

