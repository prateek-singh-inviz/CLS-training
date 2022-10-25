import os
import yaml
from loguru import logger

ENV = os.environ.get('ENV','dev')

def load_config(job_name):
    try:
        # job_config = job_name[:job_name.rfind("_")]
        # logger.info(f"Job configs name found {job_config}")
        config = yaml.full_load(open(f"./resource/configs/{ENV}-config.yaml"))
        return config[job_name]
    except Exception as e:
        logger.error(e)
        raise Exception("Config not found")
