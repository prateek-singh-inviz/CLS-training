import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)


def log_time(start_time):
    logging.info("Time taken for Job" + " : " + str(datetime.datetime.now() - start_time))
    return time.time()


