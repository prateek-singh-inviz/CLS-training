import logging
import time

import pandas as pd
import pg8000


class RedshiftPreaction:

    def __init__(self, config, conn_details, redshift_credentials) -> None:
        self.__config = config
        self.__conn_details = conn_details
        self.__redshift_credentials = redshift_credentials

    def fetch_data(self, query):
        conn = None
        try:
            conn = RedshiftPreaction.get_conn(self.__config, self.__conn_details, self.__redshift_credentials)
            data = RedshiftPreaction.get_redshift_data(conn, query)
            return data
        finally:
            if conn is not None:
                conn.close()

    @classmethod
    def get_conn(cls, conf, conn_details, redshift_credentials):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__class__.__name__)
        try:
            conn = pg8000.Connection(
                database=conf[conn_details]["database"],
                port=5439,
                host=redshift_credentials["JDBC_CONNECTION_URL"][16:-10],
                user=redshift_credentials["USERNAME"],
                password=redshift_credentials["PASSWORD"],
            )
        except Exception as err:
            logger.error(err)
            raise err
        return conn

    @classmethod
    def run(cls, query, conf, conn_details, redshift_credentials):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__class__.__name__)
        logger.info("Started preaction ......")
        start_time = time.time()
        conn = cls.get_conn(conf, conn_details, redshift_credentials)
        conn.run(query)
        conn.close()
        elapsed_time = time.time() - start_time
        logger.info(f"Preaction completed in {elapsed_time}")

    @classmethod
    def get_redshift_data(cls, conn, query):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__class__.__name__)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            colnames = [desc[0] for desc in cursor.description]
            data = pd.DataFrame(cursor.fetchall(), columns=colnames)
        except Exception as err:
            logger.error(err)
            raise err
        return data
