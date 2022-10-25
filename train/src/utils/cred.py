import boto3


def get_redshift_credentials(connection_name):
    client = boto3.client("glue")
    response = client.get_connection(Name=connection_name)
    conn_properties = response["Connection"]["ConnectionProperties"]
    return conn_properties


def get_s3_credentials():
    session = boto3.Session()
    credentials = session.get_credentials()

    # Credentials are refreshable, so accessing your access key / secret key
    # separately can lead to a race condition. Use this to get an actual matched
    # set.
    credentials = credentials.get_frozen_credentials()
    return credentials.access_key, credentials.secret_key
