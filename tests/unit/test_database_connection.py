from randforest.connection_engine import Session


def test_db_connection_successful():
    connection_response = Session(
        driver='postgresql',
        user='api',
        password='girl11877',
        host='localhost',
        port='5432',
        database='malware'
    )

    assert connection_response()
