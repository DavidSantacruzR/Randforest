from hercule.data_fetching import SQL
from hercule.connection_engine import Session

session = Session(
        driver='postgresql',
        user='api',
        password='girl11877',
        host='localhost',
        port='5432',
        database='malware'
    )


def test_fetching_sql_data():
    data = SQL(session(), 0.20, 'malware_data')

    assert data() == {'sample_size': 480, 'number_of_features': 8}
