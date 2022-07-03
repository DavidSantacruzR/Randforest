import pytest
from hercule.connection_engine import Session
from hercule.data_fetching import SQL


@pytest.fixture(autouse=True)
def mock_create_database_session(mocker):
    return mocker.patch.object(Session, 'create_database_session', return_value=object)


@pytest.fixture(autouse=True)
def mock_fetch_query_data(mocker):
    response = {'sample_size': 480, 'number_of_features': 8}
    return mocker.patch.object(SQL, '__call__', return_value=response)


@pytest.fixture(autouse=True)
def mock_fetch_number_of_records(mocker):
    return mocker.patch.object(SQL, 'fetch_number_of_records')


@pytest.fixture(autouse=True)
def mock_fetch_number_of_features(mocker):
    return mocker.patch.object(SQL, 'fetch_number_of_features')
