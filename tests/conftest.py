import pytest
from randforest.connection_engine import Session
from randforest.data_fetching import SQL

# No es necesario testear la conexión. Más bien la conexión al query. Remover cuando no se necesita.
# Diferentes conjuntos de prueba para los diferentes queries de los motores.
# Así no se necesite el insert per sé, en el mock de la data.
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
