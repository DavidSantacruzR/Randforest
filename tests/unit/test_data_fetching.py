from hercule.data_fetching import SQL
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def create_session(connection_string: str) -> object:
    session = sessionmaker(bind=create_engine(connection_string, pool_size=5), 
    expire_on_commit= False) 
    
    return session()


def test_fetching_sql_data():
    test_session = create_session('postgresql://api:girl11877@localhost:5432/malware')
    data = SQL(test_session, 0.20, 'malware_data')()

    result = {'sample_size': len(data), 'number_of_features': len(data[0])}

    assert result == {'sample_size': 480, 'number_of_features': 8}
