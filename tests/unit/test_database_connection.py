from hercule.connection_engine import Session


def test_db_connection_successful():
    response = Session('postgresql://api:girl11877@localhost:5432/malware')

    assert response().execute('SELECT true').first()[0] == True

def test_db_connection_error():
    try:
        engine = Session('postgresqla://apis:girl11877@localhest:3000/malware')
        if engine():
            response = {'connection': True}
    except Exception as error:
        response = {'connection': False}

    assert response == {'connection': False}
