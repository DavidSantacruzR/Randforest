from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class Session:
    
    def __init__(self, connection: str):
        self.connection = connection
        self.engine = create_engine(connection, pool_size=5)

    def __call__(self):
        session = sessionmaker(bind=self.engine.connect(), expire_on_commit=False)
        return session()
