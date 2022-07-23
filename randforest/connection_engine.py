from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging


class Session:
    
    def __init__(self, driver: str, user: str, password: str, host: str, port: str, database: str):
        self.driver = driver
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.connection = self.build_connection_string()
        self.engine = create_engine(self.connection, pool_size=5)

    def build_connection_string(self):
        return f'{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'

    def create_database_session(self):
        try:
            session = sessionmaker(bind=self.engine.connect(), expire_on_commit=False)
            return session()
        except Exception as e:
            logging.debug(f'Review connection parameters, particular error{e}')
            return {'connection_refused': True}

    def __call__(self):
        return self.create_database_session()
