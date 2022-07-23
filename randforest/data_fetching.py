class Database:

    def __init__(self, session, sample_size: float, table: str):
        self.session = session()
        self.size = sample_size
        self.table = table
        self.total_sample = self.fetch_number_of_records()
        self.features = self.fetch_number_of_features()

    def fetch_number_of_records(self) -> int:
        pass

    def fetch_number_of_features(self) -> int:
        pass

    def query_data(self):
        pass

    def __call__(self):
        self.fetch_number_of_records()
        self.fetch_number_of_features()

        return self.query_data()


class SQL(Database):  

    def fetch_number_of_records(self) -> int:
        return self.session.execute(
            f'''
                SELECT count(*) FROM {self.table}
            '''
            ).fetchone()[0]

    def fetch_number_of_features(self) -> int:
        return self.session.execute(
            f'''
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = '{self.table}'
            '''
        ).fetchone()[0]

    def query_data(self):
        return self.session.execute(
            f'''
                SELECT * FROM {self.table}
                limit({self.total_sample * self.size})
            '''
            ).fetchall()
