import functools
import json
import uuid

import duckdb

# type alias for json serialized list
jslist = str


def json_wrap(func):
    """A decorator that decodes json parameters and returns a json serialized.
    This is for user-defined functions in duckdb

     Args:
         func: takes multiple args
               json serialized lists or int/float/str
               returns a json serialized list or int/float/str
     Returns:
         wrapper function
    """

    @functools.wraps(func)
    def wrapper(*args):
        # Helper function to safely deserialize JSON
        def deserialize(arg):
            if not isinstance(arg, str):
                return arg
            try:
                return json.loads(arg)
            except json.JSONDecodeError:
                return arg

        # Apply safe_deserialize to all arguments
        deserialized_args = [deserialize(arg) for arg in args]

        # Call the original function with potentially deserialized arguments
        result = func(*deserialized_args)

        # Attempt to serialize the result back to a JSON string
        if isinstance(result, list):
            return json.dumps(result)
        return result

    return wrapper


class Connection:
    """Context manager for duckdb connection, enclosed by transaction

    Example:
        conn = DuckDBConnection('sample.duckdb')
        with conn:
            ...
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = duckdb.connect(self.db_path)
        # define some macros
        # WARNING!! macros are case-sensitive for column identifiers!!!
        self.conn.execute(
            """
            CREATE OR REPLACE MACRO js(x) AS to_json(list(x))
                          """
        )
        # Begin a new transaction
        self.conn.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Explicit roll back the transaction on exception
            self.conn.rollback()
            # reraise the exception
            raise exc_val
        # Commit the transaction on success
        self.conn.commit()
        self.conn.close()

    # Pass through all methods of the DuckDB connection object
    def __getattr__(self, name):
        # Check if the method exists in the connection object
        if hasattr(self.conn, name):
            attr = getattr(self.conn, name)
            # If it's callable (a method), return it
            if callable(attr):
                return attr
        raise AttributeError(f"'DuckDBConnection' object has no attribute '{name}'")

    def regfn(self, func):
        """register udf"""
        # create_function begins a new transaction so commit all before
        self.conn.commit()
        self.conn.create_function(func.__name__, func)

    # The following 3 iterations (iter_group, iter_window, iter_chunk)
    # can be considered only when
    #     1. duckDB functions cannot handle the problem AND
    #     2. the data size is out of RAM capacity.
    # Otherwise, either stay in duckDB or load the whole table in memory
    def iter_group(self, table_name, columns):
        """Yiels df for each group by columns

        Args:
            table_name: str
            columns: list of columns,
                     NO sql expressions, only explicit column names allowed.

        Yields:
            Pandas DF

        Example:
            for df in conn.iter_group('table', ['col1', 'col2']):
                newdf = fn(df)
                conn.insert(newdf, 'new_table')
                ...
        """
        if not isinstance(columns, list):
            raise TypeError("'columns' must be a list")

        cols = ",".join(columns)

        index_name = "tempIndex" + str(uuid.uuid4()).replace("-", "_")
        self.conn.execute(
            f"""
            CREATE INDEX {index_name} ON {table_name}({cols})
            """
        )

        groups = self.conn.execute(
            f"""
            SELECT DISTINCT {cols} FROM {table_name}
            ORDER BY {cols}
            """
        ).fetchall()

        for row in groups:
            column_clause = " AND ".join([f"{col} = ?" for col in columns])
            query = f"""
            SELECT * FROM {table_name} WHERE {column_clause}
            """
            yield self.conn.execute(query, row).df()

        self.conn.execute(f"DROP INDEX {index_name}")

    def iter_window(
        self, table_name, refcol, window_size, step_size=None, start=None, end=None
    ):
        """Yiels df for each window over a table

        Args:
            table_name: str
            refcol: str, reference column for the range decision,
                    often 'datetime' column, '2023-01-23 09:32:34'::timestamp
                    No sql expression, only explicit column name.
            window_size: value for addition to the 'refcol' value
            step_size: value for addition to the 'refcol' value
                       if None, then the same as 'window_size', no overlaps
            start: start value, MIN(refcol) if not specified
            end: end value, MAX(refcol) if not specified

        Yields:
            Pandas DF

        Example:
            for df in conn.iter_window('table', 'datetime',
                                       relativedelta(months=3),
                                       relativedelta(months=1)):
                newdf = fn(df)
                conn.insert(newdf, 'new_table')
                ...
        """

        if not step_size:
            step_size = window_size

        index_name = "tempIndex" + str(uuid.uuid4()).replace("-", "_")
        self.conn.execute(
            f"""
        CREATE INDEX {index_name} ON {table_name}({refcol})
        """
        )

        # Query to find the start and end dates
        query = f"SELECT MIN({refcol}), MAX({refcol}) FROM {table_name}"
        table_start, table_end = self.conn.execute(query).fetchone()
        start = start or table_start
        end = end or table_end

        current_start = start
        current_end = current_start + window_size

        while current_start <= end:
            query = f"""
            SELECT * FROM {table_name}
            WHERE {refcol} >= ? AND {refcol} < ?
            ORDER BY {refcol}
            """

            # Empty dataframes should also be yielded
            yield self.conn.execute(query, (current_start, current_end)).df()

            current_start += step_size
            current_end = current_start + window_size

        self.conn.execute(f"DROP INDEX {index_name}")

    def iter_chunk(self, table_name, chunk_size):
        """Yiels df in a chunk of rows

        Args:
            table_name: str
            chunk_size: int

        Yields:
            Pandas DF

        Example:
            for df in conn.iter_chunk('table', 100):
                newdf = fn(df)
                conn.insert(newdf, 'new_table')
                ...
        """

        # Query to find the start and end dates
        query = f"SELECT MIN(rowid), MAX(rowid) FROM {table_name}"
        start, end = self.conn.execute(query).fetchone()

        current_start = start
        current_end = current_start + chunk_size

        while current_start <= end:
            query = f"""
            SELECT * FROM {table_name}
            WHERE rowid >= ? AND rowid < ?
            ORDER BY rowid
            """

            # Empty dataframes should also be yielded
            yield self.conn.execute(query, (current_start, current_end)).df()

            current_start += chunk_size
            current_end = current_start + chunk_size

    def push(self, df, table_name):
        """Insert df to table

        Args:
            df: dataframe
            table_name: str

        Returns:
            None
        """
        # To check the current tables
        query = "SHOW TABLES"
        tables = set(x[0].lower() for x in self.conn.execute(query).fetchall())

        dfview_name = "tempDFView" + str(uuid.uuid4()).replace("-", "_")
        self.conn.register(dfview_name, df)
        if not table_name.lower() in tables:
            self.conn.execute(
                f"CREATE TABLE {table_name} AS SELECT * FROM {dfview_name}"
            )

        else:
            self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM {dfview_name}")
        self.conn.unregister(dfview_name)
