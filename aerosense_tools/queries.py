import datetime as dt
from google.cloud import bigquery


ROW_LIMIT = 10000


class BigQuery:
    def __init__(self):
        self.client = bigquery.Client()

    def get_sensor_data(
        self,
        installation_reference,
        node_id,
        sensor_type_reference,
        start=None,
        finish=None,
        row_limit=ROW_LIMIT,
    ):
        """Get sensor data for the given sensor type on the given node of the given installation over the given time
        period. The time period defaults to the last day.

        :param str installation_reference: the reference of the installation to get sensor data from
        :param str|None node_id: the node on the installation to get sensor data from
        :param str sensor_type_reference: the type of sensor from which to get the data
        :param datetime.datetime|None start: defaults to 1 day before the given finish
        :param datetime.datetime|None finish: defaults to the current datetime
        :param int|None row_limit: if set to `None`, no row limit is applied; if set to an integer, the row limit is set to this; defaults to 10000
        :return (pandas.Dataframe, bool): the sensor data and whether the data has been limited by a row limit
        """
        table_name = f"aerosense-twined.greta.sensor_data_{sensor_type_reference}"

        conditions = """
        WHERE datetime BETWEEN @start AND @finish
        AND installation_reference = @installation_reference
        AND node_id = @node_id
        """

        start, finish = self._get_time_period(start, finish)

        query_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "DATETIME", start),
                bigquery.ScalarQueryParameter("finish", "DATETIME", finish),
                bigquery.ScalarQueryParameter("installation_reference", "STRING", installation_reference),
                bigquery.ScalarQueryParameter("node_id", "STRING", node_id),
            ]
        )

        data_query = f"""
        SELECT *
        FROM `{table_name}`
        {conditions}
        ORDER BY datetime
        """

        data_limit_applied = False

        if row_limit:
            count_query = f"""
            SELECT COUNT(datetime)
            FROM `{table_name}`
            {conditions}
            """

            number_of_rows = list(self.client.query(count_query, job_config=query_config).result())[0][0]

            if number_of_rows > row_limit:
                data_query += f"\nLIMIT {row_limit}"
                data_limit_applied = True

        return (self.client.query(data_query, job_config=query_config).to_dataframe(), data_limit_applied)

    def get_sensor_data_at_datetime(
        self,
        installation_reference,
        node_id,
        sensor_type_reference,
        datetime,
        tolerance=1,
    ):
        """Get sensor data for the given sensor type on the given node of the given installation at the given datetime.
        The first datetime within a tolerance of ±0.5 * `tolerance` is used.

        :param str installation_reference: the reference of the installation to get sensor data from
        :param str|None node_id: the node on the installation to get sensor data from
        :param str sensor_type_reference: the type of sensor from which to get the data
        :param datetime.datetime|None datetime: the datetime to get the data at
        :param float tolerance: the tolerance on the given datetime in seconds
        :return pandas.Dataframe: the sensor data at the given datetime
        """
        table_name = f"aerosense-twined.greta.sensor_data_{sensor_type_reference}"

        start_datetime = datetime - dt.timedelta(seconds=tolerance / 2)
        finish_datetime = datetime + dt.timedelta(seconds=tolerance / 2)

        query_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_datetime", "DATETIME", start_datetime),
                bigquery.ScalarQueryParameter("finish_datetime", "DATETIME", finish_datetime),
                bigquery.ScalarQueryParameter("installation_reference", "STRING", installation_reference),
                bigquery.ScalarQueryParameter("node_id", "STRING", node_id),
            ]
        )

        query = f"""
        SELECT *
        FROM `{table_name}`
        WHERE datetime >= @start_datetime
        AND datetime < @finish_datetime
        AND installation_reference = @installation_reference
        AND node_id = @node_id
        ORDER BY datetime
        """

        return self.client.query(query, job_config=query_config).to_dataframe()

    def get_aggregated_connection_statistics(self, installation_reference, node_id, start=None, finish=None):
        """Get minute-wise aggregated connection statistics over the given time period. The time period defaults to the
        last day.

        :param str installation_reference: the reference of the installation to get sensor data from
        :param str|None node_id: the node on the installation to get sensor data from
        :param datetime.datetime|None start: defaults to 1 day before the given finish
        :param datetime.datetime|None finish: defaults to the current datetime
        :return pandas.Dataframe: the aggregated connection statistics
        """
        query = """
        SELECT datetime, filtered_rssi, raw_rssi, tx_power, allocated_heap_memory
        FROM `aerosense-twined.greta.connection_statistics_agg`
        WHERE datetime BETWEEN @start AND @finish
        AND installation_reference = @installation_reference
        AND node_id = @node_id
        ORDER BY datetime
        """

        start, finish = self._get_time_period(start, finish)

        query_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "DATETIME", start),
                bigquery.ScalarQueryParameter("finish", "DATETIME", finish),
                bigquery.ScalarQueryParameter("installation_reference", "STRING", installation_reference),
                bigquery.ScalarQueryParameter("node_id", "STRING", node_id),
            ]
        )

        return self.client.query(query, job_config=query_config).to_dataframe()

    def get_installations(self):
        """Query for all installations (without sensor coordinate data)"""
        installations_sql = """
        SELECT reference, turbine_id, location
        FROM `aerosense-twined.greta.installation`
        ORDER BY reference
        """

        installations = self.client.query(installations_sql).to_dataframe().to_dict(orient="records")

        return [
            {"label": f"{row['reference']} (Turbine {row['turbine_id']})", "value": row["reference"]}
            for row in installations
        ]

    def get_sensor_types(self):
        """Get the available sensor types.

        :return list:
        """
        query = """
        SELECT name
        FROM `aerosense-twined.greta.sensor_type`
        ORDER BY name
        """

        return self.client.query(query).to_dataframe()["name"].to_list()

    def get_nodes(self, installation_reference):
        """Get the IDs of the nodes available for the given installation.

        :param str installation_reference:
        :return list(str):
        """
        query = """
        SELECT node_id FROM `aerosense-twined.greta.sensor_data`
        WHERE installation_reference = @installation_reference
        GROUP BY node_id
        ORDER BY node_id
        """

        query_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("installation_reference", "STRING", installation_reference)]
        )

        return self.client.query(query, job_config=query_config).to_dataframe()["node_id"].to_list()

    def _get_time_period(self, start=None, finish=None):
        """Get the time period for the query. Defaults to the past day.

        :param datetime.datetime|None start: defaults to 1 day before the given finish
        :param datetime.datetime|None finish: defaults to the current datetime
        :return (datetime.datetime, datetime.datetime):
        """
        finish = finish or dt.datetime.now()
        start = start or finish - dt.timedelta(days=1)
        return start, finish
