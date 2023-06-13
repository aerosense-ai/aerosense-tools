import datetime
import datetime as dt
import json
import logging
import os

import jsonschema
from google.cloud import bigquery
from octue.cloud.storage import GoogleCloudStorageClient

from aerosense_tools.preprocess import RawSignal
from aerosense_tools.utils import remove_metadata_columns_and_set_datetime_index


logger = logging.getLogger(__name__)


DATASET_NAME = "aerosense-twined.greta"
ROW_LIMIT = 10000
SENSOR_COORDINATES_SCHEMA_URI = "https://jsonschema.registry.octue.com/aerosense/sensor-coordinates/0.1.4.json"


class BigQuery:
    """A collection of queries for working with the Aerosense BigQuery dataset.

    :param str project_name: the name of the Google Cloud project the BigQuery dataset belongs to
    :return None:
    """

    def __init__(self, project_name="aerosense-twined"):
        self.client = bigquery.Client(project=project_name)

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
        :param str node_id: the node on the installation to get sensor data from
        :param str sensor_type_reference: the type of sensor from which to get the data
        :param datetime.datetime|None start: defaults to 1 day before the given finish
        :param datetime.datetime|None finish: defaults to the current datetime
        :param int|None row_limit: if set to `None`, no row limit is applied; if set to an integer, the row limit is set to this; defaults to 10000
        :return (pandas.Dataframe, bool): the sensor data and whether the data has been limited by a row limit
        """
        table_name = f"{DATASET_NAME}.sensor_data_{sensor_type_reference}"

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
        :param str node_id: the node on the installation to get sensor data from
        :param str sensor_type_reference: the type of sensor from which to get the data
        :param datetime.datetime|None datetime: the datetime to get the data at
        :param float tolerance: the tolerance on the given datetime in seconds
        :return pandas.Dataframe: the sensor data at the given datetime
        """
        table_name = f"{DATASET_NAME}.sensor_data_{sensor_type_reference}"

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
        :param str node_id: the node on the installation to get sensor data from
        :param datetime.datetime|None start: defaults to 1 day before the given finish
        :param datetime.datetime|None finish: defaults to the current datetime
        :return pandas.Dataframe: the aggregated connection statistics
        """
        query = f"""
        SELECT datetime, filtered_rssi, raw_rssi, tx_power, allocated_heap_memory
        FROM `{DATASET_NAME}.connection_statistics_agg`
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

    def get_microphone_metadata(self, installation_reference, node_id, start=None, finish=None):
        """Get metadata for microphone data for the given node of the given installation over the given time period. The
        time period defaults to the last day.

        :param str installation_reference: the reference of the installation to get microphone metadata from
        :param str node_id: the node on the installation to get microphone metadata from
        :param datetime.datetime|None start: the start of the time period; defaults to 1 day before the given finish
        :param datetime.datetime|None finish: the end of the time period; defaults to the current datetime
        :return pandas.Dataframe: the microphone metadata
        """
        query = f"""
        SELECT *
        FROM `{DATASET_NAME}.microphone_data`
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

        return self.client.query(query, job_config=query_config).to_dataframe()

    def download_microphone_data_at_datetime(self, installation_reference, node_id, datetime, tolerance=1):
        """Download the microphone datafile for the given node of the given installation at the given datetime (within
        the given tolerance). If more than one datetime is found within the tolerance, the datafile with the earliest
        timestamp is downloaded.

        :param str installation_reference: the reference of the installation to get microphone data from
        :param str node_id: the node on the installation to get microphone data from
        :param datetime.datetime datetime: the datetime to get the data for
        :param float tolerance: the tolerance on the given datetime in seconds
        :return str: the local path the microphone datafile was downloaded to
        """
        start_datetime = datetime - dt.timedelta(seconds=tolerance / 2)
        finish_datetime = datetime + dt.timedelta(seconds=tolerance / 2)

        microphone_metadata = self.get_microphone_metadata(
            installation_reference=installation_reference,
            node_id=node_id,
            start=start_datetime,
            finish=finish_datetime,
        )

        cloud_path = microphone_metadata.iloc[0]["path"]
        extension = os.path.splitext(cloud_path)[-1]
        local_path = f"microphone-data-{datetime.isoformat()}" + extension

        GoogleCloudStorageClient().download_to_file(local_path=local_path, cloud_path=cloud_path)
        return os.path.abspath(local_path)

    def get_installations(self):
        """Get the available installations.

        :return list(dict): the available installations
        """
        query = f"""
        SELECT reference, turbine_id, location
        FROM `{DATASET_NAME}.installation`
        ORDER BY reference
        """

        installations = self.client.query(query).to_dataframe().to_dict(orient="records")

        return [
            {"label": f"{row['reference']} (Turbine {row['turbine_id']})", "value": row["reference"]}
            for row in installations
        ]

    def get_sensor_types(self):
        """Get the available sensor types and their metadata.

        :return list(dict): the available sensor types and their metadata
        """
        query = f"""
        SELECT name, metadata
        FROM `{DATASET_NAME}.sensor_type`
        ORDER BY name
        """

        return {
            sensor_type["name"]: json.loads(sensor_type["metadata"])
            for sensor_type in self.client.query(query).to_dataframe().to_dict("records")
        }

    def get_nodes(self, installation_reference):
        """Get the IDs of the nodes installed on the given installation.

        :param str installation_reference: the reference of the installation to get the node IDs of
        :return list(str): the node IDs for the installation
        """
        query = f"""
        SELECT node_id FROM `{DATASET_NAME}.sensor_data`
        WHERE installation_reference = @installation_reference
        GROUP BY node_id
        ORDER BY node_id
        """

        query_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("installation_reference", "STRING", installation_reference)]
        )

        return self.client.query(query, job_config=query_config).to_dataframe()["node_id"].to_list()

    def add_sensor_coordinates(self, coordinates):
        """Add the given sensor coordinates to the sensor coordinates table.

        :param dict coordinates: the sensor coordinates
        :return None:
        """
        jsonschema.validate(coordinates, {"$ref": SENSOR_COORDINATES_SCHEMA_URI})

        if self.get_sensor_coordinates(coordinates["reference"]):
            raise ValueError(f"Sensor coordinates with the reference {coordinates['reference']!r} already exist.")

        errors = self.client.insert_rows(
            table=self.client.get_table(DATASET_NAME + ".sensor_coordinates"),
            rows=[
                {
                    "reference": coordinates["reference"],
                    "kind": coordinates["kind"],
                    "geometry": json.dumps(coordinates["geometry"]),
                }
            ],
        )

        if errors:
            raise ValueError(errors)

    def update_sensor_coordinates(self, reference, kind, geometry):
        """Update the given sensor coordinates in the sensor coordinates table.

        :param str reference: the reference of the coordinates to update
        :param str kind: the kind of the new coordinates
        :param dict geometry: the new coordinates
        :return None:
        """
        coordinates = {"reference": reference, "kind": kind, "geometry": geometry}
        jsonschema.validate(coordinates, {"$ref": SENSOR_COORDINATES_SCHEMA_URI})

        query_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("kind", "STRING", kind),
                bigquery.ScalarQueryParameter("geometry", "JSON", json.dumps(geometry)),
                bigquery.ScalarQueryParameter("reference", "STRING", reference),
            ]
        )

        self.client.query(
            f"""UPDATE {DATASET_NAME}.sensor_coordinates
            SET kind = @kind, geometry = @geometry
            WHERE reference = @reference;
            """,
            job_config=query_config,
        )

    def get_sensor_coordinates(self, reference=None):
        """Get the sensor coordinates with the given reference from the sensor coordinates table if they exist. If no
        reference is given, get all sensor coordinates.

        :param str|None reference: the reference of the coordinates to get
        :return dict|None: the sensor coordinates if they exist
        """
        if not reference:
            return self.client.query(f"SELECT * FROM {DATASET_NAME}.sensor_coordinates").result().to_dataframe()

        else:
            query_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("reference", "STRING", reference)]
            )

            result = self.client.query(
                f"""SELECT * FROM {DATASET_NAME}.sensor_coordinates
                WHERE reference = @reference;
                """,
                job_config=query_config,
            ).result()

        if result.total_rows == 0:
            return None

        result = dict(result.to_dataframe().iloc[0])
        result["geometry"] = json.loads(result["geometry"])
        return result

    def get_measurement_sessions(
        self,
        installation_reference,
        node_id,
        sensor_type_reference,
        start=None,
        finish=None,
    ):
        """Get the measurement sessions that exist for the given sensor type, node, and installation between the given
        start and finish datetimes.

        :param str installation_reference: the reference of the installation to get measurement sessions for
        :param str node_id: the ID of the node to get measurement sessions for
        :param str sensor_type_reference: the type of sensor to get measurement sessions for
        :param datetime.datetime|None start: the time after which the sessions start
        :param datetime.datetime|None finish: the time before which the sessions end
        :return pandas.DataFrame: the measurement sessions
        """
        start, finish = self._get_time_period(start, finish)

        query_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("installation_reference", "STRING", installation_reference),
                bigquery.ScalarQueryParameter("node_id", "STRING", node_id),
                bigquery.ScalarQueryParameter("sensor_type_reference", "STRING", sensor_type_reference),
                bigquery.ScalarQueryParameter("start", "DATETIME", start),
                bigquery.ScalarQueryParameter("finish", "DATETIME", finish),
            ]
        )

        result = self.client.query(
            f"""
            SELECT * FROM {DATASET_NAME}.sessions
            WHERE installation_reference = @installation_reference
            AND node_id = @node_id
            AND sensor_type_reference = @sensor_type_reference
            AND start_datetime > @start
            AND finish_datetime <= @finish
            ORDER BY start_datetime
            """,
            job_config=query_config,
        ).result()

        return result.to_dataframe()

    def extract_and_add_new_measurement_sessions(self, sensors=None):
        """Extract new measurement sessions from the database for the given sensors and add them to the sessions table.
        If no sensors are given, sessions for the following sensors are searched for:
        - connection_statistics
        - magnetometer
        - connection
        - barometer
        - barometer_thermometer
        - accelerometer
        - gyroscope
        - battery_info
        - differential_barometer

        :param list(str)|None sensors: the sensors to search for new measurement sessions for
        :return None:
        """
        table_name = DATASET_NAME + ".sessions"

        sensors = sensors or (
            "connection_statistics",
            "magnetometer",
            "connection-statistics",
            "barometer",
            "barometer_thermometer",
            "accelerometer",
            "gyroscope",
            "battery_info",
            "differential_barometer",
        )

        installations = [installation["value"] for installation in self.get_installations()]

        for installation_reference in installations:
            nodes = self.get_nodes(installation_reference)

            for node_id in nodes:
                for sensor_type_reference in sensors:
                    logger.info(
                        "Getting latest extracted session finish datetime for installation %r, node %r, sensor type %r.",
                        installation_reference,
                        node_id,
                        sensor_type_reference,
                    )

                    result = (
                        self.client.query(
                            f"""
                            SELECT finish_datetime FROM {table_name}
                            WHERE installation_reference = @installation_reference
                            AND node_id = @node_id
                            AND sensor_type_reference = sensor_type_reference
                            ORDER BY finish_datetime DESC
                            LIMIT 1
                            """,
                            job_config=bigquery.QueryJobConfig(
                                query_parameters=[
                                    bigquery.ScalarQueryParameter(
                                        "installation_reference", "STRING", installation_reference
                                    ),
                                    bigquery.ScalarQueryParameter("node_id", "STRING", node_id),
                                    bigquery.ScalarQueryParameter(
                                        "sensor_type_reference", "STRING", sensor_type_reference
                                    ),
                                ]
                            ),
                        )
                        .result()
                        .to_dataframe()
                    )

                    try:
                        latest_session_finish_datetime = result.iloc[0]["finish_datetime"].to_pydatetime()
                    except IndexError:
                        logger.info(
                            "No new sessions available for installation %r, node %r, sensor type %r.",
                            installation_reference,
                            node_id,
                            sensor_type_reference,
                        )
                        continue

                    sensor_data_df, _ = self.get_sensor_data(
                        installation_reference=installation_reference,
                        node_id=node_id,
                        sensor_type_reference=sensor_type_reference,
                        start=latest_session_finish_datetime,
                        finish=datetime.datetime.now(),
                    )

                    if sensor_data_df.empty:
                        logger.info(
                            "No new sessions available for installation %r, node %r, sensor type %r.",
                            installation_reference,
                            node_id,
                            sensor_type_reference,
                        )
                        continue

                    sensor_data_df = remove_metadata_columns_and_set_datetime_index(sensor_data_df)

                    _, measurement_sessions = RawSignal(
                        dataframe=sensor_data_df,
                        sensor_type=sensor_type_reference,
                    ).extract_measurement_sessions()

                    # Add columns needed for sessions table.
                    measurement_sessions["installation_reference"] = installation_reference
                    measurement_sessions["node_id"] = node_id
                    measurement_sessions["sensor_type_reference"] = sensor_type_reference

                    # Reorder columns.
                    measurement_sessions = measurement_sessions[
                        [
                            "installation_reference",
                            "node_id",
                            "sensor_type_reference",
                            "start_datetime",
                            "finish_datetime",
                        ]
                    ]

                    # Add new sessions to sessions table.
                    self.client.load_table_from_dataframe(
                        dataframe=measurement_sessions,
                        destination=table_name,
                    ).result()

    def query(self, query_string):
        """Query the dataset with an arbitrary query.

        :param str query_string: the query to use
        :return pd.DataFrame: the results of the query
        """
        return self.client.query(query_string).to_dataframe()

    def _get_time_period(self, start=None, finish=None):
        """Get a time period of:
        - The past day if no arguments are given
        - The given start until the current datetime if only the start is given
        - The day previous to the finish if only the finish is given
        - The given start and finish if both are given

        :param datetime.datetime|None start: defaults to 1 day before the given finish
        :param datetime.datetime|None finish: defaults to the current datetime
        :return (datetime.datetime, datetime.datetime):
        """
        finish = finish or dt.datetime.now()
        start = start or finish - dt.timedelta(days=1)
        return start, finish
