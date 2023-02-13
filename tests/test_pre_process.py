import json
import os
import unittest
import datetime as dt
import numpy as np
import pandas as pd

from aerosense_tools.preprocess import RawSignal
from aerosense_tools.preprocess import SensorMeasurementSession

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))


class TestPreProcess(unittest.TestCase):
    def sample_timeseries(self, start, length, freq):
        time_vector = pd.date_range(start=start, end=start+length, freq=freq)
        sample_dataframe = pd.DataFrame(index=time_vector)
        sample_dataframe.index.name="datetime"
        sample_dataframe['test_data']=np.random.normal(0, 1, size=len(sample_dataframe))
        return sample_dataframe

    def example_metadata(self):
        metadata_path=os.path.join(REPOSITORY_ROOT, "example_installation_metadata.json")
        with open(metadata_path, "r") as in_file:
            session_metadata = json.load(in_file)
        return session_metadata

    def test_extract_measurement_session(self):
        data1 = self.sample_timeseries(dt.datetime(2020, 5, 4), dt.timedelta(seconds=1), dt.timedelta(seconds=0.1))
        data2 = self.sample_timeseries(dt.datetime(2020, 5, 5), dt.timedelta(seconds=1), dt.timedelta(seconds=0.1))
        data = data1.append(data2)

        signal=RawSignal(data, "test_sensor")
        measurement_sessions, measurement_session_times = signal.extract_measurement_sessions()

        self.assertEqual(len(measurement_session_times), 2)

    def test_merge_with_and_interpolate(self):
        data1 = self.sample_timeseries(dt.datetime(2020, 5, 4), dt.timedelta(seconds=1), dt.timedelta(seconds=0.1))
        data2 = self.sample_timeseries(dt.datetime(2020, 5, 4, 0, 0, 0, 50000), dt.timedelta(seconds=1), dt.timedelta(seconds=0.1))

        signal1 = RawSignal(data1, "test_sensor_1")
        signal2 = RawSignal(data2, "test_sensor_2")
        measurement_sessions1, _ = signal1.extract_measurement_sessions()
        measurement_sessions2, _ = signal2.extract_measurement_sessions()

        merged_df = measurement_sessions1[0].merge_with_and_interpolate(measurement_sessions2[0])






