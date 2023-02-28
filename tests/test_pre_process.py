import json
import os
import unittest
import datetime as dt
import numpy as np
import pandas as pd

from aerosense_tools.preprocess import RawSignal
from aerosense_tools.preprocess import SensorMeasurementSession

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))
TEST_START_TIME = dt.datetime(2020, 5, 4)
TEST_DURATION = dt.timedelta(seconds=1)
TEST_SAMPLING_RATE = dt.timedelta(seconds=0.1)


class TestPreProcess(unittest.TestCase):
    def sample_timeseries(self, start, length, freq):
        time_vector = pd.date_range(start=start, end=start+length, freq=freq)
        sample_dataframe = pd.DataFrame(index=time_vector)
        sample_dataframe.index.name="datetime"
        sample_dataframe['test_data_1'] = np.random.normal(0, 1, size=len(sample_dataframe))
        sample_dataframe['test_data_2'] = np.random.normal(0, 1, size=len(sample_dataframe))
        return sample_dataframe

    def base_data(self):
        base_data = self.sample_timeseries(TEST_START_TIME, TEST_DURATION, TEST_SAMPLING_RATE)
        return base_data

    def example_metadata(self):
        metadata_path=os.path.join(REPOSITORY_ROOT, "example_installation_metadata.json")
        with open(metadata_path, "r") as in_file:
            session_metadata = json.load(in_file)
        return session_metadata

    def test_extract_measurement_session(self):
        data1 = self.base_data()
        data2 = self.sample_timeseries(TEST_START_TIME+dt.timedelta(minutes=30), TEST_DURATION, TEST_SAMPLING_RATE)
        data = data1.append(data2)

        signal=RawSignal(data, "test_sensor")
        measurement_sessions, measurement_session_times = signal.extract_measurement_sessions()

        self.assertEqual(len(measurement_session_times), 2)

    def test_interpolate_to_constant_timestep(self):

        data = self.base_data()
        session = SensorMeasurementSession(data, "test_sensor")
        resampled_session = session.to_constant_timestep(
            TEST_SAMPLING_RATE,
            timeseries_start=TEST_START_TIME+dt.timedelta(seconds=0.05))

        self.assertAlmostEqual((data.iloc[0,0]+data.iloc[1,0])/2, resampled_session.dataframe.iloc[0,0],12)

    def test_merge_with_and_interpolate(self):

        data1 = self.base_data()
        data2 = self.sample_timeseries(
            TEST_START_TIME-dt.timedelta(seconds=0.05),
            TEST_DURATION-dt.timedelta(seconds=0.5),
            TEST_SAMPLING_RATE)

        signal1 = RawSignal(data1, "test_sensor_1")
        signal2 = RawSignal(data2, "test_sensor_2")
        measurement_sessions1, _ = signal1.extract_measurement_sessions()
        measurement_sessions2, _ = signal2.extract_measurement_sessions()

        merged_df = measurement_sessions1[0].merge_with_and_interpolate(measurement_sessions2[0])
        self.assertAlmostEqual((data2.iloc[0,0]+data2.iloc[1,0])/2, merged_df.iloc[0,2], 12)






