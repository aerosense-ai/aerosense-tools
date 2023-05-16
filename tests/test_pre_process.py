import datetime as dt
import json
import os
import unittest

import numpy as np
import pandas as pd

from aerosense_tools.preprocess import RawSignal, SensorMeasurementSession


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))
TEST_START_TIME = dt.datetime(2020, 5, 4)
TEST_DURATION = dt.timedelta(seconds=1)
TEST_SAMPLING_STEP = dt.timedelta(seconds=0.1)


class TestPreProcess(unittest.TestCase):
    """Test that pre-process RawSignal and SensorMeasurementSession class methods are performing as expected."""

    def sample_timeseries(self, start, length, freq):
        """Creates a sample pandas dataframe with a datetime index and two columns of random normal data.

        Args:
            start dt.datetime: The start datetime for the index.
            length dt.timedelta: The length of the time series.
            freq dt.timedelta: The timestep with which the time series are sampled.

        Returns:
            pandas.DataFrame: A pandas dataframe with a datetime index and two columns of random normal data.
        """
        time_vector = pd.date_range(start=start, end=start + length, freq=freq)
        sample_dataframe = pd.DataFrame(index=time_vector)
        sample_dataframe.index.name = "datetime"
        sample_dataframe["test_data_1"] = np.random.normal(0, 1, size=len(sample_dataframe))
        sample_dataframe["test_data_2"] = np.random.normal(0, 1, size=len(sample_dataframe))
        return sample_dataframe

    def base_data(self):
        """Creates a base pandas dataframe with a base datetime index and two columns of random normal data.

        Returns:
            pandas.DataFrame: A pandas dataframe with a datetime index and two columns of random normal data.
        """
        base_data = self.sample_timeseries(TEST_START_TIME, TEST_DURATION, TEST_SAMPLING_STEP)
        return base_data

    def example_metadata(self):
        """Reads in an example installation metadata file in JSON format.

        Returns:
            dict: A dictionary containing the metadata."""
        metadata_path = os.path.join(REPOSITORY_ROOT, "example_installation_metadata.json")
        with open(metadata_path, "r") as in_file:
            session_metadata = json.load(in_file)
        return session_metadata

    def test_classes_with_empty_dataframe(self):
        """Test that instantiating the RawSignal and SensorMeasurementSession classes with an empty dataframe raises
        a ValueError."""
        data = pd.DataFrame()
        self.assertRaises(ValueError, RawSignal, data, "test_sensor")
        self.assertRaises(ValueError, SensorMeasurementSession, data, "test_session")

    def test_extract_measurement_session(self):
        """Check if RawSignal.extract_measurement_sessions() method correctly extracts two measurement sessions from
        a DataFrame containing two distinct continuous measurements and that the second session start time is correct"""
        data1 = self.base_data()
        data2 = self.sample_timeseries(TEST_START_TIME + dt.timedelta(minutes=30), TEST_DURATION, TEST_SAMPLING_STEP)
        data = data1.append(data2)

        signal = RawSignal(data, "test_sensor")
        measurement_sessions, measurement_session_times = signal.extract_measurement_sessions()

        self.assertEqual(measurement_session_times["start"].loc[1], TEST_START_TIME + dt.timedelta(minutes=30))
        self.assertEqual(len(measurement_sessions), 2)

    def test_interpolate_to_constant_timestep(self):
        """Check if SensorMeasurementSession.to_constant_timestep() method correctly resamples a session to a
        constant timestep. Verify that the resampled DataFrame does not contain extrapolated values, and that the first
        resampled value in the new Dataframe is accurate."""
        data = self.base_data()
        session = SensorMeasurementSession(data, "test_sensor")
        resampled_session = session.to_constant_timestep(
            TEST_SAMPLING_STEP / 2, timeseries_start=TEST_START_TIME - dt.timedelta(seconds=0.08)
        )

        self.assertTrue(pd.isna(resampled_session.dataframe.iloc[0, 0]))
        self.assertAlmostEqual(
            0.8 * data.iloc[0, 0] + 0.2 * data.iloc[1, 0], resampled_session.dataframe.iloc[2, 0], 12
        )

    def test_merge_with_and_interpolate(self):
        """Check if SensorMeasurementSession.merge_with_and_interpolate() method correctly merges two measurement
        sessions with different sampling rates and interpolates the resulting DataFrame to a time vector from a
        primary session. Verify that the merged DataFrame does not contain extrapolated values, and that the first
        interpolated value in the new Dataframe is accurate."""
        data1 = self.base_data()
        data2 = self.sample_timeseries(
            TEST_START_TIME - dt.timedelta(seconds=0.05), TEST_DURATION - dt.timedelta(seconds=0.5), TEST_SAMPLING_STEP
        )

        signal1 = RawSignal(data1, "test_sensor_1")
        signal2 = RawSignal(data2, "test_sensor_2")
        measurement_sessions1, _ = signal1.extract_measurement_sessions()
        measurement_sessions2, _ = signal2.extract_measurement_sessions()

        merged_df = measurement_sessions1[0].merge_with_and_interpolate(measurement_sessions2[0])
        self.assertTrue(pd.isna(merged_df.iloc[-1, -1]))
        self.assertAlmostEqual((data2.iloc[0, 0] + data2.iloc[1, 0]) / 2, merged_df.iloc[0, 2], 12)
