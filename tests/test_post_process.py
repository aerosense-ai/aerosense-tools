import unittest
import datetime as dt
import numpy as np
import pandas as pd

from aerosense_tools.preprocess import SensorMeasurementSession
from aerosense_tools.postprocess import BladeIMU, PostProcess

from tests import TEST_DURATION, TEST_START_TIME, TEST_SAMPLING_STEP


class TestPostProcess(unittest.TestCase):
    """Test that pre-process RawSignal and SensorMeasurementSession class methods are performing as expected."""
    def sample_timeseries(self, start, length, freq):
        """Creates a sample pandas dataframe with a datetime index and sinusoidal signals.

       Args:
           start dt.datetime: The start datetime for the index.
           length dt.timedelta: The length of the time series.
           freq dt.timedelta: The timestep with which the time series are sampled.

       Returns:
           pandas.DataFrame: A pandas dataframe with a datetime index and two columns of random normal data.
       """
        time_vector = pd.date_range(start=start, end=start+length, freq=freq)
        sample_dataframe = pd.DataFrame(index=time_vector)
        sample_dataframe.index.name="datetime"
        sample_dataframe['Dir_1'] = np.random.normal(0, 1, size=len(sample_dataframe))
        sample_dataframe['Dir_2'] = np.random.normal(0, 1, size=len(sample_dataframe))
        sample_dataframe['Dir_3'] = np.random.normal(0, 1, size=len(sample_dataframe))
        return sample_dataframe

    def base_data(self):
        """Creates a base pandas dataframe with a base datetime index and two columns of random normal data.

        Returns:
            pandas.DataFrame: A pandas dataframe with a datetime index and two columns of random normal data.
        """
        base_data = self.sample_timeseries(TEST_START_TIME, 60*TEST_DURATION, TEST_SAMPLING_STEP)
        return base_data

    def test_imu_init(self):
        """Test that IMU class can be initialised from a merged session"""
        accelerometer = SensorMeasurementSession(self.base_data(), "accelerometer")
        accelerometer.dataframe = accelerometer.dataframe.add_prefix("Acc_")
        gyrometer = SensorMeasurementSession(self.base_data(), "gyroscope")
        gyrometer.dataframe = gyrometer.dataframe.add_prefix("Gyro_")
        imu_data = accelerometer.merge_with_and_interpolate(gyrometer)

        imu_time = (imu_data.index.astype(np.int64) / 10 ** 9) - imu_data.index[0].timestamp()

        imu=BladeIMU(
            time=imu_time,
            acc_mps2=imu_data[accelerometer.dataframe.columns].values.T,
            gyr_rps=imu_data[gyrometer.dataframe.columns].values.T,
        )

        self.assertIsInstance(imu, BladeIMU)


