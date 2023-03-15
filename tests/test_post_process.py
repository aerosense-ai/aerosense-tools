import logging
import unittest
import datetime as dt
import numpy as np
import pandas as pd

from aerosense_tools.preprocess import SensorMeasurementSession
from aerosense_tools.postprocess import BladeIMU, PostProcess

from tests import TEST_DURATION, TEST_START_TIME, TEST_SAMPLING_STEP

logger = logging.getLogger(__name__)

class TestPostProcess(unittest.TestCase):

    def sample_timeseries(self, start, length, freq):
        """Creates a sample pandas dataframe with a datetime index and test azimuth angles.

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
        # Make 10 rotations
        sample_dataframe['test_azimuth'] = np.linspace(0, 10*2*np.pi, len(time_vector)) % (2*np.pi) * (-1)
        return sample_dataframe

    def sample_accelerometer(self):
        base_data = self.sample_timeseries(TEST_START_TIME, 10*TEST_DURATION, TEST_SAMPLING_STEP/10)
        base_data["Acc_Dir_1"]=base_data["test_azimuth"].apply(np.sin) * (-9.8)
        base_data["Acc_Dir_2"] = base_data["test_azimuth"].apply(np.cos) * (-9.8) - 10*9.8
        base_data["Acc_Dir_3"] = 0
        base_data.drop("test_azimuth", axis=1, inplace=True)
        accelerometer = SensorMeasurementSession(base_data, "accelerometer")
        return accelerometer

    def sample_gyrometer(self):
        base_data = self.sample_timeseries(TEST_START_TIME, 10*TEST_DURATION, TEST_SAMPLING_STEP/10)
        base_data["Gyro_Dir_1"] = 0
        base_data["Gyro_Dir_2"] = 0
        base_data["Gyro_Dir_3"] = - 2*np.pi
        base_data.drop("test_azimuth", axis=1, inplace=True)
        gyrometer = SensorMeasurementSession(base_data, "gyroscope")
        return gyrometer

    def test_imu_init(self):
        """Test that IMU class can be initialised from a merged session"""
        accelerometer = self.sample_accelerometer()
        gyrometer = self.sample_gyrometer()
        imu_data = accelerometer.merge_with(gyrometer)

        imu_time = (imu_data.index.astype(np.int64) / 10 ** 9) - imu_data.index[0].timestamp()

        imu=BladeIMU(
            time=imu_time,
            acc_mps2=imu_data[accelerometer.dataframe.columns].values.T,
            gyr_rps=imu_data[gyrometer.dataframe.columns].values.T,
        )

        self.assertIsInstance(imu, BladeIMU)


    def test_postprocess_imu(self):
        accelerometer = self.sample_accelerometer()
        gyrometer = self.sample_gyrometer()
        wind_turbine_metadata = {"hub_height": 30}
        imu_reference_frame = {"angles":[0 ,0, 0], "r-coordinate":10}
        processed_data = PostProcess.process_imu(accelerometer, gyrometer, imu_reference_frame, wind_turbine_metadata)

        self.assertTrue(processed_data["height"].iloc[0], 40)
