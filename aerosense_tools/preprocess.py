import datetime as dt
import logging

import numpy as np
import pandas as pd
from plotly import express as px

from aerosense_tools.exceptions import EmptyDataFrameError


logger = logging.getLogger(__name__)


class RawSignal:
    """A class representing raw data received from data gateway."""

    def __init__(self, dataframe, sensor_type):
        if dataframe.empty:
            raise EmptyDataFrameError("Empty DataFrame is not allowed for the RawSignal Class")

        self.dataframe = dataframe
        self.sensor_type = sensor_type

    def pad_gaps(self, threshold=dt.timedelta(seconds=60)):
        """Checks for missing data. If the gap between samples (timedelta) is higher than the given threshold, then the
        last sample before the gap start is replaced with NaN. Thus, no interpolation will be performed during the
        non-sampling time window.

        :param datetime.timedelta threshold: maximum gap between two samples as a timedelta type
        :return pandas.Dataframe: inplace modified dataframe with the end sample of each session replaced with NaN
        """
        self.dataframe[self.dataframe.index.to_series().diff() > threshold] = np.NaN

    def filter_outliers(self, window, standard_deviation_multiplier):
        """A very primitive filter. Removes data points outside the confidence interval using a rolling median and
        standard deviation.

        :param int window: window (number of samples) for rolling median and standard deviation
        :param float standard_deviation_multiplier: multiplier to the rolling standard deviation
        :return pandas.Dataframe: inplace filtered dataframe
        """
        rolling_median = self.dataframe.rolling(window).median()
        rolling_std = self.dataframe.rolling(window).std()

        # TODO define filtering rule using rolling df here
        self.dataframe = self.dataframe[
            (self.dataframe <= rolling_median + standard_deviation_multiplier * rolling_std)
            & (self.dataframe >= rolling_median - standard_deviation_multiplier * rolling_std)
        ]

    def measurement_to_variable(self, sensor_conversion_constants=None):
        """Transform fixed point values to a physical variable.

        :param dict sensor_conversion_constants: dictionary containing calibrated conversion constants
        :return pandas.Dataframe: inplace transformed dataframe with raw values transformed to variable values
        """
        # TODO These values should be picked up from the session configuration metadata
        # TODO Refactor the name of the function to values_to_variables
        gravitational_acceleration = 9.81  # [m/s²]
        differential_pressure_range = 2 * 6000
        differential_pressure_offset = 32767
        gyroscope_sensitivity = 16.4  # Typ Gyro sensitivity LSB/deg/s
        accelerometer_sensitivity = 2048  # Typ. Accelerometer sensitivity LSB/g

        default_conversion_constants = {
            "barometer": 40.96,  # [Pa]
            "barometer_thermometer": 100,  # [Celsius]
            "differential_barometer": (58982 - 6553) / differential_pressure_range,  # [Pa]
            "accelerometer": accelerometer_sensitivity / gravitational_acceleration,  # [m/s²]
            "gyroscope": gyroscope_sensitivity * 180 / np.pi,  # [s⁻¹]
            "magnetometer": 1,  # TODO this is tbd.
        }
        sensor_conversion_constants = sensor_conversion_constants or default_conversion_constants

        if self.sensor_type == "differential_barometer":
            self.dataframe -= differential_pressure_offset

        self.dataframe /= sensor_conversion_constants[self.sensor_type]

    def extract_measurement_sessions(self, threshold=dt.timedelta(seconds=60)):
        """Extract sessions (continuous measurement periods) from raw data.

        :param datetime.timedelta threshold: Maximum gap between two consecutive measurement samples
        :return (list, pandas.DataFrame): List with SensorMeasurementSession objects, a dataframe with sessions' start and end times
        """
        measurement_sessions = []
        sample_time = pd.DataFrame(self.dataframe.index)

        session_starts = sample_time["datetime"].diff() > threshold
        session_ends = abs(sample_time["datetime"].diff(-1)) > threshold
        session_starts.iloc[0] = session_ends.iloc[-1] = True

        # Edge case of a single measurement point:
        is_single_sample = sample_time.index[session_starts] == sample_time.index[session_ends]

        if any(is_single_sample):
            logger.warning(
                "Sensor type {} has single measurement points at {}".format(
                    self.sensor_type,
                    sample_time[session_starts][is_single_sample],
                )
            )

        start_rows = sample_time.index[session_starts][~is_single_sample]
        end_rows = sample_time.index[session_ends][~is_single_sample]

        session_times = pd.DataFrame(
            {
                "start_datetime": sample_time["datetime"][start_rows].to_list(),
                "finish_datetime": sample_time["datetime"][end_rows].to_list(),
            }
        )

        for start_row, end_row in zip(start_rows, end_rows):
            measurement_sessions.append(
                SensorMeasurementSession(self.dataframe.iloc[start_row:end_row, :], self.sensor_type)
            )

        return measurement_sessions, session_times


class SensorMeasurementSession:
    """A class representing continuous measurement series for a particular sensor. The class wraps some frequently used
    Pandas.DataFrame operations as well as plotly figure setup.
    """

    def __init__(self, dataframe, sensor_type):
        if dataframe.empty:
            raise EmptyDataFrameError("Empty DataFrame is not allowed for the SensorMeasurementSession Class")

        self.dataframe = dataframe
        self.sensor_type = sensor_type
        self.start = dataframe.index[0]
        self.end = dataframe.index[-1]
        self.duration = dataframe.index[-1] - dataframe.index[0]
        self.sensor_statistics = dataframe.agg(["min", "max", "mean", "std"])

    def to_constant_timestep(self, time_step, timeseries_start=None):
        """Resample dataframe to the given time step. Linearly interpolates between samples.

        :param datetime.timedelta time_step: timestep as datetime timedelta type
        :param datetime.datetime timeseries_start: start constant step time series at specified time
        :return SensorMeasurementSession: sensor session with resampled and interpolated data
        """
        new_time_vector = pd.date_range(start=timeseries_start or self.start, end=self.end, freq=time_step)
        return self.to_new_time_vector(new_time_vector)

    def to_new_time_vector(self, new_time_vector):
        """Interpolate the original dataframe onto a new time index.

        :param pandas.DatetimeIndex new_time_vector: the new time index
        :return SensorMeasurementSession: a new SensorMeasurementSession object with the interpolated dataframe
        """
        new_dataframe = pd.DataFrame(index=new_time_vector)
        new_dataframe = pd.concat([self.dataframe, new_dataframe], axis=1)
        new_dataframe = new_dataframe.interpolate("index", limit_area="inside").reindex(new_time_vector)
        return SensorMeasurementSession(new_dataframe, self.sensor_type)

    def merge_with_and_interpolate(self, *secondary_sessions):
        """Merge current session's sensor measurements with measurements from other sensors (secondary sessions)
        The values from the secondary sessions will be interpolated onto the current session's time vector.

        :return Pandas.DataFrame: Merged dataframe
        """
        for secondary_session in secondary_sessions:
            merged_df = pd.concat([self.dataframe, secondary_session.dataframe], axis=1)
            merged_df = merged_df.interpolate("index", limit_area="inside").reindex(self.dataframe.index)

        return merged_df

    def trim_session(self, trim_from_start=dt.timedelta(), trim_from_end=dt.timedelta()):
        """Delete first and last measurements from the session

        :param datetime.timedelta trim_from_start: Amount of time to trim from the start of the session
        :param datetime.timedelta trim_from_end: Amount of time to trim from the end of the session
        :return SensorMeasurementSession: sensor session with trimmed_dataframe
        """
        time_window = (self.dataframe.index > self.start + trim_from_start) & (
            self.dataframe.index < self.end - trim_from_end
        )

        trimmed_dataframe = self.dataframe[time_window]
        return SensorMeasurementSession(trimmed_dataframe, self.sensor_type)

    def plot(self, sensor_types_metadata, sensor_names=None, plot_start_offset=dt.timedelta(), plot_max_time=None):
        """Plots the session dataframe with plotly.

        :param sensor_types_metadata: Metadata about the sensor type, used for figure layout
        :param sensor_names: Specific sensors to plot
        :param plot_start_offset: start data plot after some time
        :param plot_max_time: limit to the time plotted
        :return plotly.graph_objs.Figure: a line graph of the sensor data against time
        """
        plot_max_time = plot_max_time or self.duration - plot_start_offset
        plot_df = self.trim_session(plot_start_offset, self.duration - plot_start_offset - plot_max_time).dataframe

        if sensor_names:
            plot_df = plot_df[sensor_names]

        layout = {
            "title": sensor_types_metadata[self.sensor_type]["description"],
            "xaxis_title": "Date/Time",
            "yaxis_title": sensor_types_metadata[self.sensor_type]["variable"],
            "legend_title": "Sensor",
        }

        figure = px.line(plot_df)
        figure.update_layout(layout)
        return figure
