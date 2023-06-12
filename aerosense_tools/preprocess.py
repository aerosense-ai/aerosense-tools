import datetime as dt
import logging

import numpy as np
import pandas as pd
from plotly import express as px

from aerosense_tools.exceptions import EmptyDataFrameError


logger = logging.getLogger(__name__)


class RawData:
    """A class representing raw data received from a specific sensor through the data gateway."""

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

    def convert_int_to_measurement(self, sensor_conversion_constants=None):
        """Convert int value (originally from bytes) to a float value representing a measured physical
        variable in SI units.

        :param dict sensor_conversion_constants: dictionary containing calibrated conversion constants
        :return pandas.Dataframe: inplace transformed dataframe with raw data transformed to variable values
        """

        gravitational_acceleration = 9.81  # [m/s²]
        differential_pressure_range = 2 * 6000
        differential_pressure_offset = 32767
        gyroscope_sensitivity = 16.4  # Typ. Gyro sensitivity LSB/deg/s
        accelerometer_sensitivity = 2048  # Typ. Accelerometer sensitivity LSB/g

        default_conversion_constants = {
            "barometer": 40.96,  # [Pa]⁻¹
            "barometer_thermometer": 100,  # [Celsius]⁻¹
            "differential_barometer": (58982 - 6553) / differential_pressure_range,  # [Pa]⁻¹
            "differential_barometer_offset": differential_pressure_offset,
            "accelerometer": accelerometer_sensitivity / gravitational_acceleration,  # [m/s²]⁻¹
            "gyroscope": gyroscope_sensitivity * 180 / np.pi,  # [s⁻¹] ⁻¹
            "magnetometer": 1,  # TODO this is tbd.
        }
        sensor_conversion_constants = sensor_conversion_constants or default_conversion_constants

        if self.sensor_type == "differential_barometer":
            self.dataframe -= sensor_conversion_constants["differential_barometer_offset"]

        self.dataframe /= sensor_conversion_constants[self.sensor_type]

    def extract_measurement_sessions(self, threshold=dt.timedelta(seconds=60)):
        """Extract sessions (continuous measurement periods) from raw data.

        :param datetime.timedelta threshold: Maximum gap between two consecutive measurement samples :return  (list,
        pandas.DataFrame): List with SensorMeasurementSession objects, a dataframe with sessions' information such as
        start and end times, duration, sensor statistics
        """
        measurement_sessions = []
        session_statistics = []
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
                "start": sample_time["datetime"][start_rows].to_list(),
                "end": sample_time["datetime"][end_rows].to_list(),
            }
        )

        for start_row, end_row in zip(start_rows, end_rows):
            session = SensorMeasurementSession(self.dataframe.iloc[start_row:end_row, :], self.sensor_type)
            measurement_sessions.append(session)
            session_statistics.append(session.sensor_statistics.stack())

        session_times["duration"] = session_times["end"] - session_times["start"]
        session_information = pd.concat([session_times, pd.DataFrame(session_statistics)], axis=1)

        return measurement_sessions, session_information


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

    def merge_with(self, *secondary_sessions):
        """Merge current session's sensor measurements with measurements from other sensors (secondary sessions)
        The values from the secondary sessions will be interpolated onto the current session's time vector.

        :return Pandas.DataFrame: Merged dataframe
        """
        merged_df = self.dataframe
        for secondary_session in secondary_sessions:
            merged_df = pd.concat([merged_df, secondary_session.dataframe], axis=1)
            merged_df = merged_df.interpolate("index", limit_area="inside").reindex(self.dataframe.index)
        return merged_df

    def trim(self, from_start=dt.timedelta(), from_end=dt.timedelta()):
        """Delete first and last measurements from the session

        :param datetime.timedelta from_start: Amount of time to trim from the start of the session
        :param datetime.timedelta from_end: Amount of time to trim from the end of the session
        :return SensorMeasurementSession: sensor session with trimmed_dataframe
        """
        time_window = (self.dataframe.index > self.start + from_start) & (self.dataframe.index < self.end - from_end)

        trimmed_dataframe = self.dataframe[time_window]
        return SensorMeasurementSession(trimmed_dataframe, self.sensor_type)

    def filter_outliers(self, window, acceptable_deviation):
        """A very primitive filter. Removes data points outside the confidence interval of a rolling median +/-
        acceptable_deviation.

        :param int window: window (number of samples) for rolling median and standard deviation
        :param float acceptable_deviation: multiplier to the rolling standard deviation
        :return pandas.Dataframe: filtered dataframe
        """
        rolling_median = self.dataframe.rolling(window).median()

        filtered_dataframe = self.dataframe[
            (self.dataframe <= rolling_median + acceptable_deviation)
            & (self.dataframe >= rolling_median - acceptable_deviation)
        ]

        return filtered_dataframe

    def plot(self, sensor_types_metadata, sensor_names=None, plot_start_offset=dt.timedelta(), plot_max_time=None):
        """Plots the session dataframe with plotly.

        :param sensor_types_metadata: Metadata about the sensor type, used for figure layout
        :param sensor_names: Specific sensors to plot
        :param plot_start_offset: start data plot after some time
        :param plot_max_time: limit to the time plotted
        :return plotly.graph_objs.Figure: a line graph of the sensor data against time
        """
        plot_max_time = plot_max_time or self.duration - plot_start_offset
        plot_df = self.trim(plot_start_offset, self.duration - plot_start_offset - plot_max_time).dataframe

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
