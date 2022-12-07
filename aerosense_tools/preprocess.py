import logging
import numpy as np
import pandas as pd
import datetime as dt
from scipy.interpolate import interp1d
from aerosense_tools.plots import plot_with_layout

logger = logging.getLogger(__name__)


class RawSignal:

    def __init__(self, dataframe, sensor_type):
        self.dataframe = dataframe
        self.sensor_type = sensor_type

    def pad_gaps(self, threshold=dt.timedelta(seconds=60)):
        """Checks for missing data. If the gap between samples (timedelta) is
         higher than the given threshold, then the last sample before the gap
        start is replaced with NaN. Thus, no interpolation will be performed
        during the non-sampling time window.

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
        rolling_std = self.dataframe.rolataframeling(window).std()
        # TODO define filtering rule using rolling df here
        self.dataframe = self.dataframe[
            (self.dataframe <= rolling_median + standard_deviation_multiplier * rolling_std)
            & (self.dataframe >= rolling_median - standard_deviation_multiplier * rolling_std)
            ]

    def measurement_to_variable(self):
        """Transform fixed point values to a physical variable.

        :return pandas.Dataframe: inplace transformed dataframe with raw values transformed to variable values
        """
        # TODO These values should be picked up from the session configuration metadata
        # TODO Refactor the name of the function to values_to_variables
        diffrange = 2 * 6000

        if self.sensor_type == "barometer":
            self.dataframe /= 40.96  # [Pa]
        if self.sensor_type == "barometer_thermometer":
            self.dataframe /= 100  # [Celsius]
        if self.sensor_type == "differential_barometer":
            self.dataframe -= 32767
            self.dataframe *= diffrange/(58982-6553)

    def extract_measurement_sessions(self, threshold=dt.timedelta(seconds=60)):
        """Extract sessions (continuous measurement periods) from raw data.

        :param datetime.timedelta threshold: Maximum gap between two consecutive measurement samples
        :return  (list, pandas.DataFrame): List with SensorMeasurementSession objects, a dataframe with sessions' start and end times
        """

        measurement_sessions = []
        sample_time = pd.DataFrame(self.dataframe.index)

        session_starts = sample_time['datetime'].diff() > threshold
        session_ends = abs(sample_time['datetime'].diff(-1)) > threshold
        session_starts.iloc[0] = session_ends.iloc[-1] = True

        # Not sure if concat is the best way, but will do for now
        sessions = pd.concat([sample_time[session_starts], sample_time[session_ends]], axis=1)
        sessions.columns = ['start', 'end']

        # Edge case of a single measurement point:
        if any(sessions['start'] == sessions['end']):
            logger.warning('Sensor type {} has single measurement points'.format(self.sensor_type))
            logger.info(sessions[sessions['start'] == sessions['end']])
            sessions = sessions[sessions['start'] != sessions['end']]

        sessions['end'] = sessions['end'].shift(-1)

        session_times = sessions.dropna().reset_index(drop=True)

        for session_start, session_end in zip(session_times['start'], session_times['end']):
            time_window = ((self.dataframe.index > session_start) & (self.dataframe.index < session_end))
            measurement_sessions.append(SensorMeasurementSession(self.dataframe[time_window], self.sensor_type))

        return measurement_sessions, session_times


class SensorMeasurementSession:
    """A class representing a continuous measurement series for a particular sensor.

    """
    def __init__(self, dataframe, sensor_type):
        self.dataframe = dataframe
        self.sensor_type = sensor_type
        self.start = dataframe.index[0]
        self.end = dataframe.index[-1]
        self.duration = dataframe.index[-1] - dataframe.index[0]

    def to_constant_timestep(self, time_step, timeseries_start=None):
        """Resample dataframe to the given time step. Linearly interpolates between samples.

        :param float time_step: timestep in seconds
        :param datetime.datetime timeseries_start: start constant step time series at specified time
        :return SensorMeasurementSession: sensor session with resampled and interpolated data
        """

        old_time_vector = self.dataframe.index.values.astype(np.int64)
        new_time_vector = pd.date_range(
            start=timeseries_start or self.start,
            end=self.end,
            freq="{:.12f}S".format(time_step)
        )

        new_dataframe = pd.DataFrame(index=new_time_vector)

        for column in self.dataframe.columns:
            signal = interp1d(old_time_vector, self.dataframe[column], assume_sorted=True)
            new_dataframe[column] = signal(new_time_vector.values.astype(np.int64))

        return SensorMeasurementSession(new_dataframe, self.sensor_type)

    def trim_session(self, trim_from_start=dt.timedelta(), trim_from_end=dt.timedelta()):
        """Delete first and last measurements from the session

        :param datetime.timedelta trim_from_start: Amount of time to trim from the start of the session
        :param datetime.timedelta trim_from_end: Amount of time to trim from the end of the session
        :return SensorMeasurementSession: sensor session with trimmed_dataframe
        """
        time_window = (
                (self.dataframe.index > self.start + trim_from_start) &
                (self.dataframe.index < self.end - trim_from_end)
        )

        trimmed_dataframe = self.dataframe[time_window]
        return SensorMeasurementSession(trimmed_dataframe, self.sensor_type)

    def plot(self, sensor_types_metadata, sensor_names=None, plot_start_offset=dt.timedelta(), plot_max_time=None):
        """Plots the session dataframe with plotly.

        :param sensor_types_metadata: Metadata about the sensor type, used for figure layout
        :param sensor_names: Specific sensors to plot
        :param plot_start_offset: start data plot after some time
        :param plot_max_time: limit to the time plotted
        :return: plotly.graph_objs.Figure: a line graph of the sensor data against time
        """

        plot_max_time = plot_max_time or self.duration - plot_start_offset
        plot_df = self.trim_session(plot_start_offset, self.duration - plot_start_offset-plot_max_time).dataframe

        if sensor_names:
            plot_df = plot_df[sensor_names]

        layout = {
            "title": sensor_types_metadata[self.sensor_type]['description'],
            "xaxis_title": "Date/Time",
            "yaxis_title": sensor_types_metadata[self.sensor_type]['variable'],
            "legend_title": "Sensor"
        }
        figure = plot_with_layout(plot_df, layout_dict=layout)
        return figure
