import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from aerosense_tools.plots import plot_raw_signal


class RawSignal:

    def __init__(self, dataframe, sensor_type):
        self.dataframe = dataframe
        self.sensor_type = sensor_type

    def plot(self, sensor_types_metadata):
        layout = {
            "title": sensor_types_metadata[self.sensor_type]['description'],
            "xaxis_title": "Date/Time",
            "yaxis_title": sensor_types_metadata[self.sensor_type]['variable'],
            "legend_title": "Sensor"
        }
        figure = plot_raw_signal(self.dataframe, layout_dict=layout)
        return figure

    def pad_gaps(self, threshold):
        """Checks for missing data. If the gap between samples (timedelta) is
         higher than the given threshold, then the last sample before the gap
        start is replaced with NaN. Thus, no interpolation will be performed
        during the non-sampling time window.

        :param threshold: maximum gap between two samples as a timedelta type
        """

        self.dataframe[self.dataframe.index.to_series().diff() > threshold] = np.NaN

    def to_constant_timestep(self, time_step):
        """Resample dataframe to the given time step. Linearly interpolates between samples.

        :param float time_step: timestep in seconds
        :return: resampled and interpolated data
        """
        old_time_vector = self.dataframe.index.values.astype(np.int64)
        new_time_vector = pd.date_range(
            start=self.dataframe.index[0],
            end=self.dataframe.index[-1],
            freq="{:.12f}S".format(time_step)
        )

        new_dataframe = pd.DataFrame(index=new_time_vector)

        for column in self.dataframe.columns:
            signal = interp1d(old_time_vector, self.dataframe[column], assume_sorted=True)
            new_dataframe[column] = signal(new_time_vector.values.astype(np.int64))

        self.dataframe = new_dataframe

    def filter_outliers(self, window, std_multiplier):
        """A very primitive filter. Removes data points outside the confidence interval using a rolling median and
        standard deviation.

        :param int window: window (number of samples) for rolling median and standard deviation
        :param float std_multiplier: multiplier to the rolling standard deviation
        """
        rolling_median = self.dataframe.rolling(window).median()
        rolling_std = self.dataframe.rolataframeling(window).std()
        # TODO define filtering rule using rolling df here
        self.dataframe = self.dataframe[
            (self.dataframe <= rolling_median + std_multiplier * rolling_std)
            & (self.dataframe >= rolling_median - std_multiplier * rolling_std)
            ]

    def measurement_to_variable(self):
        """Transform fixed point values to a physical variable."""
        # TODO These values should be picked up from the session configuration metadata
        diffrange = 2 * 6000

        if self.sensor_type == "barometer":
            self.dataframe /= 40.96  # [Pa]
        if self.sensor_type == "barometer_thermometer":
            self.dataframe /= 100  # [Celsius]
        if self.sensor_type == "differential_barometer":
            self.dataframe -= 32767
            self.dataframe *= diffrange/(58982-6553)



