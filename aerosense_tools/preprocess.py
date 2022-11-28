import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class RawSignal:
    def __init__(self, dataframe, sensor):
        self.dataframe = dataframe
        self.sensor = sensor

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
        if self.sensor == "barometer":
            self.dataframe /= 40.96  # [Pa]
        if self.sensor == "barometer_thermometer":
            self.dataframe /= 100  # [Celsius]


