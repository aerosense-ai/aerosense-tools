import datetime
import re


TIME_RANGE_OPTIONS = {
    "Last minute": datetime.timedelta(minutes=1),
    "Last hour": datetime.timedelta(hours=1),
    "Last day": datetime.timedelta(days=1),
    "Last week": datetime.timedelta(weeks=1),
    "Last month": datetime.timedelta(days=31),
    "Last year": datetime.timedelta(days=365),
}


def generate_time_range(time_range, custom_start_date=None, custom_end_date=None):
    """Generate a convenient time range to plot. The options are:
    - Last minute
    - Last hour
    - Last day
    - Last week
    - Last month
    - Last year
    - All time
    - Custom

    :param str time_range:
    :param datetime.date|None custom_start_date:
    :param datetime.date|None custom_end_date:
    :return (datetime.datetime, datetime.datetime, bool): the start and finish datetimes
    """
    if time_range == "All time":
        return datetime.datetime.min, datetime.datetime.now()

    if time_range == "Custom":
        return custom_start_date, custom_end_date

    finish = datetime.datetime.now()
    start = finish - TIME_RANGE_OPTIONS[time_range]
    return start, finish


def get_cleaned_sensor_column_names(dataframe):
    """Get cleaned sensor column names for a dataframe when the columns are named like "f0_", "f1_"... "fn_" for `n`
    sensors.

    :param pandas.DataFrame dataframe: a dataframe containing columns of sensor data named like "f0_", "f1_"...
    :return (list, list): the uncleaned and cleaned sensor names
    """
    original_names = [column for column in dataframe.columns if column.startswith("f") and column.endswith("_")]
    cleaned_names = [re.findall(r"f(\d+)_", sensor_name)[0] for sensor_name in original_names]
    return original_names, cleaned_names
