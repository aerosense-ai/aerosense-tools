import re


def get_cleaned_sensor_column_names(dataframe):
    """Get cleaned sensor column names for a dataframe when the columns are named like "f0_", "f1_"... "fn_" for `n`
    sensors.

    :param pandas.DataFrame dataframe: a dataframe containing columns of sensor data named like "f0_", "f1_"...
    :return (list, list): the uncleaned and cleaned sensor names
    """
    original_names = [column for column in dataframe.columns if column.startswith("f") and column.endswith("_")]
    cleaned_names = [re.findall(r"f(\d+)_", sensor_name)[0] for sensor_name in original_names]
    return original_names, cleaned_names


def remove_metadata_columns_and_set_datetime_index(dataframe):
    """Remove non-sensor-data columns from the dataframe and set its index to its datetime column.

    :param pandas.DataFrame dataframe: a dataframe containing columns of sensor data named like "f0_", "f1_"... as well as metadata columns
    :return pandas.DataFrame: a dataframe containing only datetime and sensor data columns
    """
    data_columns = dataframe.columns[dataframe.columns.str.startswith("f")].tolist()
    return dataframe[["datetime"] + data_columns].set_index("datetime")
