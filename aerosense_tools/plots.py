from plotly import express as px

from aerosense_tools.utils import get_cleaned_sensor_column_names


def plot_connection_statistic(df, connection_statistic_name):
    """Plot a line graph of the given connection statistic from the given dataframe against time. The dataframe must
    include a "datetime" column and a column with the same name as `connection_statistic_name`.

    :param pandas.DataFrame df: a dataframe of connection statistics filtered for the time period to be plotted
    :param str connection_statistic_name: the name of the column in the dataframe representing the connection statistic to be plotted
    :return plotly.graph_objs.Figure: a line graph of the connection statistic against time
    """
    figure = px.line(df, x="datetime", y=connection_statistic_name)
    figure.update_layout(xaxis_title="Date/time", yaxis_title="Raw value")
    return figure


def plot_sensors(df):
    """Plot a line graph of the sensor data from the given dataframe against time. The dataframe must include a
    "datetime" column.

    :param pandas.DataFrame df: a dataframe of sensor data filtered for the time period to be plotted
    :return plotly.graph_objs.Figure: a line graph of the sensor data against time
    """
    original_sensor_names, cleaned_sensor_names = get_cleaned_sensor_column_names(df)

    df.rename(
        columns={
            original_name: cleaned_name
            for original_name, cleaned_name in zip(original_sensor_names, cleaned_sensor_names)
        },
        inplace=True,
    )

    figure = px.line(df, x="datetime", y=cleaned_sensor_names)
    figure.update_layout(xaxis_title="Date/time", yaxis_title="Raw value")
    return figure


def plot_pressure_bar_chart(df, y_minimum, y_maximum):
    """Plot a bar chart of pressures against barometers that measured them for a given instant in time.

    :param pandas.DataFrame df: a dataframe of pressure data from any number of barometers for an instant in time
    :param int|float y_minimum: the minimum range for the y-axis (pressure axis)
    :param int|float y_maximum: the maximum range for the y-axis (pressure axis)
    :return plotly.graph_objs.Figure: a bar chart of pressures against barometer number
    """
    original_sensor_names, cleaned_sensor_names = get_cleaned_sensor_column_names(df)
    df_transposed = df[original_sensor_names].transpose()
    df_transposed["Barometer number"] = cleaned_sensor_names

    if len(df) == 0:
        df_transposed["Raw value"] = 0
    else:
        df_transposed["Raw value"] = df_transposed.iloc[:, 0]

    figure = px.line(df_transposed, x="Barometer number", y="Raw value")
    figure.add_bar(x=df_transposed["Barometer number"], y=df_transposed["Raw value"])
    figure.update_layout(showlegend=False, yaxis_range=[y_minimum, y_maximum])
    return figure
