import pandas as pd
from plotly import express as px

from aerosense_tools.queries import BigQuery
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


def plot_sensors(df, line_descriptions=None):
    """Plot a line graph of the sensor data from the given dataframe against time. One line is plotted per sensor in the
    dataframe. The dataframe must include a "datetime" column.

    :param pandas.DataFrame df: a dataframe of sensor data filtered for the time period to be plotted
    :param list|str|None line_descriptions: the descriptions to give each sensor in the plot. If this is a list, there must be the same number of elements as sensors. If this is a string, it will be applied to all sensors.
    :return plotly.graph_objs.Figure: a line graph of the sensor data against time
    """
    legend_title = "Legend"
    y_axis_title = "Raw value"
    original_sensor_names, cleaned_sensor_names = get_cleaned_sensor_column_names(df)

    df.rename(
        columns={
            original_name: cleaned_name
            for original_name, cleaned_name in zip(original_sensor_names, cleaned_sensor_names)
        },
        inplace=True,
    )

    if line_descriptions:
        y_axis_title = ""

        if isinstance(line_descriptions, str):
            legend_title = line_descriptions
            figure = px.line(df, x="datetime", y=cleaned_sensor_names)
        else:
            df.rename(
                columns={
                    cleaned_name: column_name
                    for cleaned_name, column_name in zip(cleaned_sensor_names, line_descriptions)
                },
                inplace=True,
            )

            figure = px.line(df, x="datetime", y=line_descriptions)

    else:
        figure = px.line(df, x="datetime", y=cleaned_sensor_names)

    figure.update_layout(xaxis_title="Date/time", yaxis_title=y_axis_title, legend_title=legend_title)
    return figure


def plot_with_layout(df, layout_dict):
    """Plot a line graph of each column of dataframe with index for x-axis. The layout is updated with data from a dict.

    :param pandas.DataFrame df: a dataframe of sensor data filtered for the time period to be plotted
    :param dict layout_dict: layout dictionary, to name the plot, axis and legend
    :return: plotly.graph_objs.Figure: a line graph of the sensor data against time
    """
    figure = px.line(df)
    figure.update_layout(layout_dict)
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


def plot_sensor_coordinates(reference, labels=None):
    """Plot the sensor coordinates for the given sensor coordinates reference.

    :param str reference: the reference of the sensor coordinates to plot
    :param list(str)|None labels: labels to give to each sensor in the plot (should be as long as the number of sensors)
    :return plotly.graph_objs.Figure: a plot of sensor coordinates
    """
    sensor_coordinates = BigQuery().get_sensor_coordinates(reference=reference)

    sensor_coordinates_df_dict = {
        "x": sensor_coordinates["geometry"]["xy_coordinates"]["x_coordinates"],
        "y": sensor_coordinates["geometry"]["xy_coordinates"]["y_coordinates"],
    }

    if labels:
        sensor_coordinates_df_dict["labels"] = labels

    sensor_coordinates_df = pd.DataFrame(sensor_coordinates_df_dict)

    figure = px.scatter(sensor_coordinates_df, x="x", y="y", text="labels")
    figure.update_traces(textposition="top center")
    figure.update_layout({"title": f"Sensor coordinates: {reference!r}"})
    return figure
