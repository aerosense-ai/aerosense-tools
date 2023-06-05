import datetime as dt

import pandas as pd
from plotly import express as px

from aerosense_tools.preprocess import RawSignal
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

    sensor_coordinates_df = pd.DataFrame(
        {
            "x": sensor_coordinates["geometry"]["xy_coordinates"]["x_coordinates"],
            "y": sensor_coordinates["geometry"]["xy_coordinates"]["y_coordinates"],
        }
    )

    sensor_coordinates_df["labels"] = labels or sensor_coordinates_df.index

    figure = px.scatter(sensor_coordinates_df, x="x", y="y", text="labels")
    figure.update_traces(textposition="top center")
    figure.update_layout({"title": f"Sensor coordinates: {reference!r}"})
    return figure


def plot_cp_curve(
    installation_reference,
    node_id,
    sensor_coordinates_reference,
    datetime,
    u=10,
    p_inf=1e5,
    tolerance=1,
    cp_minimum=-10,
    cp_maximum=3,
):
    client = BigQuery()
    sensor_type_reference = "barometer"
    start_datetime = datetime - dt.timedelta(seconds=tolerance / 2)
    finish_datetime = datetime + dt.timedelta(seconds=tolerance / 2)

    barometer_df, _ = client.get_sensor_data(
        installation_reference=installation_reference,
        node_id=node_id,
        sensor_type_reference=sensor_type_reference,
        start=start_datetime,
        finish=finish_datetime,
    )

    data_columns = barometer_df.columns[barometer_df.columns.str.startswith("f")].tolist()
    signal_df = barometer_df[["datetime"] + data_columns].set_index("datetime")

    sensor_types_metadata = client.get_sensor_types()
    signal_df.columns = sensor_types_metadata[sensor_type_reference]["sensors"]

    raw_sensor_data = RawSignal(signal_df, sensor_type_reference)
    raw_sensor_data.measurement_to_variable()

    barometer_coordinates = client.get_sensor_coordinates(reference=sensor_coordinates_reference)

    barometer_coordinates = pd.DataFrame(
        {
            "x": barometer_coordinates["geometry"]["xy_coordinates"]["x_coordinates"],
            "y": barometer_coordinates["geometry"]["xy_coordinates"]["y_coordinates"],
        }
    )

    barometer_coordinates["sensor"] = barometer_coordinates.index

    x_pressure_side = barometer_coordinates[barometer_coordinates["y"] < 0]["x"].sort_values()
    x_suction_side = barometer_coordinates[barometer_coordinates["y"] > 0]["x"].sort_values()

    suction_side_barometers = raw_sensor_data.dataframe.columns[x_suction_side.index]
    pressure_side_barometers = raw_sensor_data.dataframe.columns[x_pressure_side.index][:-1]

    q = 0.5 * 1.225 * u**2
    aerodynamic_pressures = raw_sensor_data.dataframe - p_inf
    cp = aerodynamic_pressures / q

    suction_cp = cp[suction_side_barometers].iloc[0]
    pressure_cp = cp[pressure_side_barometers].iloc[0]

    layout_dict = {
        "title_text": f"Cp curve at {datetime}",
        "xaxis": {"title": "Chordwise sensor position [m/1m]"},
        "yaxis": {"title": "Cp", "range": [2, -2], "autorange": True},
    }

    figure = px.scatter(height=800)

    figure.add_scatter(
        x=x_pressure_side,
        y=pressure_cp[(cp_minimum < pressure_cp) & (pressure_cp < cp_maximum)],
        marker={"color": "red"},
        mode="markers",
        name="Pressure Side Aerosense",
    )

    figure.add_scatter(
        x=x_suction_side,
        y=suction_cp[(cp_minimum < suction_cp) & (suction_cp < cp_maximum)],
        marker={"color": "blue"},
        mode="markers",
        name="Suction Side Aerosense",
    )

    figure.update_layout(layout_dict)
    return figure
