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


def plot_cp_curve(df, sensor_coordinates_reference, u=10, p_inf=1e5, cp_minimum=-10, cp_maximum=3):
    """Plot a Cp curve for a blade based on barometer data and the positions of the barometers.

    :param pandas.DataFrame df: a dataframe of pressure data from any number of barometers for an instant in time
    :param str sensor_coordinates_reference: the reference name of the barometers' coordinates
    :param float u: the free stream fluid velocity in m/s
    :param float p_inf: the freestream pressure in Pa
    :param float cp_minimum: the minimum Cp value to include in the plot
    :param float cp_maximum: the maximum Cp value to include in the plot
    :return plotly.graph_objs.Figure: the Cp plot
    """
    client = BigQuery()
    sensor_type_reference = "barometer"
    raw_barometer_data = df.iloc[:1]
    datetime = raw_barometer_data["datetime"].iloc[0]

    # Drop metadata columns.
    data_column_names = raw_barometer_data.columns[raw_barometer_data.columns.str.startswith("f")].tolist()
    raw_barometer_data = raw_barometer_data[["datetime"] + data_column_names].set_index("datetime")

    # Get sensor names.
    sensor_types_metadata = client.get_sensor_types()
    raw_barometer_data.columns = sensor_types_metadata[sensor_type_reference]["sensors"]

    # Transform raw barometer data to correct units.
    converted_barometer_data = RawSignal(raw_barometer_data, sensor_type_reference)
    converted_barometer_data.measurement_to_variable()

    # Get barometer coordinates.
    barometer_coordinates = client.get_sensor_coordinates(reference=sensor_coordinates_reference)
    barometer_coordinates = pd.DataFrame(
        {
            "x": barometer_coordinates["geometry"]["xy_coordinates"]["x_coordinates"],
            "y": barometer_coordinates["geometry"]["xy_coordinates"]["y_coordinates"],
        }
    )
    barometer_coordinates["sensor"] = barometer_coordinates.index

    # Split x-coordinates into suction side and pressure side of blade.
    x_suction_side = barometer_coordinates[barometer_coordinates["y"] > 0]["x"].sort_values()
    suction_side_barometer_names = converted_barometer_data.dataframe.columns[x_suction_side.index]

    x_pressure_side = barometer_coordinates[barometer_coordinates["y"] < 0]["x"].sort_values()
    pressure_side_barometer_names = converted_barometer_data.dataframe.columns[x_pressure_side.index]

    # Calculate Cp.
    q = 0.5 * 1.225 * u**2
    aerodynamic_pressures = converted_barometer_data.dataframe - p_inf
    cp = aerodynamic_pressures / q

    # Split Cp values into suction side and pressure side of blade.
    suction_cp = cp[suction_side_barometer_names].iloc[0]
    pressure_cp = cp[pressure_side_barometer_names].iloc[0]

    figure = px.scatter(height=800)

    figure.add_scatter(
        x=x_suction_side,
        y=suction_cp[(cp_minimum < suction_cp) & (suction_cp < cp_maximum)],
        marker={"color": "blue"},
        mode="markers",
        name="Suction side",
    )

    figure.add_scatter(
        x=x_pressure_side,
        y=pressure_cp[(cp_minimum < pressure_cp) & (pressure_cp < cp_maximum)],
        marker={"color": "red"},
        mode="markers",
        name="Pressure side",
    )

    figure.update_layout(
        {
            "title_text": f"Cp curve at {datetime}",
            "xaxis": {"title": "Chordwise sensor position [m/1m]"},
            "yaxis": {"title": "Cp", "range": [2, -2], "autorange": True},
        }
    )

    return figure
