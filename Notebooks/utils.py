from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.preprocessing import RobustScaler

import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio



NUMERIC_FONT_FAMILY = "Consolas"
TEXT_FONT_FAMILY = "Times New Roman"

pio.templates["custom_dark"] = pio.templates["plotly_dark"]

pio.templates["custom_dark"].layout.update(
    font=dict(
        family=TEXT_FONT_FAMILY
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        tickfont=dict(family=NUMERIC_FONT_FAMILY)
    ),
    yaxis=dict(
        tickfont=dict(family=NUMERIC_FONT_FAMILY)
    ),
    coloraxis=dict(
        colorbar=dict(
            tickfont=dict(family=NUMERIC_FONT_FAMILY)
        )
    )
)

pio.templates.default = "custom_dark"




#Define global variables
COLORMAP_VALUES = 'GnBu_r'
COLORMAP_GAPS = 'Burgyl'

FIG_WIDTH = 1600
FIG_HEIGHT = 20*30 + 300
FAMILY_FONT = "Times New Roman"


# Function to calculate the number of consecutive NaN values for each variable and bin them into specified categories
def get_data_gaps(data, bins = None, labels = None, 
                  no_gap_value = 0, format_gaps = None,
                  datetime_column = 'datetime', variable_column = 'variable',
                  aux_columns = []):
    
    # Define the bins and corresponding labels
    if bins is None:
        bins = [0, 1, 3, 12, 24, 24*3, 24*7, 24*30, 24*30*3, 24*30*6, np.inf]
    if labels is None:
        labels = np.arange(1,len(bins))
        if format_gaps is None:
            format_gaps = int
        

    # Create a copy of the dataframe
    df_consecutive_nans = data.copy()

    # Group by the variable_column column and sort by 'date'
    df_consecutive_nans.sort_values(by=datetime_column, inplace=True)
    grouped = df_consecutive_nans.groupby(variable_column)

    # Iterate over each group to calculate consecutive NaNs
    for name, group in grouped:
        for col in group.columns:
            if col not in [variable_column, datetime_column] + aux_columns:
                # Identify NaN values
                is_nan = group[col].isna()
                
                # Group consecutive NaNs and calculate their count
                nan_group = (is_nan != is_nan.shift()).cumsum()
                df_consecutive_nans.loc[group.index, col] = is_nan.groupby(nan_group).transform('sum')
                
                # Set non-NaN values to 0
                df_consecutive_nans.loc[group.index[~is_nan], col] = 0

    # Convert all columns except 'year' and 'month' to integers
    columns_to_convert = [col for col in df_consecutive_nans.
                          columns if col not in [variable_column, datetime_column] + aux_columns]
    
    df_consecutive_nans[columns_to_convert] = df_consecutive_nans[columns_to_convert].astype(int)

    df_temp_gaps = df_consecutive_nans.copy()

    for column in df_temp_gaps.columns:
        if column not in [variable_column, datetime_column] + aux_columns:
            # Replace 0 with NaN for the purpose of binning
            df_temp_gaps[column] = np.digitize(df_temp_gaps[column], bins, right=True)
            df_temp_gaps[column] = df_temp_gaps[column].map(lambda x: 
                labels[x-1] if x > 0 else no_gap_value)
    
    if format_gaps is not None:
        df_temp_gaps[columns_to_convert] = df_temp_gaps[columns_to_convert].astype(format_gaps)
            
    return df_temp_gaps.sort_index()














def plot_heatmap(df,
                 df_gaps = None, 
                 stations = None,
                 query = None,
                 color_values_mode = "minmax", 
                 plot_title = None,
                 sort_cols = True,
                 drop_empty_cols = False,
                 timestamp_mode = "days", timestamp_frequency = [1, 15],
                 time_window_width = 31,
                 colormap_values = COLORMAP_VALUES,
                 colormap_gaps = COLORMAP_GAPS):

    if query:
        df = df.query(query).copy()
        if df_gaps is not None:
            df_gaps = df_gaps.query(query).copy()
    
    if stations is not None:
        df = df[stations].copy()
        if df_gaps is not None:
            df_gaps = df_gaps[stations].copy()
    
    if sort_cols:
        if stations is None:
            print("Warning: sort_cols is True but stations is None. Columns will not be sorted.")
        else:
            # Sort columns by their mean values (descending)
            stations = df.isna().sum()[stations].sort_values(ascending=True).index.tolist()
            df = df[stations].copy()
            if df_gaps is not None:
                df_gaps = df_gaps[stations].copy()
    
    if drop_empty_cols:
        # Delete columns with all NaN values
        df = df.dropna(axis=1, how='all')
        if df_gaps is not None:
            df_gaps = df_gaps[df.columns].copy()
        
    
    array = df.values
    array_gaps = df_gaps.values if df_gaps is not None else None
    
    # Dependind on the color_values_mode, 
    # apply different transformations to the array for the values plot
    if color_values_mode == "minmax":
        array_color = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
        
    elif color_values_mode == "robust":
        
        scaler = RobustScaler()
        df_color = df.copy()
        df_color = scaler.fit_transform(df_color)
        array_color = df_color.values
    else:
        array_color = array.copy()
        
        
        
    # -----------------------------------------------------------------------------------------
    # General figure setup


    # Create empty figure with y inverted for better visualization
    fig = go.Figure()
    fig.update_yaxes(autorange="reversed")



    if timestamp_mode == "hours":
        # --- Timestamps each X hours
        # Add x axis labels at every hour in timestamp_frequency
        hours = timestamp_frequency
        tickindex = df.index.hour.isin(hours) & (df.index.minute == 0)
        tickvals = df.index[tickindex]
        ticktext = df.index[tickindex].strftime('%Y-%m-%d %H')
    
    elif timestamp_mode == "days":
        # --- Timmestamps each X day
        # Add x axis labels at every day in timestamp_frequency
        days = timestamp_frequency
        tickindex=df.index.day.isin(days)\
            & (df.index.hour == 0)
        tickvals=df.index[tickindex]
        ticktext=df.index[tickindex].strftime('%Y-%m-%d')

    fig.update_xaxes(tickvals= tickvals, ticktext= ticktext)


    # Small window of time_window_width days from the start of the data for better visualization

    start = df.index.min().strftime('%Y-%m-%d')
    end = df.index.min() + pd.Timedelta(days=time_window_width)
    fig.update_layout(xaxis_range=[start, end])

    # Adjust scale and labels
    fig.update_layout(width=FIG_WIDTH, 
                    height=FIG_HEIGHT)

    # Update axis labels
    fig.update_xaxes(title_text='← Time →',
                    title_font=dict(size=20),
                    tickfont=dict(size=16),
                    tickangle=30
                    )
    fig.update_yaxes(title_text='← Stations →',
                    title_font=dict(size=20),
                    tickfont=dict(size=16)
                    )

    # Ad month annotations 
    month_start = pd.date_range(start=start, periods=12, freq="MS")  

    # Add a vertical line end of month
    for inicio, middle in zip(month_start, 
                            month_start + pd.Timedelta(days=14)):
        fig.add_vline(x=inicio, 
                    line_width=2, 
                    line_dash="dashdot", 
                    line_color="red")
        

        fig.add_annotation(
        x=middle,
        yref="paper", y=1.05,  # posición relativa al canvas
        text=middle.strftime('%B').capitalize(),
        showarrow=False,
        font=dict(size=20, color="gray"),
        xanchor="center"
        )


    # Change background to transparent (both paper and plot, change as needed)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)')





    # -----------------------------------------------------------------------------------------
    # Plot values

    values_fig = px.imshow(array_color.T, 
                    aspect='auto', 
                    title=f'Visualización {plot_title}' if plot_title is not None else None,
                    x=df.index,
                    y=df.columns,
                    )

    # Add hovertemplate to include array_color information
    values_fig.update_traces(hovertemplate='''
                            Date: %{x}<br>
                            Station: %{y}<br>
                            Value: %{customdata}''')

    # Add customdata to the figure
    values_fig.data[0].update(customdata=np.where(np.isnan(array.T), 
                                                'Gap ' + array_gaps.T.astype(int).astype(str), 
                                                array.T.astype(str)))

    # First trace use color axis (coloraxis1)
    values_fig.update_traces(
        showscale=True,
        coloraxis='coloraxis1'
    )





    # -----------------------------------------------------------------------------------------
    # Plot gaps

    if df_gaps is not None:
        gaps_fig = px.imshow(
            array_gaps.T,
            x=df_gaps.index,
            y=df_gaps.columns,
        )

        # Add hovertemplate to include array_color information
        gaps_fig.update_traces(hovertemplate='''
                                Date: %{x}<br>
                                Station: %{y}<br>
                                Value: Gap %{z}''')


        # Second trace use different color axis (coloraxis2)
        gaps_fig.update_traces(
            showscale=True,
            coloraxis='coloraxis2'
        )



    # -----------------------------------------------------------------------------------------
    # Add colorbars

    colorbars_font = dict(
        size=20,
        color="gray",
    )

    colorbars_tickfont = dict(
        size=17,
        color="gray",
    )

    fig.update_layout(
        coloraxis1=dict(
            colorscale=colormap_values,
            colorbar=dict(
                title=dict(
                    text="Values",
                    font=colorbars_font
                ),
                x=0.5, y=-0.3, yref="paper", orientation="h",
                tickvals=np.linspace(0, 1, 11),
                ticktext=np.linspace(
                    df.min().min(),
                    df.max().max(),
                    11
                ).round(2).astype(str).tolist(),
                tickfont=colorbars_tickfont
            )
        )
    )

    # Colorbar for gaps
    if df_gaps is not None:
        fig.update_layout(
            coloraxis2=dict(
                colorscale=colormap_gaps,
                colorbar=dict(
                    title=dict(
                        text="_Gaps_",
                        font=colorbars_font
                    ),
                    x=0.5, y=-0.41, yref="paper", orientation="h",
                    tickfont=colorbars_tickfont
                )
            )
        )
    

    # Add plots to the main figure
    
    if df_gaps is not None:
        for trace in gaps_fig.data:
            fig.add_trace(trace)
        
    for trace in values_fig.data:
        fig.add_trace(trace)



    # Add visibility menu
    
    if df_gaps is not None:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=-0.1,
                    y=1,
                    buttons=[
                        dict(label="Values & Gaps", method="restyle", args=[{"visible": [True, True]}]),
                        dict(label="Values only", method="restyle", args=[{"visible": [False, True]}]),
                        dict(label="Gaps only", method="restyle", args=[{"visible": [True, False]}])
                    ]
                )
            ]
        )


    return fig





def plot_timeseries_var_station(df,
                                query = None,
                                stations=None,
                                var_name = '',
                                drop_empty_cols = False,
                                timestamp_mode = "days",
                                timestamp_frequency = [1, 15],
                                time_window_width = 31):
    
    if query is not None:
        df = df.query(query).copy()
        
    if stations is not None:
        df = df[stations].copy()
        
    if drop_empty_cols:
        # Delete columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        
    fig = go.Figure()

    # Add a line for each station
    for station in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[station],
            mode='lines',
            name=station
        ))

    # Update general layout
    fig.update_layout(
        title=f'{var_name.upper()} Time series',
        title_font=dict(size=24),
        xaxis_title='Date',
        yaxis_title=f'{var_name.upper()} Value',
        height=800,
        width=1600
    )

 
 
    if timestamp_mode == "hours":
        # --- Timestamps each X hours
        # Add x axis labels at every hour in timestamp_frequency
        hours = timestamp_frequency
        tickindex = df.index.hour.isin(hours) & (df.index.minute == 0)
        tickvals = df.index[tickindex]
        ticktext = df.index[tickindex].strftime('%Y-%m-%d %H')
    
    elif timestamp_mode == "days":
        # --- Timmestamps each X day
        # Add x axis labels at every day in timestamp_frequency
        days = timestamp_frequency
        tickindex=df.index.day.isin(days)\
            & (df.index.hour == 0)
        tickvals=df.index[tickindex]
        ticktext=df.index[tickindex].strftime('%Y-%m-%d')


    fig.update_xaxes(tickvals= tickvals, ticktext= ticktext)

    # Small window of time_window_width days from the start of the data for better visualization

    start = df.index.min().strftime('%Y-%m-%d')
    end = df.index.min() + pd.Timedelta(days=time_window_width)
    fig.update_layout(xaxis_range=[start, end])


    # Ad month annotations 
    month_start = pd.date_range(start=start, periods=12, freq="MS")  

    # Add a vertical line end of month
    for inicio, middle in zip(month_start, 
                            month_start + pd.Timedelta(days=14)):
        fig.add_vline(x=inicio, 
                    line_width=2, 
                    line_dash="dashdot", 
                    line_color="red")
        
        fig.add_annotation(
        x=middle,
        yref="paper", y=1.05,  # posición relativa al canvas
        text=middle.strftime('%B').capitalize(),
        showarrow=False,
        font=dict(size=20, color="gray"),
        xanchor="center"
        )

    # Change background to transparent (both paper and plot, change as needed)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)')
    
    
    # Add dashed grid lines with alpha 0.5
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray', griddash='dash')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray', griddash='dash')

    # Update axis labels
    fig.update_xaxes(title_font=dict(size=20 ),
                    tickfont=dict(size=16),
                    tickangle=30
                    )
    fig.update_yaxes(title_font=dict(size=20),
                    tickfont=dict(size=16)
                    )

    # Update legend 
    fig.update_layout(
        legend=dict(
            font=dict(size=16)
        )
    )


    return fig