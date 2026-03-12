from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.preprocessing import RobustScaler

import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio

from plotly.subplots import make_subplots


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


def plot_gap_analysis(df,
                      stations = None,
                      variable_column = 'variable',
                      bar_mode = 'stack',
                      subplot_pan_columns = 4,
                      map_gap = None):

    # variable_column = 'variable'
    # bar_mode = 'stack'
    # subplot_pan_columns = 4
    # Count the number of elements equal to 0 for each column
    
    if stations is not None:
        df = df[[variable_column] + stations].copy()
    else:
        stations = df.columns.tolist()
        stations.remove(variable_column)
        
        
    
    variables = df[variable_column].unique().tolist()
    zero_counts = df.isna().sum()

    # Sort specified_order and remaining_columns by the number of zeros in descending order
    specified_order = sorted(stations, key=lambda col: zero_counts[col], reverse=True)
    
    df = df[[variable_column] + specified_order].replace({np.nan: 0})

    # Group by 'variable' and calculate the percentage of each value in each station
    variable_groups = df.groupby(variable_column)

    # Create a dictionary to store the percentage data for each variable
    percentage_data = {}

    for variable, group in variable_groups:
        
        group.sort_index(inplace=True)
        # For each column identify with nan all values before the first 0
        for col in group.columns:
            temp = (group[col].copy()==0).cumsum() > 0
            group[col] = group[col].where(temp, np.nan)
        
        
        # Calculate the percentage for each station
        percentage_data[variable] = group.drop(columns=variable_column).apply(
            lambda x: x.value_counts(normalize=True) * 100
        ).fillna(0)
    
    all_values = sorted(set(val for d in percentage_data.values() for val in d.index))
    color_scale = px.colors.qualitative.Bold  # You can also use px.colors.sequential.OrRd, etc.

    # If there are more values than colors in the scheme, they repeat
    color_map = {val: color_scale[i % len(color_scale)] for i, val in enumerate(all_values)}

    fig = make_subplots(
        rows=len(variables), 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        shared_yaxes=True,
    )

    # Adjust the position of subplot titles
    for i, annotation in enumerate(fig['layout']['annotations']):
        annotation['y'] -= 0.005

    # Iterate over variables and add traces to the subplot
    for i, variable in enumerate(variables, start=1):
        if variable in percentage_data:
            data_var = percentage_data[variable].reset_index().melt(id_vars='index', var_name='Station', value_name='Percentage')
            data_var.rename(columns={'index': 'Value'}, inplace=True)
            
            for value in data_var['Value'].unique():
                filtered_data = data_var[data_var['Value'] == value]
                
                if map_gap is not None:
                    legend_name = map_gap.get(str(int(value)), 'Nan')
                else:
                    legend_name = str(int(value))

                # Add bar trace for each value, reference to the same legend group
                fig.add_trace(
                    go.Bar(
                        x=filtered_data['Station'],
                        y=filtered_data['Percentage'],
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=(i == 1),              
                        marker_color=color_map[value],
                        textposition='outside'
                    ),
                    row=i, col=1
                )

    # Update layout for all subplots
    fig.update_yaxes(
        side='left',
        range=[0, 112],
        ticksuffix='%',
        tickvals=[0, 20, 40, 60, 80, 100],
        showgrid=True,
    )

    # Set y-axis titles for each subplot (variable)
    for i, variable in enumerate(variables, start=1):
        fig.update_yaxes(
            title_text=variable, 
            title_font=dict(size=18),
            tickfont=dict(size=14),
            row=i, col=1
        )
        
    # Update axis font
    fig.update_xaxes(tickfont=dict(size=14))

    # Update layout to show legend on the top of the plot
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=18)
        )
    )

    # Update overall figure layout
    fig.update_layout(
        height=150 * len(variables),  # Adjust height based on the number of variables
        width=1000,
        barmode=bar_mode
    )



    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray', griddash='solid')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    if bar_mode == 'group':
        fig.update_xaxes(
            range=[-0.5, subplot_pan_columns - 0.5],
        )

    return fig




def plot_spatial_gap(df, 
                     variable_column = 'variable', 
                     frequency = 'month',
                     agg_method = 'mean'):
    
    df_spatial_gaps = df.reset_index(drop=False).copy()
    time_columns = ['datetime', 'year', 'month', 'day']
    
    for col in time_columns:
        if col not in df_spatial_gaps.columns:
            print (f'Column {col} must be in dataset')
            return
    
    
    
    df_spatial_gaps = df.reset_index().rename(columns={variable_column:'var'}).melt(id_vars = ['datetime', 'year', 'month', 'day', 'var'])
    df_spatial_gaps = df_spatial_gaps.groupby(['year','month','day','datetime','var']).agg({'value':'count'}).reset_index()
    # Summary resampling


    if frequency == 'year':
        sampling = ['year']
        time_format = '%Y'
    elif frequency == 'month':
        sampling = ['year', 'month']
        time_format = '%Y-%m'
    elif frequency == 'day':
        sampling = ['year', 'month', 'day']
        time_format = '%Y-%m-%d'

    # Aggregate
    df_spatial_gaps = df_spatial_gaps.groupby(sampling +['var']).agg({'value':agg_method}).reset_index()
    df_spatial_gaps['time_res'] = df_spatial_gaps.apply(lambda x: '-'.join([str(x[r]).zfill(2) for r in sampling]), axis=1)
    df_spatial_gaps = df_spatial_gaps.pivot_table(values='value', index = 'time_res', columns='var')

    df_spatial_gaps.index = pd.to_datetime(df_spatial_gaps.index, format=time_format)
    
    

    fig = go.Figure()

    # Add a line for each variable
    for var_name in df_spatial_gaps.columns:
        fig.add_trace(go.Scatter(
            x=df_spatial_gaps.index,
            y=df_spatial_gaps[var_name],
            mode='lines',
            name=var_name
        ))

    # Update layout
    fig.update_layout(
        title=f'Network coverage',
        xaxis_title='Date',
        yaxis_title=f'# Stations',
        height=600,
        width=1600
    )


    # --- Timmestamps 
    # Add x axis labels at every month
    if frequency == 'month':
        tickindex=df_spatial_gaps.index.month.isin([1])
        tickvals=df_spatial_gaps.index[tickindex]
        ticktext=df_spatial_gaps.index[tickindex].strftime('%Y-%m-%d')

    elif frequency == 'year':
        tickindex=df_spatial_gaps.index.year.isin([1])
        tickvals=df_spatial_gaps.index[tickindex]
        ticktext=df_spatial_gaps.index[tickindex].strftime('%Y')

    fig.update_xaxes(tickvals= tickvals, ticktext= ticktext)

    start = df.index.min().strftime(time_format)
    end = df.index.max().strftime(time_format)

    fig.update_layout(xaxis_range=[start, end])

        

    # Add dashed grid lines with alpha 0.5
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray', griddash='dash')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray', griddash='dash')


    # fig.update_layout(template='plotly_dark')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.show()



