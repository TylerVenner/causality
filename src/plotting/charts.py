# src/plotting/charts.py

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
import plotly.graph_objects as go
import plotly.figure_factory as ff

import networkx as nx
import graphviz 

def create_scatter_plot(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    title: str
) -> Figure:
    """
    Creates an interactive scatter plot with a regression trendline.
    
    Args:
        df (pd.DataFrame): The data to plot.
        x_col (str): The name of the column for the x-axis.
        y_col (str): The name of the column for the y-axis.
        title (str): The title of the chart.
        
    Returns:
        Figure: A Plotly Figure object.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=title,
        trendline="ols", # Adds an Ordinary Least Squares regression line
        trendline_color_override="red"
    )
    fig.update_layout(title_x=0.5) # Center the title
    return fig


def create_histogram(
    df: pd.DataFrame, 
    col_name: str, 
    title: str, 
    x_label: str
) -> Figure:
    """
    Creates an interactive histogram to show a variable's distribution.
    
    Args:
        df (pd.DataFrame): The data to plot.
        col_name (str): The name of the column to create a histogram from.
        title (str): The title of the chart.
        x_label (str): The label for the x-axis.
        
    Returns:
        Figure: A Plotly Figure object.
    """
    fig = px.histogram(
        df,
        x=col_name,
        title=title,
        labels={col_name: x_label}
    )
    fig.update_layout(title_x=0.5) # Center the title
    return fig

def create_overlaid_density_plot(
    data_series1: pd.Series, 
    data_series2: pd.Series, 
    label1: str, 
    label2: str, 
    title: str
) -> go.Figure:
    """
    Creates an interactive, overlaid density plot for two data series.
    
    Args:
        data_series1 (pd.Series): The first data series (e.g., original distribution).
        data_series2 (pd.Series): The second data series (e.g., post-intervention).
        label1 (str): The name for the first data series.
        label2 (str): The name for the second data series.
        title (str): The title of the chart.
        
    Returns:
        go.Figure: A Plotly Figure object.
    """
    fig = ff.create_distplot(
        [data_series1.dropna(), data_series2.dropna()],
        [label1, label2],
        show_hist=False,
        show_rug=False,
        colors=['#1f77b4', '#ff7f0e'] 
    )
    
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        legend=dict(x=0.05, y=0.95),
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgrey'),
        yaxis=dict(gridcolor='lightgrey')
    )
    
    return fig

def create_comparison_density_plot(
    data_before: pd.Series, 
    data_after: pd.Series, 
    label_before: str, 
    label_after: str, 
    title: str,
    color: str
) -> go.Figure:
    """
    Creates an overlaid density plot to compare a distribution before and after an event.
    
    Args:
        data_before (pd.Series): The original data series.
        data_after (pd.Series): The data series after a change.
        label_before (str): Legend label for the original data.
        label_after (str): Legend label for the new data.
        title (str): The title of the chart.
        color (str): The hex or named color to use for both plots.
        
    Returns:
        go.Figure: A Plotly Figure object.
    """
    # Create the distplot with the same color for both traces
    fig = ff.create_distplot(
        [data_before.dropna(), data_after.dropna()],
        [label_before, label_after],
        show_hist=False,
        show_rug=False,
        colors=[color, color]
    )
    
    # Manually set the line styles: dashed for 'before', solid for 'after'
    fig.data[0].line.dash = 'dash'
    fig.data[1].line.dash = 'solid'
    
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        legend=dict(x=0.05, y=0.95),
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgrey'),
        yaxis=dict(gridcolor='lightgrey')
    )
    
    return fig

def create_colored_scatter_plot(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    color_col: str,
    title: str
) -> Figure:
    """
    Creates an interactive scatter plot where points are colored by a discrete category.
    """
    # Create a copy to avoid modifying the original DataFrame
    plot_df = df.copy()
    
    plot_df[color_col] = plot_df[color_col].astype(str)
    
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        color_discrete_map={'0': '#1f77b4', '1': '#d62728'}, 
        labels={color_col: 'Holiday Season'}
    )
    fig.update_layout(title_x=0.5) 
    # Update legend names
    fig.for_each_trace(lambda t: t.update(name = {'0': 'No', '1': 'Yes'}[t.name]))
    return fig

def graphviz_from_nx(graph: nx.Graph, title: str) -> graphviz.Digraph:
    """
    Converts a networkx graph (Graph or DiGraph) into a Graphviz Digraph
    object for rendering in Streamlit.
    
    Handles directed and undirected edges.
    """
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='LR') # Left-to-right layout
    dot.attr('node', shape='circle', style='filled', fillcolor='lightblue')
    dot.attr(label=title, fontsize='16')

    for node in graph.nodes():
        dot.node(str(node))

    if isinstance(graph, nx.DiGraph):
        # Handle PDAGs (nx.DiGraph where i-j means j->i and i->j)
        for u, v in graph.edges():
            if graph.has_edge(v, u):
                # It's an undirected edge
                if u < v: # Only draw once
                    dot.edge(str(u), str(v), dir='none', color='gray')
            else:
                # It's a directed edge
                dot.edge(str(u), str(v), dir='forward', color='black')
    else:
        # Handle simple undirected graphs (like the skeleton)
        for u, v in graph.edges():
            dot.edge(str(u), str(v), dir='none', color='gray')
            
    return dot