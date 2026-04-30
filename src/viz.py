from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def fig_route_bar(route_df: pd.DataFrame, *, route_col: str, score_col: str = "RouteEfficiencyScore", top_n: int = 20) -> go.Figure:
    d = route_df.head(int(top_n)).copy()
    fig = px.bar(
        d,
        x=score_col,
        y=route_col,
        orientation="h",
        color=score_col,
        color_continuous_scale="Viridis",
        hover_data=["Volume", "AvgLeadTime", "StdLeadTime", "DelayRate"],
        title="Top Routes by Efficiency Score",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_destination_heatmap(dest_df: pd.DataFrame, *, destination_col: str, value_col: str = "AvgLeadTime") -> go.Figure:
    d = dest_df.copy()
    fig = px.density_heatmap(
        d,
        x=destination_col,
        y=value_col,
        nbinsx=min(50, max(5, len(d))),
        title=f"Destination Heatmap ({value_col})",
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_ship_mode_compare(mode_df: pd.DataFrame, *, ship_mode_col: str) -> go.Figure:
    fig = px.bar(
        mode_df,
        x=ship_mode_col,
        y="AvgLeadTime",
        color="DelayRate",
        hover_data=["Volume", "StdLeadTime"],
        title="Ship Mode Comparison",
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_monthly_trend(trend_df: pd.DataFrame, *, month_col: str = "_OrderMonth") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df[month_col], y=trend_df["AvgLeadTime"], mode="lines+markers", name="Avg Lead Time"))
    fig.add_trace(go.Scatter(x=trend_df[month_col], y=trend_df["DelayRate"], mode="lines+markers", name="Delay Rate", yaxis="y2"))
    fig.update_layout(
        title="Monthly Trend: Lead Time & Delay Rate",
        xaxis_title="Month",
        yaxis=dict(title="Avg Lead Time (days)"),
        yaxis2=dict(title="Delay Rate", overlaying="y", side="right", tickformat=".0%"),
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h"),
    )
    return fig

