"""The Plotly charts of the app. Each one answers a single question and labels
itself (reference lines and bands carry a text label on the chart), so it can
be read without a legend hunt. The reading guide of each chart lives in
core/guides.py and is shown next to it with ui.chart_header().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme
from .i18n import lang, num, t
from .theme import AXIS, BEAT_STYLE, CRITICAL, DOUBT_BAND, INK, INK_2, MODEL_COLORS, MUTED


def _finish(fig: go.Figure, height: int, legend: bool = False, **kw) -> go.Figure:
    fig.update_layout(template=theme.TEMPLATE, height=height, showlegend=legend, plot_bgcolor=theme.SURFACE,
                      paper_bgcolor=theme.SURFACE, separators=",." if lang() == "es" else ".,", **kw)
    return fig


def _label_line(fig: go.Figure, x: float, text: str, color: str = MUTED, dash: str = "dot") -> None:
    fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1.5,
                  annotation_text=text, annotation_position="top", annotation_font=dict(size=11, color=color))


def opinions(rows: list[tuple[str, str, float]]) -> go.Figure:
    """Q: what does each model say, and what is the final answer?

    Horizontal bars (key, label, probability); the last row is the final answer.
    The 50 % line is the decision threshold.
    """
    labels = [r[1] for r in rows][::-1]
    values = [r[2] for r in rows][::-1]
    colors = [MODEL_COLORS.get(r[0], INK) for r in rows][::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
                           texttemplate="%{x:.0%}", textposition="outside", cliponaxis=False,
                           textfont=dict(color=INK, size=13), hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    _label_line(fig, 0.5, t("chart.threshold"))
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", dtick=0.25)
    fig.update_yaxes(tickfont=dict(color=INK, size=13))
    return _finish(fig, 80 + 44 * len(rows), margin=dict(l=8, r=16, t=28, b=8))


def sweep_line(df: pd.DataFrame, key: str, current: float, x_title: str) -> go.Figure:
    """Q: how would the risk change if only this value changed?"""
    fig = go.Figure(go.Scatter(x=df[key], y=df["probability"], mode="lines", line=dict(color=INK, width=2.5),
                               hovertemplate="%{x:.0f} mmHg → %{y:.0%}<extra></extra>"))
    fig.add_hrect(y0=0.35, y1=0.65, fillcolor=DOUBT_BAND, opacity=0.7, line_width=0, layer="below",
                  annotation_text=t("chart.doubt_zone"), annotation_position="top left",
                  annotation_font=dict(size=11, color="#92400E"))
    here = float(np.interp(current, df[key], df["probability"]))
    fig.add_scatter(x=[current], y=[here], mode="markers+text", marker=dict(size=12, color=CRITICAL,
                    line=dict(width=2, color="white")), text=[t("chart.your_patient")], textposition="top center",
                    textfont=dict(color=CRITICAL, size=12), hoverinfo="skip")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title=t("chart.risk_axis"))
    fig.update_xaxes(title=x_title)
    return _finish(fig, 300)


def ecg_strip(clean: np.ndarray, fs: float, beats: pd.DataFrame, start: float, length: float,
              hover: str, x_title: str) -> go.Figure:
    """Q: which beats of this stretch of ECG are normal and which are not?

    Grey trace = filtered ECG; one marker above each beat. Colour AND shape
    encode the status (blue circle normal, red diamond abnormal).
    """
    i0, i1 = int(start * fs), int(min(len(clean), (start + length) * fs))
    tt = np.arange(i0, i1) / fs
    fig = go.Figure(go.Scatter(x=tt, y=clean[i0:i1], mode="lines", line=dict(color="#64748B", width=1.2),
                               hoverinfo="skip", showlegend=False))
    win = beats[(beats["t"] >= start) & (beats["t"] < start + length)]
    names = {"normal": t("ecg.normal"), "abnormal": t("ecg.abnormal"), "unclassifiable": t("ecg.unclass")}
    for status in ("normal", "abnormal", "unclassifiable"):
        group = win[win["status"] == status]
        if group.empty:
            continue
        idx = np.clip(np.round(group["t"].values * fs).astype(int), 0, len(clean) - 1)
        style = BEAT_STYLE[status]
        fig.add_scatter(x=group["t"], y=clean[idx] + 0.2, mode="markers", name=names[status],
                        marker=dict(size=12, color=style["color"], symbol=style["symbol"],
                                    line=dict(width=2, color="white")),
                        customdata=np.c_[group["p"].fillna(0)], hovertemplate=hover + "<extra></extra>")
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="mV")
    return _finish(fig, 340, legend=True, dragmode="zoom", margin=dict(l=8, r=16, t=36, b=8))


def tachogram(beats: pd.DataFrame, x_title: str, y_title: str) -> go.Figure:
    """Q: is the rhythm regular over the whole recording?

    One dot per beat: time (x) vs. the interval since the previous beat (y).
    A flat band = regular rhythm; dots that drop below the band = premature beats.
    """
    fig = go.Figure()
    names = {"normal": t("ecg.normal"), "abnormal": t("ecg.abnormal"), "unclassifiable": t("ecg.unclass")}
    for status in ("normal", "abnormal", "unclassifiable"):
        group = beats[beats["status"] == status]
        if group.empty:
            continue
        style = BEAT_STYLE[status]
        fig.add_scatter(x=group["t"], y=group["pre_rr"] * 1000, mode="markers", name=names[status],
                        marker=dict(size=6 if status == "normal" else 8, color=style["color"], symbol=style["symbol"],
                                    opacity=0.8),
                        hovertemplate="%{x:.1f} s · %{y:.0f} ms<extra></extra>")
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    return _finish(fig, 320, legend=True, margin=dict(l=8, r=16, t=36, b=8))


def beat_shape(beat: np.ndarray, template: np.ndarray, fs: float, n_before: int,
               names: tuple[str, str], x_title: str) -> go.Figure:
    """Q: does this beat look like the patient's usual beat?

    Dotted = the patient's usual (median normal) beat; solid = the chosen beat.
    Time 0 is the R peak (the tallest spike of the beat).
    """
    tms = (np.arange(len(beat)) - n_before) / fs * 1000
    fig = go.Figure()
    fig.add_scatter(x=tms, y=template, name=names[0], line=dict(color=MUTED, dash="dot", width=2.5), hoverinfo="skip")
    fig.add_scatter(x=tms, y=beat, name=names[1], line=dict(color=CRITICAL, width=2.5), hoverinfo="skip")
    fig.add_vline(x=0, line_color=AXIS, line_width=1, annotation_text=t("chart.r_peak"), annotation_position="top",
                  annotation_font=dict(size=11, color=INK_2))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(showticklabels=False, title=t("chart.shape_axis"))
    return _finish(fig, 300, legend=True, margin=dict(l=8, r=16, t=36, b=8))


def accuracy_bars(rows: list[tuple[str, str, float]], target: float = 0.95) -> go.Figure:
    """Q: how often is each model right on cases it never saw, and does the system reach 95 %?"""
    labels = [r[1] for r in rows][::-1]
    values = [r[2] for r in rows][::-1]
    colors = [MODEL_COLORS.get(r[0], "#CBD5E1") for r in rows][::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
                           texttemplate="%{x:.1%}", textposition="outside", cliponaxis=False,
                           textfont=dict(color=INK, size=13), hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    _label_line(fig, target, t("chart.target"), color=CRITICAL, dash="dash")
    low = max(0.0, min(values) - 0.08)
    fig.update_xaxes(range=[low, 1.02], tickformat=".0%")
    fig.update_yaxes(tickfont=dict(color=INK, size=13))
    return _finish(fig, 90 + 44 * len(rows), margin=dict(l=8, r=16, t=34, b=8))


def fold_dots(values: list[float], mean: float) -> go.Figure:
    """Q: does the result depend on which patients were left out? (one dot per fold)"""
    labels = [t("chart.fold", n=i + 1) for i in range(len(values))]
    fig = go.Figure(go.Scatter(x=values, y=labels, mode="markers+text", marker=dict(size=14, color=MODEL_COLORS["final"]),
                               text=[f"{v:.1%}" for v in values], textposition="middle right",
                               textfont=dict(color=INK, size=12), hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    fig.add_vline(x=mean, line_color=INK, line_width=1, annotation_text=t("chart.mean") + " " + num(mean, ".1%"),
                  annotation_position="top", annotation_font=dict(size=11, color=INK))
    lo = min(values) - 0.03
    fig.update_xaxes(range=[lo, 1.0], tickformat=".0%")
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK))
    return _finish(fig, 80 + 42 * len(values), margin=dict(l=8, r=16, t=34, b=8))
