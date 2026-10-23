"""Design tokens and the Plotly template shared by every chart of the app.

One place for colours, so a model or a beat type keeps the same colour on
every page. The categorical palette was checked with a colour-vision-deficiency
simulator (neighbouring colours stay apart for protan, deutan and tritan
vision); status colours (green / amber / red) are only used for good /
warning / bad meanings and always next to a text label.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ------------------------------------------------------------------ tokens
BRAND = "#2563EB"
INK, INK_2, MUTED, GRID, AXIS = "#0F172A", "#475569", "#94A3B8", "#EEF1F5", "#CBD5E1"
SURFACE = "#FFFFFF"

# categorical slots, fixed order (same as the ML pipeline figures)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#6250D6"
MODEL_COLORS = {"LR": BLUE, "KNN": ORANGE, "RF": AQUA, "SVM": YELLOW, "final": INK}

# status (meaning, never identity)
GOOD, WARNING, CRITICAL = "#16A34A", "#D97706", "#DC2626"
LEVEL_COLORS = {"low": GOOD, "mid": WARNING, "high": CRITICAL, "info": BRAND}

# ECG beats: colour AND marker shape, so the meaning never relies on colour alone
BEAT_STYLE = {
    "normal": {"color": BLUE, "symbol": "circle"},
    "abnormal": {"color": CRITICAL, "symbol": "diamond"},
    "unclassifiable": {"color": MUTED, "symbol": "circle-open"},
}
DOUBT_BAND = "#FEF3C7"

FONT = "Inter, -apple-system, 'Segoe UI', sans-serif"


def register() -> go.layout.Template:
    """Register the 'cardioai' Plotly template and make it the default."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT, size=13, color=INK_2),
        title=dict(font=dict(size=15, color=INK)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,      # charts always sit on white cards
        colorway=[BLUE, ORANGE, AQUA, YELLOW, VIOLET],
        margin=dict(l=8, r=16, t=16, b=8),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, size=13, color=INK)),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12, color=INK_2),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", automargin=True,
                   title=dict(font=dict(size=12, color=INK_2))),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", automargin=True,
                   title=dict(font=dict(size=12, color=INK_2))),
        bargap=0.35,
    )
    pio.templates["cardioai"] = template
    pio.templates.default = "cardioai"
    return template


TEMPLATE = register()
