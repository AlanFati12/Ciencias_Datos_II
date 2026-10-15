"""Visual building blocks shared by every page (one design system for the whole app).

Texts passed to these helpers come from i18n (trusted). Anything typed by the
user (patient code, file names) must go through `esc()` first.
"""
from __future__ import annotations

from html import escape as esc  # noqa: F401  (re-exported for the pages)

import streamlit as st

from .guides import guide
from .i18n import t

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

CSS = """
<style>
:root {
  --cv-bg: #F6F8FB; --cv-surface: #FFFFFF; --cv-ink: #0F172A; --cv-ink-2: #475569; --cv-muted: #64748B;
  --cv-line: #E6EAF0; --cv-brand: #2563EB; --cv-brand-soft: #EFF4FF;
  --cv-good: #16A34A; --cv-warn: #D97706; --cv-bad: #DC2626;
  --cv-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 6px 20px rgba(15, 23, 42, .05);
}
/* page canvas and cards */
[data-testid="stAppViewContainer"], [data-testid="stMain"] {background: var(--cv-bg);}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1120px;}
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
  background: var(--cv-surface); border-color: var(--cv-line) !important; border-radius: 16px;
  box-shadow: var(--cv-shadow); padding: 1.15rem 1.25rem;}
[data-testid="stSidebar"] {background: var(--cv-surface); border-right: 1px solid var(--cv-line);}
[data-testid="stExpander"] details {background: var(--cv-surface); border-color: var(--cv-line); border-radius: 14px;}
[data-testid="stExpander"] summary {font-weight: 600;}
.stTabs [data-baseweb="tab-list"] {gap: .25rem; border-bottom: 1px solid var(--cv-line);}
.stTabs [data-baseweb="tab"] {font-weight: 600; padding: .55rem .9rem;}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {box-shadow: 0 4px 14px rgba(37, 99, 235, .25);}
h1, h2, h3 {letter-spacing: -0.015em; color: var(--cv-ink);}
/* Streamlit paints the plot area with the theme's secondary colour even with theme=None: keep it white */
.js-plotly-plot .plotly rect.bg {fill: #FFFFFF !important;}

/* icons inside custom HTML use the Material Symbols font that Streamlit already loads */
.cv-ic {font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal; line-height: 1;
        letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr;
        -webkit-font-feature-settings: "liga"; font-feature-settings: "liga"; -webkit-font-smoothing: antialiased;}

/* page header */
.cv-header {margin-bottom: .5rem;}
.cv-eyebrow {display: inline-flex; align-items: center; gap: .35rem; font-size: .78rem; font-weight: 700;
             letter-spacing: .08em; text-transform: uppercase; color: var(--cv-brand); margin-bottom: .35rem;}
.cv-eyebrow .cv-ic {font-size: 1.05rem;}
.cv-header h1 {font-size: 2rem; margin: 0 0 .3rem 0; padding: 0; line-height: 1.15;}
.cv-header p {color: var(--cv-ink-2); font-size: 1.04rem; margin: 0; max-width: 780px;}

/* hero (home) */
.cv-hero {position: relative; overflow: hidden; border-radius: 22px; padding: 2.1rem 2.2rem 4.6rem;
          background: radial-gradient(1200px 300px at 0% 0%, #DCE8FF 0%, rgba(220,232,255,0) 60%),
                      linear-gradient(135deg, #F3F7FF 0%, #FFFFFF 70%);
          border: 1px solid var(--cv-line); box-shadow: var(--cv-shadow); margin-bottom: 1.2rem;}
.cv-hero h1 {font-size: 2.45rem; line-height: 1.12; margin: .2rem 0 .7rem; padding: 0; max-width: 760px;}
.cv-hero p {font-size: 1.12rem; color: var(--cv-ink-2); max-width: 700px; margin: 0;}
.cv-badges {display: flex; gap: .5rem; flex-wrap: wrap; margin-top: 1.1rem;}
.cv-badge {display: inline-flex; align-items: center; gap: .35rem; background: rgba(255,255,255,.85);
           border: 1px solid var(--cv-line); border-radius: 999px; padding: .3rem .75rem; font-size: .85rem;
           color: var(--cv-ink-2); font-weight: 500;}
.cv-badge .cv-ic {font-size: 1rem; color: var(--cv-brand);}
.cv-pulse {position: absolute; left: 0; right: 0; bottom: 10px; height: 64px; opacity: .9;}
.cv-pulse path {fill: none; stroke: #2563EB; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round;
                stroke-dasharray: 1400; stroke-dashoffset: 1400; animation: cv-draw 4.5s ease-in-out infinite;}
@keyframes cv-draw {0% {stroke-dashoffset: 1400;} 55% {stroke-dashoffset: 0;} 100% {stroke-dashoffset: -1400;}}
@media (prefers-reduced-motion: reduce) {.cv-pulse path {animation: none; stroke-dashoffset: 0;}}

/* feature cards */
.cv-card-icon {width: 3rem; height: 3rem; border-radius: 14px; display: grid; place-items: center; margin-bottom: .75rem;}
.cv-card-icon .cv-ic {font-size: 1.6rem;}
.cv-card-title {font-size: 1.25rem; font-weight: 700; color: var(--cv-ink); margin: 0 0 .3rem;}
.cv-card-text {color: var(--cv-ink-2); font-size: .97rem; margin: 0 0 .7rem; min-height: 3rem;}
.cv-meta {display: flex; gap: .9rem; flex-wrap: wrap; color: var(--cv-muted); font-size: .85rem; margin-bottom: .9rem;}
.cv-meta span {display: inline-flex; align-items: center; gap: .25rem;}
.cv-meta .cv-ic {font-size: 1rem;}

/* numbered steps (how it works) and wizard steps */
.cv-num {width: 2.1rem; height: 2.1rem; border-radius: 999px; background: var(--cv-brand-soft); color: var(--cv-brand);
         font-weight: 700; display: grid; place-items: center; margin-bottom: .55rem;}
.cv-steps {display: flex; align-items: center; gap: .6rem; margin: .9rem 0 1.2rem; flex-wrap: wrap;}
.cv-step {display: flex; align-items: center; gap: .5rem; color: #94A3B8; font-weight: 600; font-size: .92rem;}
.cv-step .n {width: 1.7rem; height: 1.7rem; border-radius: 999px; display: grid; place-items: center;
             background: #E2E8F0; color: #64748B; font-size: .82rem;}
.cv-step.active {color: var(--cv-ink);}
.cv-step.active .n {background: var(--cv-brand); color: #FFFFFF; box-shadow: 0 0 0 4px #DBEAFE;}
.cv-step.done {color: #15803D;}
.cv-step.done .n {background: #DCFCE7; color: #15803D;}
.cv-step-sep {width: 2.6rem; height: 2px; background: #E2E8F0; border-radius: 2px;}

/* result card */
.cv-result {border-radius: 20px; padding: 1.5rem 1.7rem 1.25rem; border: 1px solid; margin: .2rem 0 1.1rem;
            box-shadow: var(--cv-shadow);}
.cv-result.low {background: linear-gradient(135deg, #ECFDF3, #FFFFFF 70%); border-color: #BBF7D0;}
.cv-result.mid {background: linear-gradient(135deg, #FFF8E6, #FFFFFF 70%); border-color: #FDE68A;}
.cv-result.high {background: linear-gradient(135deg, #FEF0F0, #FFFFFF 70%); border-color: #FECACA;}
.cv-result.info {background: linear-gradient(135deg, #EEF4FF, #FFFFFF 70%); border-color: #BFDBFE;}
.cv-result .tag {display: inline-flex; align-items: center; gap: .35rem; font-size: .78rem; font-weight: 700;
                 letter-spacing: .07em; text-transform: uppercase;}
.cv-result .tag .cv-ic {font-size: 1.05rem;}
.cv-result.low .tag {color: #15803D;} .cv-result.mid .tag {color: #B45309;}
.cv-result.high .tag {color: #B91C1C;} .cv-result.info .tag {color: #1D4ED8;}
.cv-result .row {display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap; margin-top: .45rem;}
.cv-result .big {font-size: 3.3rem; font-weight: 800; line-height: 1; min-width: 6.8rem; letter-spacing: -0.03em;}
.cv-result.low .big {color: var(--cv-good);} .cv-result.mid .big {color: var(--cv-warn);}
.cv-result.high .big {color: var(--cv-bad);} .cv-result.info .big {color: var(--cv-brand);}
.cv-result .title {font-size: 1.4rem; font-weight: 700; color: var(--cv-ink); margin: 0 0 .2rem;}
.cv-result .msg {color: #334155; font-size: 1rem; margin: 0; max-width: 680px;}
.cv-result .sub {color: var(--cv-ink-2); font-size: .9rem; margin-top: .75rem;}
.cv-bar {position: relative; height: 12px; border-radius: 999px; margin: 1.3rem 0 .35rem;
         background: linear-gradient(90deg, #86EFAC 0 35%, #FDE68A 35% 65%, #FCA5A5 65% 100%);}
.cv-bar .marker {position: absolute; top: -7px; width: 6px; height: 26px; border-radius: 3px;
                 background: var(--cv-ink); transform: translateX(-50%); box-shadow: 0 0 0 3px #FFFFFF;}
.cv-bar-labels {display: flex; justify-content: space-between; font-size: .78rem; color: var(--cv-muted);}

/* stat tiles */
.cv-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .75rem; margin: .3rem 0 1.1rem;}
.cv-stat {background: var(--cv-surface); border: 1px solid var(--cv-line); border-radius: 14px; padding: .85rem 1rem;
          box-shadow: 0 1px 2px rgba(15, 23, 42, .04);}
.cv-stat .k {display: flex; align-items: center; gap: .35rem; font-size: .82rem; color: var(--cv-muted);}
.cv-stat .k .cv-ic {font-size: 1.05rem; color: var(--cv-brand);}
.cv-stat .v {font-size: 1.45rem; font-weight: 700; color: var(--cv-ink); line-height: 1.3;}
.cv-stat .h {font-size: .78rem; color: #94A3B8;}

/* factor list */
.cv-factor {display: grid; grid-template-columns: 2rem 1fr auto; gap: .8rem; align-items: center;
            padding: .7rem .2rem; border-bottom: 1px solid #F1F5F9;}
.cv-factor:last-child {border-bottom: none;}
.cv-factor .ic {width: 2rem; height: 2rem; border-radius: 10px; display: grid; place-items: center; font-weight: 700;}
.cv-factor.up .ic {background: #FEE2E2; color: var(--cv-bad);}
.cv-factor.down .ic {background: #DCFCE7; color: var(--cv-good);}
.cv-factor .name {font-weight: 600; color: var(--cv-ink);}
.cv-factor .val {font-size: .87rem; color: var(--cv-muted);}
.cv-factor .pts {font-weight: 700; font-size: .95rem; white-space: nowrap;}
.cv-factor.up .pts {color: var(--cv-bad);} .cv-factor.down .pts {color: var(--cv-good);}
.cv-factor .bar {height: 6px; border-radius: 999px; margin-top: .35rem; max-width: 260px;}
.cv-factor.up .bar {background: #FCA5A5;} .cv-factor.down .bar {background: #86EFAC;}

/* text blocks */
.cv-section {font-weight: 700; font-size: 1.08rem; color: var(--cv-ink); margin: 0 0 .15rem;}
.cv-section-sub {color: var(--cv-muted); font-size: .9rem; margin: 0 0 .6rem;}
.cv-muted {color: var(--cv-muted); font-size: .9rem;}
.cv-callout {display: flex; gap: .65rem; align-items: flex-start; background: var(--cv-brand-soft); border: 1px solid #C7D7FE;
             border-radius: 14px; padding: .85rem 1rem; color: #1E3A8A; font-size: .95rem; margin: .4rem 0 .8rem;}
.cv-callout .cv-ic {font-size: 1.25rem; margin-top: .05rem;}
.cv-warn {display: flex; gap: .65rem; align-items: flex-start; background: #FFFBEB; border: 1px solid #FDE68A;
          border-radius: 14px; padding: .75rem 1rem; color: #92400E; font-size: .92rem; margin: .4rem 0 .8rem;}
.cv-warn .cv-ic {font-size: 1.2rem;}
.cv-chip {display: inline-flex; align-items: center; gap: .3rem; padding: .2rem .7rem; border-radius: 999px;
          font-size: .82rem; font-weight: 600;}
.cv-chip.low {background: #DCFCE7; color: #15803D;} .cv-chip.mid {background: #FEF3C7; color: #B45309;}
.cv-chip.high {background: #FEE2E2; color: #B91C1C;} .cv-chip.info {background: #E0E7FF; color: #3730A3;}
.cv-chip.gray {background: #F1F5F9; color: #475569;}
.cv-legend {display: flex; gap: 1.1rem; flex-wrap: wrap; font-size: .86rem; color: var(--cv-ink-2); margin: .2rem 0 .4rem;}
.cv-legend span {display: inline-flex; align-items: center; gap: .35rem;}
.cv-legend b {color: var(--c); font-size: 1rem;}
.cv-guide h4 {font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; color: var(--cv-brand); margin: .6rem 0 .15rem;}
.cv-guide p {margin: 0 0 .3rem; color: var(--cv-ink-2); font-size: .93rem;}
.cv-guide .note {background: #F8FAFC; border-left: 3px solid var(--cv-brand); padding: .45rem .7rem; border-radius: 6px;
                 margin-top: .5rem; font-size: .9rem; color: var(--cv-ink-2);}
.cv-tour {text-align: left;}
.cv-tour .big-ic {width: 3.4rem; height: 3.4rem; border-radius: 16px; display: grid; place-items: center;
                  background: var(--cv-brand-soft); color: var(--cv-brand); margin-bottom: .7rem;}
.cv-tour .big-ic .cv-ic {font-size: 1.9rem;}
.cv-tour h3 {margin: 0 0 .4rem; font-size: 1.3rem;}
.cv-tour p, .cv-tour li {color: var(--cv-ink-2); font-size: .98rem;}
.cv-dots {display: flex; gap: .4rem; margin: .6rem 0 .2rem;}
.cv-dots span {width: .55rem; height: .55rem; border-radius: 999px; background: #CBD5E1;}
.cv-dots span.on {background: var(--cv-brand); width: 1.4rem;}
.cv-footer {color: #94A3B8; font-size: .8rem; margin-top: 2.5rem; border-top: 1px solid var(--cv-line); padding-top: .8rem;}
[data-testid="stSidebar"] .cv-side-note {color: #94A3B8; font-size: .78rem; line-height: 1.4;}

/* phones */
@media (max-width: 640px) {
  .block-container {padding-left: 1rem; padding-right: 1rem;}
  .cv-hero {padding: 1.4rem 1.2rem 4rem; border-radius: 18px;}
  .cv-hero h1 {font-size: 1.75rem; hyphens: auto;}
  .cv-hero p {font-size: 1rem;}
  .cv-header h1 {font-size: 1.6rem;}
  .cv-result {padding: 1.1rem 1.1rem 1rem;}
  .cv-result .big {font-size: 2.5rem; min-width: 0;}
  .cv-result .title {font-size: 1.2rem;}
  [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {padding: .9rem .9rem;}
}
</style>
"""

PULSE_SVG = (
    '<svg class="cv-pulse" viewBox="0 0 1200 64" preserveAspectRatio="none" aria-hidden="true"><path d="'
    "M0 40 L180 40 L200 36 L215 40 L240 40 L252 46 L262 8 L274 56 L286 40 L330 40 L350 32 L380 40 L560 40 "
    "L580 36 L595 40 L620 40 L632 46 L642 8 L654 56 L666 40 L710 40 L730 32 L760 40 L940 40 L960 36 L975 40 "
    'L1000 40 L1012 46 L1022 8 L1034 56 L1046 40 L1090 40 L1110 32 L1140 40 L1200 40"/></svg>')


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def icon(name: str) -> str:
    """A Material Symbols icon for custom HTML (e.g. icon('favorite'))."""
    return f'<span class="cv-ic">{name}</span>'


def header(title: str, subtitle: str = "", eyebrow: str = "", eyebrow_icon: str = "") -> None:
    eb = (f'<div class="cv-eyebrow">{icon(eyebrow_icon) if eyebrow_icon else ""}{eyebrow}</div>' if eyebrow else "")
    html(f'<div class="cv-header">{eb}<h1>{title}</h1>{f"<p>{subtitle}</p>" if subtitle else ""}</div>')


def hero(title: str, subtitle: str, badges: list[tuple[str, str]], eyebrow: str = "") -> None:
    chips = "".join(f'<span class="cv-badge">{icon(i)}{text}</span>' for i, text in badges)
    eb = f'<div class="cv-eyebrow">{icon("ecg_heart")}{eyebrow}</div>' if eyebrow else ""
    html(f'<div class="cv-hero">{eb}<h1>{title}</h1><p>{subtitle}</p><div class="cv-badges">{chips}</div>'
         f"{PULSE_SVG}</div>")


def steps(labels: list[str], current: int) -> None:
    parts = []
    for i, label in enumerate(labels):
        state = "done" if i < current else "active" if i == current else ""
        mark = "✓" if i < current else str(i + 1)
        parts.append(f'<div class="cv-step {state}"><span class="n">{mark}</span>{label}</div>')
    html('<div class="cv-steps">' + '<div class="cv-step-sep"></div>'.join(parts) + "</div>")


LEVEL_ICON = {"low": "check_circle", "mid": "help", "high": "warning", "info": "info"}


def result_card(level: str, tag: str, big: str, title: str, message: str, sub: str = "",
                bar: float | None = None, bar_labels: tuple[str, str, str] | None = None) -> None:
    bar_html = ""
    if bar is not None:
        left = min(max(bar * 100, 1.5), 98.5)
        labels = "".join(f"<span>{x}</span>" for x in (bar_labels or ("", "", "")))
        bar_html = (f'<div class="cv-bar"><div class="marker" style="left:{left:.1f}%"></div></div>'
                    f'<div class="cv-bar-labels">{labels}</div>')
    html(f'<div class="cv-result {level}"><div class="tag">{icon(LEVEL_ICON.get(level, "info"))}{tag}</div>'
         f'<div class="row"><div class="big">{big}</div><div><div class="title">{title}</div>'
         f'<p class="msg">{message}</p></div></div>{bar_html}'
         f'{f"<div class=sub>{sub}</div>" if sub else ""}</div>')


def stats(items: list[tuple]) -> None:
    """items: (label, value, hint[, icon])."""
    cells = []
    for item in items:
        k, v, h = item[:3]
        ic = icon(item[3]) if len(item) > 3 and item[3] else ""
        cells.append(f'<div class="cv-stat"><div class="k">{ic}{k}</div><div class="v">{v}</div>'
                     f'{f"<div class=h>{h}</div>" if h else ""}</div>')
    html(f'<div class="cv-stats">{"".join(cells)}</div>')


def factors(rows: list[dict]) -> None:
    """rows: {'up': bool, 'name': str, 'value': str, 'points': str, 'size': 0..1}"""
    out = []
    for r in rows:
        cls, arrow = ("up", "↑") if r["up"] else ("down", "↓")
        width = max(4, min(100, r.get("size", 0) * 100))
        out.append(f'<div class="cv-factor {cls}"><div class="ic">{arrow}</div>'
                   f'<div><div class="name">{r["name"]}</div><div class="val">{r["value"]}</div>'
                   f'<div class="bar" style="width:{width:.0f}%"></div></div>'
                   f'<div class="pts">{r["points"]}</div></div>')
    html("".join(out))


def section(title: str, subtitle: str = "") -> None:
    html(f'<div class="cv-section">{title}</div>' + (f'<div class="cv-section-sub">{subtitle}</div>' if subtitle else ""))


def chip(text: str, level: str = "info") -> str:
    return f'<span class="cv-chip {level}">{text}</span>'


def callout(text: str, ic: str = "info") -> None:
    html(f'<div class="cv-callout">{icon(ic)}<div>{text}</div></div>')


def warn(text: str) -> None:
    html(f'<div class="cv-warn">{icon("warning")}<div>{text}</div></div>')


def muted(text: str) -> None:
    html(f'<div class="cv-muted">{text}</div>')


def legend(items: list[tuple[str, str, str]]) -> None:
    """items: (colour, symbol, label) — the symbol repeats the chart marker, so colour is never alone."""
    html('<div class="cv-legend">' + "".join(f'<span style="--c:{c}"><b>{s}</b>{label}</span>' for c, s, label in items)
         + "</div>")


def footer(text: str) -> None:
    html(f'<div class="cv-footer">{text}</div>')


# ----------------------------------------------------------------- charts
def guide_html(g: dict[str, str]) -> str:
    out = (f'<div class="cv-guide"><h4>{t("guide.what")}</h4><p>{g["what"]}</p>'
           f'<h4>{t("guide.read")}</h4><p>{g["read"]}</p>')
    if g.get("note"):
        out += f'<div class="note">{g["note"]}</div>'
    return out + "</div>"


def chart_header(title: str, subtitle: str = "", guide_key: str | None = None) -> None:
    """Section title of a chart with a «How to read this chart?» popover on the right."""
    section(title, subtitle)
    if guide_key:              # under the title, so it never gets squeezed on narrow screens
        with st.popover(t("guide.button"), icon=":material/help:"):
            html(guide_html(guide(guide_key)))


def chart(fig, key: str | None = None) -> None:
    """Render a Plotly figure with the app's own template (not Streamlit's theme)."""
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key=key)


# -------------------------------------------------------------- onboarding
TOUR = [("waving_hand", "tour.1"), ("monitor_heart", "tour.2"), ("traffic", "tour.3"), ("rocket_launch", "tour.4")]


@st.dialog(" ", width="medium")
def tour() -> None:
    """Short guided tour shown on the first visit (and from the sidebar)."""
    i = st.session_state.get("tour_step", 0)
    ic, key = TOUR[i]
    dots = "".join(f'<span class="{"on" if j == i else ""}"></span>' for j in range(len(TOUR)))
    html(f'<div class="cv-tour"><div class="big-ic">{icon(ic)}</div><h3>{t(key + "_t")}</h3>'
         f'<div>{t(key)}</div><div class="cv-dots">{dots}</div></div>')
    row = st.container(horizontal=True, horizontal_alignment="distribute")
    if i > 0:
        if row.button(t("tour.back"), icon=":material/arrow_back:"):
            st.session_state["tour_step"] = i - 1
            st.rerun(scope="fragment")
    else:
        if row.button(t("tour.skip")):
            st.session_state.update(onboarded=True, tour_step=0)
            st.rerun()
    if i < len(TOUR) - 1:
        if row.button(t("tour.next"), type="primary", icon=":material/arrow_forward:", icon_position="right"):
            st.session_state["tour_step"] = i + 1
            st.rerun(scope="fragment")
    else:
        if row.button(t("tour.start"), type="primary", icon=":material/check:"):
            st.session_state.update(onboarded=True, tour_step=0)
            st.rerun()
