"""Analyse an ECG: step 1 = choose the signal, step 2 = result in plain language.

Functionalities 6-7 + save + PDF. The viewer is the only chart on screen; the
beat inspector, the rhythm plot, the table and the technical details live in
expanders.
"""
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from core import charts, ecg, files, report_pdf, storage
from core.i18n import lang, num, t
from core.model_store import require_tier, tier_kpis
from core.ui import (callout, chart, chart_header, esc, guide_html, header, html, muted, result_card, section, stats,
                     steps)

tier = require_tier("ecg")
st.session_state.setdefault("ecg_step", "choose")
LEVEL = {"low": "low", "moderate": "mid", "high": "high"}

header(t("ecg.title"), t("ecg.subtitle"), t("nav.sec_patients"), "ecg_heart")


@st.cache_data(show_spinner=False, max_entries=8)
def run_analysis(signal_key: str, _signal: np.ndarray, fs: float) -> dict:
    return ecg.analyze(tier, _signal, fs)


# ======================================================== step 1: choose ECG
def choose_step() -> None:
    steps([t("ecg.step1"), t("ecg.step2")], 0)
    signal, fs, label = None, None, ""
    with st.container(border=True):
        section(t("ecg.source_q"))
        source = st.segmented_control(t("ecg.source_q"), ["demo", "csv", "wfdb"], default="demo", required=True,
                                      format_func=lambda s: t(f"ecg.src.{s}"), label_visibility="collapsed")
        st.write("")
        if source == "demo":
            demos = ecg.demo_catalog()
            if not demos:
                st.error(t("ecg.no_demos"))
                st.stop()
            captions = []
            for d in demos:
                counts = d["annotated_beats"]
                abnormal = counts["S"] + counts["V"] + counts["F"]
                captions.append(d[f"description_{lang()}"] + " " + t("ecg.demo_labels", n=counts["N"], a=abnormal))
            choice = st.radio(t("ecg.demo_pick"), range(len(demos)), captions=captions,
                              format_func=lambda i: demos[i][f"title_{lang()}"])
            signal, fs = ecg.load_demo(demos[choice])
            label = demos[choice][f"title_{lang()}"]
        elif source == "csv":
            up = st.file_uploader(t("ecg.csv_upload"))          # no browser filter: checked below
            with st.expander(t("ecg.csv_format"), icon=":material/help:"):
                st.markdown(t("ecg.csv_format_text"))
            if up and not files.is_table(up.name):
                st.error(t("files.wrong_csv", name=up.name))
                st.stop()
            if up:
                try:
                    data = ecg.read_csv(up.getvalue())
                except Exception as err:                      # malformed text file
                    st.error(t("files.unreadable", error=err))
                    st.stop()
                if data.empty:
                    st.error(t("ecg.csv_no_numeric"))
                    st.stop()
                a, b, c = st.columns(3)
                col = a.selectbox(t("ecg.column"), list(data.columns), index=len(data.columns) - 1)
                fs = b.number_input(t("ecg.fs"), 50.0, 2000.0, 360.0, help=t("ecg.fs_help"))
                unit = c.segmented_control(t("ecg.units"), ["mV", "µV"], default="mV", required=True)
                signal = data[col].to_numpy(float) / (1000.0 if unit == "µV" else 1.0)
                label = up.name
        else:
            ups = st.file_uploader(t("ecg.wfdb_upload"), accept_multiple_files=True)
            if ups:
                headers = [u for u in ups if files.extension(u.name) == ".hea"]
                if not headers:
                    st.error(t("files.need_hea"))
                    st.stop()
                header = headers[0]
                stem = header.name[:-4]
                if not any(u.name == stem + ".dat" for u in ups):
                    st.error(t("files.need_dat", name=stem))
                    st.stop()
                try:
                    sig, fs, names = ecg.read_wfdb({u.name: u.getvalue() for u in ups
                                                    if u.name.startswith(stem + ".")})
                except Exception as err:
                    st.error(t("files.unreadable", error=err))
                    st.stop()
                default = names.index("MLII") if "MLII" in names else 0
                ch = st.selectbox(t("ecg.lead"), range(len(names)), index=default, format_func=lambda i: names[i])
                signal = sig[:, ch]
                label = header.name

    code = st.text_input(t("common.patient_code"), placeholder=t("common.patient_code_ph"),
                         help=t("common.patient_code_help"))
    if signal is None:
        muted(t("ecg.waiting"))
        return
    muted(t("ecg.ready", s=len(signal) / fs, fs=fs))
    if st.button(t("ecg.analyze"), type="primary", width="stretch", icon=":material/arrow_forward:",
                 icon_position="right"):
        key = hashlib.sha1(np.ascontiguousarray(signal).tobytes()).hexdigest() + f"_{fs}"
        try:
            with st.spinner(t("ecg.running")):
                result = run_analysis(key, signal, fs)
        except ValueError as err:
            st.error(t(f"ecg.error.{err}"))
            return
        st.session_state.update(ecg_step="result", ecg={"result": result, "label": label, "code": code},
                                ecg_start=0.0)
        st.session_state.pop("ecg_saved", None)
        st.rerun()


# ============================================================ step 2: result
def jump(direction: int, abn_times: np.ndarray, length: int, max_start: float) -> None:
    """Move the viewer to the previous/next abnormal beat."""
    centre = st.session_state["ecg_start"] + length / 2
    later = abn_times[abn_times > centre + 0.5] if direction > 0 else abn_times[abn_times < centre - 0.5][::-1]
    if len(later):
        st.session_state["ecg_start"] = float(min(max_start, max(0.0, np.floor(later[0] - length / 2))))


def result_step() -> None:
    state = st.session_state["ecg"]
    res = state["result"]
    beats, s = res["beats"], res["summary"]
    level = LEVEL[ecg.burden_level(s["burden"])]

    if st.button(t("ecg.back"), icon=":material/arrow_back:"):
        st.session_state["ecg_step"] = "choose"
        st.rerun()
    steps([t("ecg.step1"), t("ecg.step2")], 1)

    tag = t("ecg.tag") + " · " + esc(state["label"]) + (f" · {esc(state['code'])}" if state["code"] else "")
    result_card(level, tag, num(s["burden"], ".0%"), t(f"ecg.level.{level}"), t(f"ecg.msg.{level}"),
                t("ecg.big_sub", a=s["abnormal"], n=s["classified"]))
    with st.popover(t("guide.result_button"), icon=":material/help:"):
        html(guide_html({"what": t("ecg.guide_result_what"), "read": t("ecg.guide_result_read")}))
    stats([(t("ecg.stat_hr"), t("ecg.stat_hr_v", v=s["heart_rate"]), t("ecg.stat_hr_h", lo=s["hr_min"], hi=s["hr_max"]),
            "favorite"),
           (t("ecg.stat_dur"), t("ecg.stat_dur_v", m=s["duration_s"] / 60), None, "timer"),
           (t("ecg.stat_beats"), num(s["beats"]), None, "monitor_heart"),
           (t("ecg.stat_run"), num(s["longest_abnormal_run"]), t("ecg.stat_run_h"), "repeat")])

    # -------------------------------------------------- the viewer (7)
    with st.container(border=True):
        chart_header(t("ecg.viewer_title"), t("ecg.viewer_sub"), "ecg_strip")      # the chart carries its own legend
        abn_times = beats.loc[beats["status"] == "abnormal", "t"].to_numpy()
        c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
        length = c1.selectbox(t("ecg.window"), [10, 20, 30, 60], index=1,
                              format_func=lambda n: t("ecg.seconds", n=n))
        max_start = float(max(1.0, np.floor(s["duration_s"] - length)))
        st.session_state["ecg_start"] = min(float(st.session_state.get("ecg_start", 0.0)), max_start)
        start = c2.slider(t("ecg.start"), 0.0, max_start, step=1.0, key="ecg_start", format="%.0f s")
        nav = st.container(horizontal=True)
        nav.button(t("ecg.prev_abn"), icon=":material/chevron_left:", disabled=not len(abn_times),
                   on_click=jump, args=(-1, abn_times, length, max_start))
        nav.button(t("ecg.next_abn"), icon=":material/chevron_right:", icon_position="right",
                   disabled=not len(abn_times), on_click=jump, args=(1, abn_times, length, max_start))
        chart(charts.ecg_strip(res["clean"], res["fs_clean"], beats, start, length, t("ecg.hover"), t("ecg.time_s")))

    # ------------------------------------------------ details on demand
    status_name = {"normal": t("ecg.normal"), "abnormal": t("ecg.abnormal"), "unclassifiable": t("ecg.unclass")}
    with st.expander(t("ecg.exp_beat"), icon=":material/zoom_in:"):
        pool = beats[beats["p"].notna()]
        abn_idx = beats.index[beats["status"] == "abnormal"]
        default = int(abn_idx[len(abn_idx) // 2]) if len(abn_idx) else int(pool.index[len(pool) // 2])
        i = st.selectbox(t("ecg.beat_pick"), list(pool.index), index=list(pool.index).index(default),
                         format_func=lambda j: t("ecg.beat_label", t=beats.loc[j, "t"], status=status_name[beats.loc[j, "status"]]))
        beat = beats.loc[i]
        cols = res["waveform_columns"]
        normal = beats["status"] == "normal"
        template = np.median(beats.loc[normal if normal.any() else beats.index, cols].values, axis=0)
        ratio, corr = float(beat["rr_ratio"]), max(0.0, float(beat["template_corr"])) * 100
        reason = t("ecg.beat_reason_ontime", corr=corr) if abs(1 - ratio) < 0.05 else \
            t("ecg.beat_reason", pct=abs(1 - ratio) * 100, when=t("ecg.earlier" if ratio < 1 else "ecg.later"), corr=corr)
        callout(reason, "neurology")
        chart_header(t("ecg.beat_shape"), t("ecg.beat_shape_sub"), "beat_shape")
        chart(charts.beat_shape(beat[cols].to_numpy(float), template, res["fs_clean"], res["n_before"],
                                (t("ecg.template"), t("ecg.this_beat")), t("ecg.time_ms")))
        chart_header(t("ecg.models_title"), t("ecg.models_sub"), "opinions")
        rows = [(m, t(f"model.{m}"), float(beat[f"p_{m}"])) for m in ("LR", "KNN", "RF", "SVM")]
        rows.append(("final", t("model.final"), float(beat["p"])))
        chart(charts.opinions(rows))

    with st.expander(t("ecg.exp_rhythm"), icon=":material/timeline:"):
        chart_header(t("ecg.exp_rhythm"), t("ecg.rhythm_sub"), "tachogram")
        chart(charts.tachogram(beats, t("ecg.time_s"), t("ecg.rr_ms")))

    with st.expander(t("ecg.exp_table"), icon=":material/table_rows:"):
        table = pd.DataFrame({t("ecg.col_time"): beats["t"].round(2), t("ecg.col_rr"): (beats["pre_rr"] * 1000).round(0),
                              t("ecg.col_p"): (beats["p"] * 100).round(0), t("ecg.col_status"): beats["status"].map(status_name)})
        st.dataframe(table, hide_index=True, width="stretch", height=320,
                     column_config={t("ecg.col_p"): st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d %%")})
        st.download_button(t("common.csv"), table.to_csv(index=False).encode(), "ecg_latidos.csv", "text/csv",
                           icon=":material/download:")

    with st.expander(t("ecg.exp_tech"), icon=":material/science:"):
        k = tier_kpis("ecg")
        html("<p>" + t("ecg.tech_text", sdnn=s["sdnn_ms"], rmssd=s["rmssd_ms"], unc=s["unclassifiable"],
                       ref=s["referred_beats"], known=k.get("accuracy", 0), new=k.get("new_accuracy", 0),
                       sd=k.get("new_accuracy_sd", 0)) + "</p>")

    # ------------------------------------------------------ save + PDF
    st.write("")
    row = st.container(horizontal=True)
    if row.button(t("common.save"), icon=":material/save:", disabled="ecg_saved" in st.session_state):
        st.session_state["ecg_saved"] = storage.save("ecg", state["code"], s["burden"], level,
                                                     {"source": state["label"]}, s, None, level != "low")
        st.rerun()
    rows = [(t("common.patient_code"), state["code"] or "-"), (t("ecg.source_q"), state["label"]),
            (t("pdf.result"), t(f"ecg.level.{level}")), (t("ecg.stat_beats"), num(s["beats"])),
            (t("ecg.abnormal"), f"{num(s['abnormal'])} ({num(s['burden'], '.1%')})"),
            (t("ecg.stat_hr"), t("ecg.stat_hr_v", v=s["heart_rate"])),
            (t("ecg.stat_dur"), t("ecg.stat_dur_v", m=s["duration_s"] / 60)),
            (t("ecg.stat_run"), num(s["longest_abnormal_run"]))]
    pdf = report_pdf.build(t("pdf.ecg_title"), [(t("pdf.result"), rows), (t("pdf.recommendation"), t(f"ecg.msg.{level}"))],
                           t("common.disclaimer"))
    row.download_button(t("common.pdf"), pdf, file_name=f"ecg_{state['code'] or 'paciente'}.pdf", mime="application/pdf",
                        icon=":material/picture_as_pdf:")
    if "ecg_saved" in st.session_state:
        st.success(t("common.saved", id=st.session_state["ecg_saved"]), icon=":material/check_circle:")


if st.session_state["ecg_step"] == "result" and "ecg" in st.session_state:
    result_step()
else:
    choose_step()
