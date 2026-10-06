"""App de pr\u00e9vision de consommation \u00e9lectrique."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TEAL = "#0F766E"
ORANGE = "#E4572E"
NAMES = {
    "naive": "Na\u00eff (dernier jour)",
    "seasonal_naive_7": "Saisonnier 7 jours",
    "seasonal_naive_365": "Saisonnier 365 jours",
    "ets_weekly": "ETS hebdomadaire",
    "sarima_weekly": "SARIMA hebdomadaire",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "ets_prophet": "ETS + Prophet",
}
BASELINE = "seasonal_naive_7"

st.set_page_config(page_title="Pr\u00e9vision \u00e9lectricit\u00e9", layout="wide")


@st.cache_data
def load():
    daily = pd.read_parquet("data/daily.parquet")["kwh"]
    fc = pd.read_parquet("data/forecast.parquet")
    bt = pd.read_parquet("data/backtest_results.parquet").dropna(subset=["y", "pred"])
    with open("models/metrics.json", encoding="ascii") as f:
        metrics = json.load(f)
    return daily, fc, bt, metrics


daily, fc, bt, metrics = load()
best = metrics["best_model"]
m = metrics["models"]
base_mae, best_mae = m[BASELINE]["mae"], m[best]["mae"]
gain = (base_mae - best_mae) / base_mae * 100

st.title("Pr\u00e9vision de la consommation \u00e9lectrique d'un foyer")
st.caption("Donn\u00e9es UCI, un foyer \u00e0 Sceaux, 2006-2010. Pr\u00e9vision \u00e0 7 jours, "
           "valid\u00e9e par backtest \u00e0 origine glissante (%d fen\u00eatres)." % metrics["n_windows"])

c1, c2, c3 = st.columns(3)
c1.metric("Erreur moyenne (MAE)", "%.2f kWh/jour" % best_mae)
c2.metric("Gain vs baseline", "%.0f %%" % gain, help="Baseline : m\u00eame jour de la semaine pr\u00e9c\u00e9dente")
c3.metric("Couverture de la fourchette", "%.0f %%" % (metrics["holdout_coverage"] * 100),
          help="Mesur\u00e9e hors \u00e9chantillon, pour une fourchette annonc\u00e9e \u00e0 80 %")

tab1, tab2, tab3 = st.tabs(["Pr\u00e9vision", "Comparaison des mod\u00e8les", "R\u00e9sidus"])

with tab1:
    span = st.segmented_control("P\u00e9riode affich\u00e9e", ["60 jours", "180 jours", "Tout"],
                                default="60 jours")
    n = {"60 jours": 60, "180 jours": 180}.get(span, len(daily))
    hist = daily.iloc[-n:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, mode="lines", name="Mesur\u00e9",
                             line=dict(color=TEAL, width=1.5)))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["high"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["low"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(228,87,46,0.18)",
                             name="Fourchette 80 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["pred"], mode="lines+markers", name="Pr\u00e9vision",
                             line=dict(color=ORANGE, width=2.5)))
    fig.update_layout(template="plotly_white", height=460, yaxis_title="kWh par jour",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    show = fc.round(1).rename(columns={"pred": "Pr\u00e9vision (kWh)", "low": "Bas", "high": "Haut"})
    show.index = show.index.strftime("%d/%m/%Y")
    st.dataframe(show, width="stretch")

with tab2:
    rows = []
    for key, v in m.items():
        rows.append({"Mod\u00e8le": NAMES[key], "MAE (kWh)": v["mae"], "RMSE (kWh)": v["rmse"],
                     "sMAPE (%)": v["smape"], "Gain vs baseline (%)": (base_mae - v["mae"]) / base_mae * 100})
    tab = pd.DataFrame(rows).sort_values("MAE (kWh)").reset_index(drop=True)
    st.dataframe(tab.style.format({c: '{:.2f}' for c in tab.columns[1:]}), width="stretch", hide_index=True)

    bar = go.Figure(go.Bar(x=tab["MAE (kWh)"], y=tab["Mod\u00e8le"], orientation="h",
                           marker_color=[TEAL if x == NAMES[best] else "#94A3B8" for x in tab["Mod\u00e8le"]]))
    bar.update_layout(template="plotly_white", height=380, xaxis_title="MAE (kWh par jour)",
                      yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar, width="stretch")

    st.subheader("Erreur selon l'horizon de pr\u00e9vision")
    bt["err"] = (bt["y"] - bt["pred"]).abs()
    by_h = bt.pivot_table(index="h", columns="model", values="err", aggfunc="mean")
    chosen = st.multiselect("Mod\u00e8les", list(NAMES.values()),
                            default=[NAMES[BASELINE], NAMES["ets_weekly"], NAMES["prophet"], NAMES[best]])
    line = go.Figure()
    for key, label in NAMES.items():
        if label in chosen:
            line.add_trace(go.Scatter(x=by_h.index, y=by_h[key], mode="lines+markers", name=label))
    line.update_layout(template="plotly_white", height=380, xaxis_title="Horizon (jours)",
                       yaxis_title="MAE (kWh)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(line, width="stretch")
    st.caption("L'\u00e9cart entre ETS+Prophet et ETS seul (0,06 kWh, intervalle de confiance : -0,28 \u00e0 +0,14) n'est pas significatif "
               "sur %d fen\u00eatres : le r\u00e9sultat solide est le gain sur la baseline." % metrics["n_windows"])

with tab3:
    r = bt[bt["model"] == best].copy()
    r["resid"] = r["y"] - r["pred"]
    a, b = st.columns(2)
    hist_fig = go.Figure(go.Histogram(x=r["resid"], nbinsx=40, marker_color=TEAL))
    hist_fig.update_layout(template="plotly_white", height=360, title="Distribution des r\u00e9sidus",
                           xaxis_title="Mesur\u00e9 - pr\u00e9vu (kWh)", margin=dict(l=10, r=10, t=40, b=10))
    a.plotly_chart(hist_fig, width="stretch")
    t = r.groupby("date")["resid"].mean()
    sc = go.Figure(go.Scatter(x=t.index, y=t, mode="markers", marker=dict(color=TEAL, size=5)))
    sc.add_hline(y=0, line_color=ORANGE)
    sc.update_layout(template="plotly_white", height=360, title="R\u00e9sidus dans le temps",
                     yaxis_title="kWh", margin=dict(l=10, r=10, t=40, b=10))
    b.plotly_chart(sc, width="stretch")
    st.caption("Biais moyen : %+.2f kWh. Fourchette 80 %% : [%.1f ; %+.1f] kWh."
               % (r["resid"].mean(), metrics["q10"], metrics["q90"]))
