import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Revenue vs wRVU Agent", layout="wide")
st.title("Revenue vs wRVU Agent (Clinic Benchmarking)")
st.caption("Upload clinic_code, revenue, wRVU (month ignored). Computes Rev/wRVU, ranks clinics, flags outliers, and answers simple prompts.")

@st.cache_data
def load_csv_flexible(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("File is empty.")

    # Try read with header
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        df = None

    # If headerless 3-col, assume: revenue, clinic_code, wRVU
    if df is None or (df.shape[1] == 3 and set(df.columns) == {0, 1, 2}):
        df = pd.read_csv(io.StringIO(text), header=None)
        df.columns = ["revenue", "clinic_code", "wRVU"]

    df.columns = [c.strip() for c in df.columns]

    # Normalize column names
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["clinic_code", "clinic", "clinicid", "clinic_id"]:
            colmap[c] = "clinic_code"
        elif lc in ["revenue", "net_revenue", "payments", "allowed"]:
            colmap[c] = "revenue"
        elif lc in ["wrvu", "work_rvu", "w_rvu"]:
            colmap[c] = "wRVU"
    df = df.rename(columns=colmap)

    needed = {"clinic_code", "revenue", "wRVU"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing required columns. Need: {sorted(needed)}. Found: {list(df.columns)}")

    df = df[["clinic_code", "revenue", "wRVU"]].copy()
    df["clinic_code"] = df["clinic_code"].astype(str).str.strip()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["wRVU"] = pd.to_numeric(df["wRVU"], errors="coerce")

    bad = df[df[["revenue", "wRVU"]].isna().any(axis=1)]
    if len(bad) > 0:
        raise ValueError("Found non-numeric revenue/wRVU rows:\n" + bad.head(10).to_string(index=False))

    if (df["wRVU"] <= 0).any():
        raise ValueError("wRVU must be > 0 for all rows.")
    if (df["revenue"] < 0).any():
        st.warning("Some revenue values are negative. If that’s expected, ignore this warning.")

    return df.groupby("clinic_code", as_index=False)[["revenue", "wRVU"]].sum()

@st.cache_data
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rev_per_wrvu"] = d["revenue"] / d["wRVU"]
    mean = d["rev_per_wrvu"].mean()
    std = d["rev_per_wrvu"].std(ddof=0)
    d["z_score"] = (d["rev_per_wrvu"] - mean) / (std if std != 0 else np.nan)
    d["pct_rank"] = d["rev_per_wrvu"].rank(pct=True) * 100
    return d

def dispersion_stats(d: pd.DataFrame) -> dict:
    r = d["rev_per_wrvu"]
    mean = float(r.mean())
    std = float(r.std(ddof=0))
    return {
        "mean": mean,
        "median": float(r.median()),
        "std": std,
        "min": float(r.min()),
        "max": float(r.max()),
        "cv": float(std / mean) if mean != 0 else float("nan"),
    }

def agent_answer(msg: str, metrics: pd.DataFrame, z_thresh: float = 2.0):
    t = msg.strip().lower()
    tables = []

    if any(k in t for k in ["rank", "top", "highest", "lowest"]):
        tables.append(("Ranked clinics (Rev/wRVU)", metrics.sort_values("rev_per_wrvu", ascending=False)))
        return "Clinics ranked by Rev/wRVU.", tables

    if any(k in t for k in ["outlier", "anomal", "flag"]):
        out = metrics[metrics["z_score"].abs() >= z_thresh].sort_values("z_score", ascending=False)
        tables.append((f"Outliers (|z| ≥ {z_thresh})", out))
        return f"Clinics flagged as outliers using |z| ≥ {z_thresh}.", tables

    if any(k in t for k in ["worst", "underperform"]):
        worst = metrics.sort_values("rev_per_wrvu", ascending=True).head(1)
        tables.append(("Lowest Rev/wRVU", worst))
        c = worst.iloc[0]["clinic_code"]
        return f"{c} has the lowest Rev/wRVU. Likely drivers: payer mix, contract rates, site-of-service billing, coding mix.", tables

    if any(k in t for k in ["best", "overperform"]):
        best = metrics.sort_values("rev_per_wrvu", ascending=False).head(1)
        tables.append(("Highest Rev/wRVU", best))
        c = best.iloc[0]["clinic_code"]
        return f"{c} has the highest Rev/wRVU. Validate attribution/mapping and payer mix.", tables

    s = dispersion_stats(metrics)
    tables.append(("Clinic metrics", metrics.sort_values("rev_per_wrvu", ascending=False)))
    md = (
        f"Summary: mean Rev/wRVU **{s['mean']:.2f}**, median **{s['median']:.2f}**, "
        f"range **{s['min']:.2f} → {s['max']:.2f}**, CV **{s['cv']:.2f}**.\n\n"
        "Try: `rank clinics`, `show outliers`, `who is worst`, `who is best`."
    )
    return md, tables

with st.sidebar:
    st.header("Upload")
    file = st.file_uploader("CSV (or headerless 3-column txt)", type=["csv", "txt"])
    z_thresh = st.slider("Outlier threshold |z|", 1.0, 5.0, 2.0, 0.1)

if not file:
    st.info("Upload a file with columns: clinic_code, revenue, wRVU.")
    st.stop()

df = load_csv_flexible(file.read(), file.name)
metrics = compute_metrics(df)
stats = dispersion_stats(metrics)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clinics", f"{len(metrics)}")
c2.metric("Mean Rev/wRVU", f"{stats['mean']:.2f}")
c3.metric("Median Rev/wRVU", f"{stats['median']:.2f}")
c4.metric("CV (dispersion)", f"{stats['cv']:.2f}")

st.subheader("Ranked Clinics")
st.dataframe(metrics.sort_values("rev_per_wrvu", ascending=False), use_container_width=True)

st.subheader("Outliers")
out = metrics[metrics["z_score"].abs() >= z_thresh].sort_values("z_score", ascending=False)
st.dataframe(out, use_container_width=True)

st.subheader("Rev/wRVU Chart")
m_sorted = metrics.sort_values("rev_per_wrvu", ascending=False)
fig, ax = plt.subplots()
ax.bar(m_sorted["clinic_code"], m_sorted["rev_per_wrvu"])
ax.tick_params(axis="x", labelrotation=45)
ax.set_ylabel("Revenue per wRVU")
ax.set_xlabel("Clinic")
st.pyplot(fig)
plt.close(fig)

st.divider()
st.subheader("Ask the agent")
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Try: rank clinics | show outliers | who is worst | who is best")
if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    md, tables = agent_answer(prompt, metrics, z_thresh=z_thresh)
    with st.chat_message("assistant"):
        st.markdown(md)
        for name, tdf in tables:
            st.markdown(f"**{name}**")
            st.dataframe(tdf, use_container_width=True)

    st.session_state.chat.append(("assistant", md))
