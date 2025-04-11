import streamlit as st
import pandas as pd

st.set_page_config(page_title="BTC Monday Filter", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("btc_data.csv", parse_dates=["Date"])
    return df

df = load_data()

st.title("📊 BTC Monday Range Filter Tool")

# Sidebar Filters
with st.sidebar:
    st.header("🔍 Filter Options")

    # Date range
    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = st.date_input("Date Range", [min_date, max_date])

    # Monday Range %
    min_range = float(df["Monday Size"].min())
    max_range = float(df["Monday Size"].max())
    monday_range = st.slider("Monday Range (%)", min_value=min_range, max_value=max_range, value=(min_range, max_range))

    # Other Side Taken
    other_side = st.selectbox("Other Side Taken", ["All", "Yes", "No"])

    # Quarter + Year
    selected_years = st.multiselect("Select Year(s)", sorted(df["Year"].unique()), default=df["Year"].unique())
    selected_quarters = st.multiselect("Select Quarter(s)", [1, 2, 3, 4], default=[1, 2, 3, 4])

    # Bull/Bear filters
    week_filter = st.multiselect("Weekly", ["Bull", "Bear"], default=["Bull", "Bear"])
    mon_filter = st.multiselect("Monday", ["Bull", "Bear"], default=["Bull", "Bear"])
    tue_filter = st.multiselect("Tuesday", ["Bull", "Bear"], default=["Bull", "Bear"])

    # Search Button
    run_filter = st.button("🔎 Search")

# Data Filtering
filtered_df = df.copy()

if run_filter:
    filtered_df = filtered_df[
        (filtered_df["Date"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["Date"] <= pd.to_datetime(date_range[1])) &
        (filtered_df["Monday Size"] >= monday_range[0]) &
        (filtered_df["Monday Size"] <= monday_range[1]) &
        (filtered_df["Year"].isin(selected_years)) &
        (df["Month"].apply(lambda x: (x - 1) // 3 + 1).isin(selected_quarters)) &
        (df["Weekly"].isin(week_filter)) &
        (df["Monday"].isin(mon_filter)) &
        (df["Tuesday"].isin(tue_filter))
    ]
    if other_side != "All":
        filtered_df = filtered_df[filtered_df["Other Side Taken"] == other_side]

    st.success(f"✅ {len(filtered_df)} rows matched the filters.")
    st.dataframe(filtered_df, use_container_width=True)
else:
    st.dataframe(df, use_container_width=True)
