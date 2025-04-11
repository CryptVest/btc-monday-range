import streamlit as st
import pandas as pd

st.set_page_config(page_title="BTC Monday Filter", layout="wide")

@st.cache_data
def load_data():
    # Load semicolon-delimited CSV
    df = pd.read_csv("btc_data.csv", delimiter=";")

    # Check for required column
    if "Date" not in df.columns:
        st.error("❌ 'Date' column not found in CSV. Check delimiter and column names.")
        st.stop()

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d, %Y")

    # Parse Monday Size from string with %
    df["Monday Size"] = df["Monday Size"].str.rstrip('%').astype(float)

    # Convert Year to integer
    df["Year"] = df["Year"].astype(int)

    # Convert Month name to number
    df["Month_Num"] = pd.to_datetime(df["Month"], format="%B").dt.month
    df["Quarter"] = ((df["Month_Num"] - 1) // 3 + 1)

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

    # Year & Quarter
    selected_years = st.multiselect("Select Year(s)", sorted(df["Year"].unique()), default=df["Year"].unique())
    selected_quarters = st.multiselect("Select Quarter(s)", [1, 2, 3, 4], default=[1, 2, 3, 4])

    # Weekly direction
    week_filter = st.multiselect("Weekly", df["Weekly"].unique(), default=df["Weekly"].unique())
    mon_filter = st.multiselect("Monday", df["Monday"].unique(), default=df["Monday"].unique())
    tue_filter = st.multiselect("Tuesday", df["Tuesday"].unique(), default=df["Tuesday"].unique())

    # Filter button
    run_filter = st.button("🔎 Search")

# Filter logic
filtered_df = df.copy()

if run_filter:
    filtered_df = filtered_df[
        (filtered_df["Date"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["Date"] <= pd.to_datetime(date_range[1])) &
        (filtered_df["Monday Size"] >= monday_range[0]) &
        (filtered_df["Monday Size"] <= monday_range[1]) &
        (filtered_df["Year"].isin(selected_years)) &
        (filtered_df["Quarter"].isin(selected_quarters)) &
        (filtered_df["Weekly"].isin(week_filter)) &
        (filtered_df["Monday"].isin(mon_filter)) &
        (filtered_df["Tuesday"].isin(tue_filter))
    ]

    if other_side != "All":
        filtered_df = filtered_df[filtered_df["Other Side Taken"] == other_side]

    st.success(f"✅ {len(filtered_df)} rows matched the filters.")
    st.dataframe(filtered_df.drop(columns=["Month_Num", "Quarter"]), use_container_width=True)
else:
    st.dataframe(df.drop(columns=["Month_Num", "Quarter"]), use_container_width=True)
