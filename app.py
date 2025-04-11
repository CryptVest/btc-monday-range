import streamlit as st
import pandas as pd
import numpy as np

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

# Add tabs for single filter vs comparison mode
tab1, tab2 = st.tabs(["Single Filter", "Comparison Mode"])

with tab1:
    # Sidebar Filters for Single Mode
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

with tab2:
    st.header("Comparison Tool")
    st.write("Set up two different filter conditions and compare the results")
    
    # Create two columns for the filters
    col1, col2 = st.columns(2)
    
    # Define function to create filter UI and apply filters
    def create_filter_ui(container, label):
        with container:
            st.subheader(f"{label} Filter")
            
            # Date range
            min_date, max_date = df["Date"].min(), df["Date"].max()
            date_range = st.date_input(f"Date Range {label}", [min_date, max_date])
            
            # Monday Range
            min_range = float(df["Monday Size"].min())
            max_range = float(df["Monday Size"].max())
            monday_range = st.slider(f"Monday Range (%) {label}", 
                                     min_value=min_range, 
                                     max_value=max_range, 
                                     value=(min_range, max_range))
            
            # Other Side Taken
            other_side = st.selectbox(f"Other Side Taken {label}", ["All", "Yes", "No"])
            
            # Year & Quarter
            selected_years = st.multiselect(f"Select Year(s) {label}", 
                                          sorted(df["Year"].unique()), 
                                          default=df["Year"].unique())
            
            selected_quarters = st.multiselect(f"Select Quarter(s) {label}", 
                                             [1, 2, 3, 4], 
                                             default=[1, 2, 3, 4])
            
            # Weekly direction
            week_filter = st.multiselect(f"Weekly {label}", 
                                       df["Weekly"].unique(), 
                                       default=df["Weekly"].unique())
            
            mon_filter = st.multiselect(f"Monday {label}", 
                                      df["Monday"].unique(), 
                                      default=df["Monday"].unique())
            
            tue_filter = st.multiselect(f"Tuesday {label}", 
                                      df["Tuesday"].unique(), 
                                      default=df["Tuesday"].unique())
            
            return {
                "date_range": date_range,
                "monday_range": monday_range,
                "other_side": other_side,
                "selected_years": selected_years,
                "selected_quarters": selected_quarters,
                "week_filter": week_filter,
                "mon_filter": mon_filter,
                "tue_filter": tue_filter
            }
    
    # Create filters for each condition
    condition1 = create_filter_ui(col1, "Condition 1")
    condition2 = create_filter_ui(col2, "Condition 2")
    
    # Function to apply filters
    def apply_filters(filter_params):
        filtered = df.copy()
        filtered = filtered[
            (filtered["Date"] >= pd.to_datetime(filter_params["date_range"][0])) &
            (filtered["Date"] <= pd.to_datetime(filter_params["date_range"][1])) &
            (filtered["Monday Size"] >= filter_params["monday_range"][0]) &
            (filtered["Monday Size"] <= filter_params["monday_range"][1]) &
            (filtered["Year"].isin(filter_params["selected_years"])) &
            (filtered["Quarter"].isin(filter_params["selected_quarters"])) &
            (filtered["Weekly"].isin(filter_params["week_filter"])) &
            (filtered["Monday"].isin(filter_params["mon_filter"])) &
            (filtered["Tuesday"].isin(filter_params["tue_filter"]))
        ]
        
        if filter_params["other_side"] != "All":
            filtered = filtered[filtered["Other Side Taken"] == filter_params["other_side"]]
            
        return filtered
    
    # Run comparison
    if st.button("🔍 Compare Conditions"):
        # Apply filters
        df1 = apply_filters(condition1)
        df2 = apply_filters(condition2)
        
        # Calculate similarity metrics
        common_rows = pd.merge(df1, df2, how='inner')
        only_in_1 = df1[~df1['Date'].isin(df2['Date'])]
        only_in_2 = df2[~df2['Date'].isin(df1['Date'])]
        
        # Show similarity metrics
        st.subheader("Comparison Results")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Condition 1 Total", len(df1))
        with metric_col2:
            st.metric("Condition 2 Total", len(df2))
        with metric_col3:
            st.metric("Common Rows", len(common_rows))
        with metric_col4:
            if len(df1) > 0 and len(df2) > 0:
                similarity_percent = (len(common_rows) / max(len(df1), len(df2))) * 100
                st.metric("Similarity %", f"{similarity_percent:.1f}%")
            else:
                st.metric("Similarity %", "0%")
        
        # Display results
        tab_results1, tab_results2, tab_common, tab_diff1, tab_diff2 = st.tabs([
            f"Condition 1 ({len(df1)})", 
            f"Condition 2 ({len(df2)})", 
            f"Common ({len(common_rows)})",
            f"Only in Condition 1 ({len(only_in_1)})",
            f"Only in Condition 2 ({len(only_in_2)})"
        ])
        
        with tab_results1:
            st.dataframe(df1.drop(columns=["Month_Num", "Quarter"]), use_container_width=True)
            
        with tab_results2:
            st.dataframe(df2.drop(columns=["Month_Num", "Quarter"]), use_container_width=True)
            
        with tab_common:
            st.dataframe(common_rows.drop(columns=["Month_Num", "Quarter"]), use_container_width=True)
            
        with tab_diff1:
            st.dataframe(only_in_1.drop(columns=["Month_Num", "Quarter"]), use_container_width=True)
            
        with tab_diff2:
            st.dataframe(only_in_2.drop(columns=["Month_Num", "Quarter"]), use_container_width=True)
