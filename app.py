import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta

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
    # Add week number for calendar view
    df["Week_Num"] = df["Date"].dt.isocalendar().week
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
        
        # Display data and calendar view
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Filtered Data")
            st.dataframe(filtered_df.drop(columns=["Month_Num", "Quarter", "Week_Num"]), use_container_width=True)
        
        with col2:
            st.subheader("Calendar View")
            create_calendar_view(filtered_df)
    else:
        st.dataframe(df.drop(columns=["Month_Num", "Quarter", "Week_Num"]), use_container_width=True)

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
    
    # Function to create calendar view
    def create_calendar_view(data_df, condition1_df=None, condition2_df=None):
        if data_df.empty:
            st.info("No data to display in calendar view")
            return
        
        # Get unique years in the filtered data
        years = sorted(data_df["Year"].unique())
        
        for year in years:
            st.write(f"### {year}")
            
            # Get data for this year
            year_data = data_df[data_df["Year"] == year]
            
            # Create a 4x3 grid for months
            for row in range(4):
                cols = st.columns(3)
                for col in range(3):
                    month_num = row * 3 + col + 1
                    month_name = calendar.month_name[month_num]
                    
                    with cols[col]:
                        st.write(f"**{month_name}**")
                        
                        # Get data for this month
                        month_data = year_data[year_data["Month_Num"] == month_num]
                        
                        # Create calendar grid
                        weeks_in_month = set()
                        if not month_data.empty:
                            weeks_in_month = set(month_data["Week_Num"].unique())
                        
                        # Get all possible weeks in this month
                        first_day = datetime(year, month_num, 1)
                        last_day = datetime(year, month_num, calendar.monthrange(year, month_num)[1])
                        
                        start_week = first_day.isocalendar()[1]
                        end_week = last_day.isocalendar()[1]
                        
                        # Handle year boundary cases
                        if end_week < start_week:  # December-January transition
                            possible_weeks = list(range(start_week, 54)) + list(range(1, end_week + 1))
                        else:
                            possible_weeks = list(range(start_week, end_week + 1))
                        
                        # Create week indicators
                        week_cols = st.columns(len(possible_weeks))
                        
                        for i, week_num in enumerate(possible_weeks):
                            with week_cols[i]:
                                # Determine color based on presence in datasets
                                if condition1_df is not None and condition2_df is not None:
                                    # For comparison mode
                                    in_cond1 = False
                                    in_cond2 = False
                                    
                                    cond1_year_data = condition1_df[condition1_df["Year"] == year]
                                    cond1_month_data = cond1_year_data[cond1_year_data["Month_Num"] == month_num]
                                    if not cond1_month_data.empty and week_num in set(cond1_month_data["Week_Num"].unique()):
                                        in_cond1 = True
                                    
                                    cond2_year_data = condition2_df[condition2_df["Year"] == year]
                                    cond2_month_data = cond2_year_data[cond2_year_data["Month_Num"] == month_num]
                                    if not cond2_month_data.empty and week_num in set(cond2_month_data["Week_Num"].unique()):
                                        in_cond2 = True
                                    
                                    if in_cond1 and in_cond2:
                                        box_color = "green"  # Both conditions
                                        tooltip = "Both conditions"
                                    elif in_cond1:
                                        box_color = "blue"  # Only condition 1
                                        tooltip = "Condition 1 only"
                                    elif in_cond2:
                                        box_color = "orange"  # Only condition 2
                                        tooltip = "Condition 2 only"
                                    else:
                                        box_color = "lightgray"  # Neither condition
                                        tooltip = "No match"
                                else:
                                    # For single filter mode
                                    if week_num in weeks_in_month:
                                        box_color = "green"
                                        tooltip = "Match"
                                    else:
                                        box_color = "lightgray"
                                        tooltip = "No match"
                                
                                # Display week box with color
                                st.markdown(
                                    f"""
                                    <div title="{tooltip}" style="
                                        background-color: {box_color};
                                        color: white;
                                        text-align: center;
                                        padding: 5px 0;
                                        border-radius: 5px;
                                        font-size: 0.8em;
                                    ">{week_num}</div>
                                    """, 
                                    unsafe_allow_html=True
                                )
    
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
        tab_results1, tab_results2, tab_common, tab_diff1, tab_diff2, tab_calendar = st.tabs([
            f"Condition 1 ({len(df1)})", 
            f"Condition 2 ({len(df2)})", 
            f"Common ({len(common_rows)})",
            f"Only in Condition 1 ({len(only_in_1)})",
            f"Only in Condition 2 ({len(only_in_2)})",
            "Calendar View"
        ])
        
        with tab_results1:
            st.dataframe(df1.drop(columns=["Month_Num", "Quarter", "Week_Num"]), use_container_width=True)
            
        with tab_results2:
            st.dataframe(df2.drop(columns=["Month_Num", "Quarter", "Week_Num"]), use_container_width=True)
            
        with tab_common:
            st.dataframe(common_rows.drop(columns=["Month_Num", "Quarter", "Week_Num"]), use_container_width=True)
            
        with tab_diff1:
            st.dataframe(only_in_1.drop(columns=["Month_Num", "Quarter", "Week_Num"]), use_container_width=True)
            
        with tab_diff2:
            st.dataframe(only_in_2.drop(columns=["Month_Num", "Quarter", "Week_Num"]), use_container_width=True)
            
        with tab_calendar:
            st.write("""
            ### Calendar View Legend
            - **Green** boxes: Weeks present in both conditions
            - **Blue** boxes: Weeks only in Condition 1
            - **Orange** boxes: Weeks only in Condition 2
            - **Light Gray** boxes: Weeks in the time range but not matching either condition
            """)
            
            # Use the combined dataframe for the calendar view
            combined_df = pd.concat([df1, df2]).drop_duplicates()
            create_calendar_view(combined_df, df1, df2)
