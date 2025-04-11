import streamlit as st
import pandas as pd
import numpy as np
import calendar
import datetime
from datetime import timedelta

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
    # Add week number
    df["Week_Number"] = df["Date"].dt.isocalendar().week
    return df

# Function to display calendar view - defined at the module level
def display_calendar_view(data_df, condition_name=None):
    if len(data_df) == 0:
        st.write("No data to display in calendar view.")
        return
    
    # Group data by year and week number
    years = sorted(data_df["Year"].unique())
    
    for year in years:
        year_data = data_df[data_df["Year"] == year]
        if len(year_data) == 0:
            continue
            
        st.write(f"### {year}")
        
        # Create a calendar-like matrix for the year
        # Each row represents a month, each column represents a week number
        
        # Get all week numbers in this year
        min_week = 1
        max_week = 53  # Max possible ISO week number
        
        # Create month labels
        months = list(range(1, 13))
        month_names = [calendar.month_name[m] for m in months]
        
        # Create a 12x53 grid for the entire year (months x weeks)
        # We'll use HTML to make it more compact and visually appealing
        
        html_calendar = f"""
        <style>
            .calendar-container {{
                display: flex;
                flex-direction: column;
                font-family: Arial, sans-serif;
                margin-bottom: 20px;
            }}
            .week-numbers {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
                border-bottom: 1px solid #ddd;
            }}
            .week-number {{
                width: 20px;
                text-align: center;
                font-size: 10px;
                color: #666;
            }}
            .month-row {{
                display: flex;
                margin-bottom: 5px;
                align-items: center;
            }}
            .month-label {{
                width: 100px;
                font-weight: bold;
                margin-right: 10px;
            }}
            .week-cells {{
                display: flex;
                flex-grow: 1;
            }}
            .week-cell {{
                width: 20px;
                height: 20px;
                margin: 0 1px;
                text-align: center;
                font-size: 11px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .week-cell.has-data {{
                background-color: #4CAF50;
                color: white;
                border-radius: 50%;
            }}
            .week-cell.condition1 {{
                background-color: #2196F3;
                color: white;
                border-radius: 50%;
            }}
            .week-cell.condition2 {{
                background-color: #FF9800;
                color: white;
                border-radius: 50%;
            }}
            .week-cell.both-conditions {{
                background-color: #9C27B0;
                color: white;
                border-radius: 50%;
            }}
            .legend {{
                display: flex;
                margin-top: 10px;
                align-items: center;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin-right: 15px;
            }}
            .legend-color {{
                width: 15px;
                height: 15px;
                border-radius: 50%;
                margin-right: 5px;
            }}
            .all-data {{
                background-color: #4CAF50;
            }}
            .cond1-color {{
                background-color: #2196F3;
            }}
            .cond2-color {{
                background-color: #FF9800;
            }}
            .both-color {{
                background-color: #9C27B0;
            }}
        </style>
        <div class="calendar-container">
            <div class="week-numbers">
                <div style="width:100px;"></div>
        """
        
        # Add week number headers (only show up to week 53)
        for week in range(min_week, max_week + 1):
            html_calendar += f'<div class="week-number">{week}</div>'
        
        html_calendar += """
            </div>
        """
        
        # For each month, create a row
        for month_num in months:
            month_name = calendar.month_name[month_num]
            month_data = year_data[year_data["Month_Num"] == month_num]
            
            html_calendar += f"""
            <div class="month-row">
                <div class="month-label">{month_name}</div>
                <div class="week-cells">
            """
            
            # For each week in the year, check if we have data for this month
            for week in range(min_week, max_week + 1):
                week_data = month_data[month_data["Week_Number"] == week]
                
                cell_class = "week-cell"
                cell_content = ""
                
                if len(week_data) > 0:
                    cell_class += " has-data"
                    # Show count of data points in this week
                    cell_content = len(week_data)
                    
                    # If we're in comparison mode with condition name
                    if condition_name:
                        cell_class = f"week-cell {condition_name}"
                
                html_calendar += f'<div class="{cell_class}">{cell_content}</div>'
            
            html_calendar += """
                </div>
            </div>
            """
        
        html_calendar += """
        </div>
        """
        
        if condition_name:
            # No legend needed for single condition view
            pass
        else:
            # Add legend for general view
            html_calendar += """
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color all-data"></div>
                    <div>Has Data</div>
                </div>
            </div>
            """
        
        st.markdown(html_calendar, unsafe_allow_html=True)

# Function to display comparison calendar view - defined at the module level
def display_comparison_calendar_view(df1, df2):
    if len(df1) == 0 and len(df2) == 0:
        st.write("No data to display in calendar view.")
        return
    
    # Combine the data from both conditions
    combined_df = pd.concat([
        df1.assign(condition="condition1"),
        df2.assign(condition="condition2")
    ])
    
    # Group data by year and week number
    years = sorted(combined_df["Year"].unique())
    
    for year in years:
        year_data = combined_df[combined_df["Year"] == year]
        if len(year_data) == 0:
            continue
            
        st.write(f"### {year}")
        
        # Create month labels
        months = list(range(1, 13))
        
        # Create a 12x53 grid for the entire year (months x weeks)
        html_calendar = f"""
        <style>
            .calendar-container {{
                display: flex;
                flex-direction: column;
                font-family: Arial, sans-serif;
                margin-bottom: 20px;
            }}
            .week-numbers {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
                border-bottom: 1px solid #ddd;
            }}
            .week-number {{
                width: 20px;
                text-align: center;
                font-size: 10px;
                color: #666;
            }}
            .month-row {{
                display: flex;
                margin-bottom: 5px;
                align-items: center;
            }}
            .month-label {{
                width: 100px;
                font-weight: bold;
                margin-right: 10px;
            }}
            .week-cells {{
                display: flex;
                flex-grow: 1;
            }}
            .week-cell {{
                width: 20px;
                height: 20px;
                margin: 0 1px;
                text-align: center;
                font-size: 11px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .week-cell.condition1 {{
                background-color: #2196F3;
                color: white;
                border-radius: 50%;
            }}
            .week-cell.condition2 {{
                background-color: #FF9800;
                color: white;
                border-radius: 50%;
            }}
            .week-cell.both-conditions {{
                background-color: #9C27B0;
                color: white;
                border-radius: 50%;
            }}
            .legend {{
                display: flex;
                margin-top: 10px;
                align-items: center;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin-right: 15px;
            }}
            .legend-color {{
                width: 15px;
                height: 15px;
                border-radius: 50%;
                margin-right: 5px;
            }}
            .cond1-color {{
                background-color: #2196F3;
            }}
            .cond2-color {{
                background-color: #FF9800;
            }}
            .both-color {{
                background-color: #9C27B0;
            }}
        </style>
        <div class="calendar-container">
            <div class="week-numbers">
                <div style="width:100px;"></div>
        """
        
        # Add week number headers (only show up to week 53)
        for week in range(1, 54):
            html_calendar += f'<div class="week-number">{week}</div>'
        
        html_calendar += """
            </div>
        """
        
        # For each month, create a row
        for month_num in months:
            month_name = calendar.month_name[month_num]
            month_data = year_data[year_data["Month_Num"] == month_num]
            
            html_calendar += f"""
            <div class="month-row">
                <div class="month-label">{month_name}</div>
                <div class="week-cells">
            """
            
            # For each week in the year, check if we have data for this month
            for week in range(1, 54):
                week_data = month_data[month_data["Week_Number"] == week]
                
                cell_class = "week-cell"
                cell_content = ""
                
                if len(week_data) > 0:
                    # Check if data is from condition1, condition2, or both
                    has_condition1 = "condition1" in week_data["condition"].values
                    has_condition2 = "condition2" in week_data["condition"].values
                    
                    if has_condition1 and has_condition2:
                        cell_class += " both-conditions"
                        # Count how many unique dates (to handle overlaps)
                        unique_dates = week_data["Date"].nunique()
                        cell_content = unique_dates
                    elif has_condition1:
                        cell_class += " condition1"
                        cell_content = len(week_data)
                    elif has_condition2:
                        cell_class += " condition2"
                        cell_content = len(week_data)
                
                html_calendar += f'<div class="{cell_class}">{cell_content}</div>'
            
            html_calendar += """
                </div>
            </div>
            """
        
        html_calendar += """
        </div>
        """
        
        # Add legend
        html_calendar += """
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color cond1-color"></div>
                <div>Condition 1</div>
            </div>
            <div class="legend-item">
                <div class="legend-color cond2-color"></div>
                <div>Condition 2</div>
            </div>
            <div class="legend-item">
                <div class="legend-color both-color"></div>
                <div>Both Conditions</div>
            </div>
        </div>
        """
        
        st.markdown(html_calendar, unsafe_allow_html=True)

# Function to apply filters - defined at the module level  
def apply_filters(filter_params, base_df):
    filtered = base_df.copy()
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

    # Filter logic for single mode
    single_filter_params = {
        "date_range": date_range,
        "monday_range": monday_range,
        "other_side": other_side,
        "selected_years": selected_years,
        "selected_quarters": selected_quarters,
        "week_filter": week_filter,
        "mon_filter": mon_filter,
        "tue_filter": tue_filter
    }
    
    filtered_df = df.copy()
    if run_filter:
        filtered_df = apply_filters(single_filter_params, df)
        st.success(f"✅ {len(filtered_df)} rows matched the filters.")
        
        # Display dataframe
        st.dataframe(filtered_df.drop(columns=["Month_Num", "Quarter"]), use_container_width=True)
        
        # Add calendar view
        st.subheader("Calendar View")
        display_calendar_view(filtered_df)
    else:
        st.dataframe(df.drop(columns=["Month_Num", "Quarter"]), use_container_width=True)
        
        # Display calendar for all data
        st.subheader("Calendar View")
        display_calendar_view(df)

with tab2:
    st.header("Comparison Tool")
    st.write("Set up two different filter conditions and compare the results")
    
    # Create two columns for the filters
    col1, col2 = st.columns(2)
    
    # Define function to create filter UI
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
    
    # Run comparison
    if st.button("🔍 Compare Conditions"):
        # Apply filters
        df1 = apply_filters(condition1, df)
        df2 = apply_filters(condition2, df)
        
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
        
        # Display calendar view
        st.subheader("Calendar View Comparison")
        display_comparison_calendar_view(df1, df2)
        
        # Display results
        st.subheader("Detailed Results")
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
