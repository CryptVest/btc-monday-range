import streamlit as st
import pandas as pd
import numpy as np
import traceback # For better error reporting

st.set_page_config(page_title="BTC Monday Filter", layout="wide")

# --- Configuration ---
# Explicitly list columns to EXCLUDE from filtering UI
# Make SURE the names here EXACTLY match your CSV header if they exist
EXCLUDED_FILTER_COLS = ['Touches', 'Month_Num', 'Quarter', 'Date'] # Date has its own widget

# Explicitly list columns known to be NUMERIC where you want a RANGE SLIDER filter
# Add the EXACT names of numeric columns you want sliders for.
# Example: NUMERIC_COLS_FOR_SLIDER_FILTER = ['Monday Size', 'High', 'Low']
NUMERIC_COLS_FOR_SLIDER_FILTER = ['Monday Size']

# Explicitly list columns with a limited, known set of CATEGORICAL values for Selectbox/Radio
# Use EXACT column names. Example: {'Some Col': ['A', 'B', 'C']}
KNOWN_LIMITED_CATEGORICAL = {'Other Side Taken': ['Yes', 'No']} # Assuming these are the only values + 'All'

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("btc_data.csv", delimiter=";")
    except FileNotFoundError:
        st.error("❌ 'btc_data.csv' not found. Please make sure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        st.stop()

    # --- Basic Date Parsing (Essential) ---
    if "Date" not in df.columns:
        st.error("❌ Critical Error: 'Date' column not found. Cannot proceed.")
        st.stop()
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%B %d, %Y")
    except Exception as e:
        st.error(f"❌ Error parsing 'Date' column: {e}. Expected format 'Month Day, Year'. Check CSV.")
        st.stop()

    # --- Derive Helper Columns ---
    try:
        if "Month" in df.columns:
            df["Month_Num"] = pd.to_datetime(df["Month"], format="%B").dt.month
            df["Quarter"] = ((df["Month_Num"] - 1) // 3 + 1)
        else:
            # st.warning("⚠️ 'Month' column not found. Cannot derive Month Number or Quarter.")
            df["Month_Num"] = np.nan
            df["Quarter"] = np.nan

        if "Year" not in df.columns and 'Date' in df.columns:
             # st.warning("⚠️ 'Year' column not found. Deriving from Date.")
             df['Year'] = df['Date'].dt.year # Attempt to derive from Date if missing
        elif "Year" in df.columns:
             df["Year"] = pd.to_numeric(df["Year"], errors='coerce').astype('Int64')
        else:
             df["Year"] = np.nan


    except Exception as e:
         st.warning(f"⚠️ Error deriving helper columns (Month_Num, Quarter, Year): {e}")

    # --- Parse/Clean OTHER Columns Dynamically ---
    # Ensure specified numeric cols are numeric, others default to string/object
    for col in df.columns:
        if col in ['Date', 'Month_Num', 'Quarter', 'Year']: # Already handled or derived
            continue

        # Attempt numeric conversion ONLY for columns intended for sliders
        if col in NUMERIC_COLS_FOR_SLIDER_FILTER:
            try:
                original_dtype = df[col].dtype
                # Handle percentage strings specifically if needed (adapt if other formats exist)
                if col == 'Monday Size' and pd.api.types.is_string_dtype(original_dtype):
                     df[col] = df[col].astype(str).str.rstrip('%').str.replace(',', '.').astype(float)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

                if df[col].isnull().sum() > 0 and not pd.api.types.is_numeric_dtype(original_dtype):
                     pass # Silently allow NaNs from coercion for sliders
                     # st.warning(f"⚠️ Column '{col}': Some values could not be converted to numeric for slider.")
            except Exception as e:
                st.warning(f"⚠️ Could not convert column '{col}' to numeric for slider: {e}. Treating as text.")
                df[col] = df[col].astype(str).fillna('NaN') # Fallback to string

        # Convert all OTHER columns (that are not date/helpers) to string for consistent filtering
        elif col not in EXCLUDED_FILTER_COLS:
             # Includes columns like 'High Touches', 'Low Touches' etc. unless they are in NUMERIC_COLS_FOR_SLIDER_FILTER
             df[col] = df[col].astype(str).fillna('NaN') # Convert to string, handle potential NaNs

    # Sort by Date for clarity
    df = df.sort_values(by="Date").reset_index(drop=True)
    return df

# --- Main App ---
try:
    df = load_data()

    # --- DEBUG: Show detected columns and types ---
    # This helps verify if column names are correct and how pandas interpreted them
    with st.expander("Debug Info: Detected Columns and Data Types"):
        st.write("**Detected Columns:**")
        st.write(df.columns.tolist())
        st.write("**Detected Data Types (after cleaning):**")
        st.dataframe(df.dtypes.astype(str), use_container_width=True)
        st.write(f"**Columns Excluded from Filters:** {EXCLUDED_FILTER_COLS}")
        st.write(f"**Columns Configured for Slider Filters:** {NUMERIC_COLS_FOR_SLIDER_FILTER}")
        st.write(f"**Columns Configured for Selectbox Filters:** {list(KNOWN_LIMITED_CATEGORICAL.keys())}")
        st.write(f"*All other non-excluded columns should get a Multi-Select filter.*")

    # Define columns available for filtering dynamically AFTER cleaning and exclusion
    ALL_AVAILABLE_COLUMNS_FOR_FILTERING = [
        col for col in df.columns if col not in EXCLUDED_FILTER_COLS
    ]

    st.title("📊 BTC Monday Range Filter Tool")
    tab1, tab2 = st.tabs(["Single Filter", "Comparison Mode"])

    # ========================== Function Definitions (Used by both Tabs) ==========================

    # --- Function to apply filters dynamically ---
    def apply_filters(base_df, filter_params):
        filtered = base_df.copy()
        widget_actions = filter_params.get("_widget_actions", {}) # Get how each widget was created

        for col, filter_value in filter_params.items():
            if col == "_widget_actions" or filter_value is None: # Skip meta-key or uncreated filters
                continue

            widget_type = widget_actions.get(col, "unknown") # Get how this filter was made

            try:
                # --- Date Range Filter ---
                if col == "date_range":
                    if filter_value and len(filter_value) == 2:
                        start_date = pd.to_datetime(filter_value[0])
                        end_date = pd.to_datetime(filter_value[1])
                        if base_df['Date'].dt.tz is None: # Handle timezone awareness
                            start_date = start_date.tz_localize(None)
                            end_date = end_date.tz_localize(None)
                        filtered = filtered[
                            (filtered["Date"] >= start_date) &
                            (filtered["Date"] <= end_date)
                        ]

                # --- Numeric Range (Slider) Filter ---
                elif widget_type == "slider": # Check using the widget type info
                    if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                        min_f, max_f = filter_value
                        # Handle potential NaNs in the source column if needed
                        # Option 1: Include NaNs that match the range (unlikely for numeric)
                        # filtered = filtered[filtered[col].between(min_f, max_f) | filtered[col].isnull()]
                        # Option 2: Exclude NaNs (typical)
                        filtered = filtered[filtered[col].between(min_f, max_f)]
                    # else: # Ignore if filter value is not a valid range
                    #    st.warning(f"Ignoring invalid slider value for {col}: {filter_value}")


                # --- Categorical (Selectbox) Filter ---
                elif widget_type == "selectbox":
                    if isinstance(filter_value, str) and filter_value != "All":
                        # Comparing as strings, as columns were converted
                        filtered = filtered[filtered[col].astype(str) == filter_value]

                # --- Categorical (Multiselect) Filter ---
                elif widget_type == "multiselect":
                    if isinstance(filter_value, list):
                        # Get all unique string options from the base df for this column
                        # Important: Use .astype(str) consistent with widget creation
                        all_options = sorted(base_df[col].astype(str).unique())
                        # Only filter if the selection is different from selecting everything
                        if set(filter_value) != set(all_options):
                            # Compare as strings
                            filtered = filtered[filtered[col].astype(str).isin(filter_value)]
                # else: # Handle unknown widget types or columns skipped during UI creation if needed
                     # st.warning(f"Skipping filter for {col}, unknown widget type '{widget_type}'")

            except Exception as e:
                st.error(f"Error applying filter for column '{col}' (Widget: {widget_type}, Value: {filter_value}): {e}")
                # Optionally continue or stop
                # st.stop()

        return filtered


    # --- Define function to create filter UI dynamically ---
    def create_filter_ui(container, label, key_suffix, base_df):
        filters = {}
        widget_actions = {} # Store how each widget was created

        with container:
            st.subheader(f"{label} Filters")

            # --- Date Range (Special Case) ---
            min_date_dt, max_date_dt = base_df["Date"].min(), base_df["Date"].max()
            min_date, max_date = min_date_dt.date(), max_date_dt.date()
            filters["date_range"] = st.date_input(f"Date Range {label}", [min_date, max_date],
                                                min_value=min_date, max_value=max_date,
                                                key=f"date_{key_suffix}")
            widget_actions["date_range"] = "date_input" # Record widget type

            # --- Dynamic Filters for other columns ---
            for col in ALL_AVAILABLE_COLUMNS_FOR_FILTERING:
                widget_key = f"{col.lower().replace(' ', '_').replace('.', '_')}_{key_suffix}" # Unique key, handle more chars
                widget_created = False
                try:
                    # --- 1. Numeric Columns for Slider ---
                    if col in NUMERIC_COLS_FOR_SLIDER_FILTER:
                        if pd.api.types.is_numeric_dtype(base_df[col]):
                            min_val = float(base_df[col].min())
                            max_val = float(base_df[col].max())
                            if pd.isna(min_val) or pd.isna(max_val):
                                st.caption(f"{col}: Skipped slider (contains NaN)")
                                filters[col] = None
                            elif min_val == max_val: # Handle single value case
                                # Option A: Slider with small range
                                max_val += 0.1
                                filters[col] = st.slider(f"{col} Range", min_val, max_val, (min_val, max_val), key=widget_key)
                                widget_actions[col] = "slider"
                                widget_created = True
                                # Option B: Selectbox/Text input for single value? (More complex)
                            else:
                                filters[col] = st.slider(f"{col} Range", min_val, max_val, (min_val, max_val), key=widget_key)
                                widget_actions[col] = "slider"
                                widget_created = True
                        else:
                            st.caption(f"{col}: Skipped slider (not numeric: {base_df[col].dtype})")
                            filters[col] = None

                    # --- 2. Specific Categorical -> Selectbox ---
                    elif col in KNOWN_LIMITED_CATEGORICAL:
                         options = ["All"] + sorted([str(v) for v in KNOWN_LIMITED_CATEGORICAL[col]])
                         filters[col] = st.selectbox(f"{col}", options, key=widget_key)
                         widget_actions[col] = "selectbox"
                         widget_created = True

                    # --- 3. Default Fallback: Multiselect for ALL OTHERS ---
                    # Includes text columns, numeric cols NOT designated for sliders, etc.
                    else:
                        # Convert column to string for unique values and options
                        unique_vals_str = sorted(base_df[col].astype(str).unique())

                        if not unique_vals_str: # Handle empty column case
                            st.caption(f"{col}: Skipped filter (no values found)")
                            filters[col] = None
                        # Limit options if too many unique values for performance/UI
                        elif len(unique_vals_str) > 150: # Increased limit slightly
                            st.caption(f"{col}: Skipped filter (too many unique values: {len(unique_vals_str)})")
                            filters[col] = None # Indicate no filter applied
                        else:
                            # Create multiselect using stringified unique values
                            filters[col] = st.multiselect(f"{col}", unique_vals_str, default=unique_vals_str, key=widget_key)
                            widget_actions[col] = "multiselect"
                            widget_created = True

                except Exception as e:
                    st.warning(f"⚠️ Error creating filter widget for {col}: {e}")
                    filters[col] = None # Skip filter on error

                if not widget_created and col not in filters:
                     # Optional: Log columns that were skipped unintentionally
                     # print(f"Debug: Column '{col}' was not assigned a widget.")
                     filters[col] = None # Ensure every column has an entry in filters dict

            filters["_widget_actions"] = widget_actions # Add the widget type info to the params
            return filters

    # ========================== Tab 1: Single Filter ==========================
    with tab1:
        with st.sidebar:
            st.header("🔍 Filter Options (Single)")
            # Use the dynamic UI generator function
            filter_params_single = create_filter_ui(st.sidebar, "Single", "single", df)
            run_filter_single = st.button("🔎 Search", key="search_single")

        # --- Filter logic for Tab 1 ---
        st.subheader("Filter Results")
        if not run_filter_single:
            st.dataframe(df.drop(columns=["Month_Num", "Quarter"], errors='ignore'), use_container_width=True)
            st.info("Adjust filters in the sidebar and click 'Search' to refine results.")
        else:
            # Use the generic apply_filters function
            filtered_df_single = apply_filters(df, filter_params_single)

            if not filtered_df_single.empty:
                st.success(f"✅ {len(filtered_df_single)} rows matched the filters.")
                st.dataframe(filtered_df_single.drop(columns=["Month_Num", "Quarter"], errors='ignore'), use_container_width=True)
            else:
                st.warning("⚠️ No rows matched the selected filters.")

    # ========================== Tab 2: Comparison Mode ==========================
    with tab2:
        st.header("📊 Comparison Tool")
        st.write("Set up two different filter conditions using available columns and compare the resulting datasets.")

        col1, col2 = st.columns(2)

        # --- Create filter UIs for both conditions ---
        condition1_params = create_filter_ui(col1, "Condition 1", "c1", df)
        condition2_params = create_filter_ui(col2, "Condition 2", "c2", df)

        # --- Run Comparison ---
        if st.button("🔍 Compare Conditions", key="compare_button"):
            # Apply filters using the dynamic function
            df1 = apply_filters(df, condition1_params)
            df2 = apply_filters(df, condition2_params)

            # --- Row Comparison Logic (using Date as key primarily) ---
            merge_col = 'Date'
            id_cols_for_merge = ['Date'] # Define the columns that uniquely identify a row for comparison
            # If Date is not unique, you might need more columns:
            # if not df['Date'].is_unique:
            #    id_cols_for_merge = ['Date', 'SomeOtherColumn'] # Adjust as needed
            #    st.warning(f"⚠️ 'Date' column is not unique. Comparing rows based on {id_cols_for_merge}.")

            try:
                 # Create unique IDs based on the chosen columns for comparison
                df1['_merge_id'] = df1[id_cols_for_merge].astype(str).agg('_'.join, axis=1)
                df2['_merge_id'] = df2[id_cols_for_merge].astype(str).agg('_'.join, axis=1)

                # Find common rows based on this unique ID
                common_ids = pd.merge(df1[['_merge_id']], df2[['_merge_id']], how='inner', on='_merge_id')
                common_rows_display = df1[df1['_merge_id'].isin(common_ids['_merge_id'])]

                only_in_1 = df1[~df1['_merge_id'].isin(common_ids['_merge_id'])]
                only_in_2 = df2[~df2['_merge_id'].isin(common_ids['_merge_id'])]

                # Calculate Similarity (Jaccard Index on rows based on the key)
                union_count = len(df1) + len(df2) - len(common_ids)
                similarity_percent = (len(common_ids) / union_count * 100) if union_count > 0 else 0

                # Clean up temporary merge ID
                df1 = df1.drop(columns=['_merge_id'], errors='ignore')
                df2 = df2.drop(columns=['_merge_id'], errors='ignore')
                common_rows_display = common_rows_display.drop(columns=['_merge_id'], errors='ignore')
                only_in_1 = only_in_1.drop(columns=['_merge_id'], errors='ignore')
                only_in_2 = only_in_2.drop(columns=['_merge_id'], errors='ignore')


            except KeyError as e:
                 st.error(f"Merge Error: Column '{e}' not found in {id_cols_for_merge}. Check `id_cols_for_merge` list and CSV.")
                 st.stop()
            except Exception as e:
                 st.error(f"An error occurred during row comparison: {e}")
                 traceback.print_exc() # Print detailed traceback to console/log
                 st.stop()

            # --- Show Row Comparison Metrics ---
            st.subheader("Row Comparison Results")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Condition 1 Total Rows", len(df1))
            metric_col2.metric("Condition 2 Total Rows", len(df2))
            metric_col3.metric(f"Common Rows (by {', '.join(id_cols_for_merge)})", len(common_ids))
            metric_col4.metric("Similarity (Jaccard Index)", f"{similarity_percent:.1f}%")

            # --- Summary Statistics Comparison (Adjust as needed) ---
            st.subheader("Summary Statistics Comparison")
            # Dynamically find numeric/categorical columns in the *filtered* results
            numeric_cols_df1 = [col for col in df1.columns if pd.api.types.is_numeric_dtype(df1[col]) and col not in ['Month_Num', 'Quarter', 'index']]
            cat_cols_df1 = [col for col in df1.columns if col not in numeric_cols_df1 and col not in EXCLUDED_FILTER_COLS + ['Month', 'Year', 'index']] # Adjust exclusion

            numeric_cols_df2 = [col for col in df2.columns if pd.api.types.is_numeric_dtype(df2[col]) and col not in ['Month_Num', 'Quarter', 'index']]
            cat_cols_df2 = [col for col in df2.columns if col not in numeric_cols_df2 and col not in EXCLUDED_FILTER_COLS + ['Month', 'Year', 'index']]

            summary_col1, summary_col2 = st.columns(2)

            # --- Helper Function to Display Summaries ---
            def display_summary(container, data, label, numeric_cols, cat_cols):
                 with container:
                    st.markdown(f"**{label} Summaries**")
                    if not data.empty:
                        # Numeric Summary
                        st.write("Numeric:")
                        numeric_cols_exist = [c for c in numeric_cols if c in data.columns]
                        if numeric_cols_exist:
                            st.dataframe(data[numeric_cols_exist].describe().T.round(2))
                        else:
                            st.caption("No numeric columns found for summary.")

                        # Categorical Summary
                        st.write("Categorical Frequencies (%):")
                        cat_cols_exist = [c for c in cat_cols if c in data.columns]
                        if cat_cols_exist:
                            cat_summary_dict = {}
                            for col in cat_cols_exist:
                                try:
                                    counts = data[col].value_counts(normalize=True, dropna=False) * 100
                                    if len(counts) > 15: # Show top N + Other if too many
                                        top_n = counts.head(10)
                                        other_perc = counts.iloc[10:].sum()
                                        final_counts = top_n.to_dict()
                                        if other_perc > 0.01: # Only show Other if it's significant
                                            final_counts['Other'] = other_perc
                                        counts_display = pd.Series(final_counts)
                                    else:
                                        counts_display = counts
                                    cat_summary_dict[col] = counts_display.round(1).astype(str) + '%'
                                except Exception as e:
                                     st.warning(f"Could not summarize {col}: {e}")

                            # Convert dict of Series to DataFrame - handle potential length mismatches
                            try:
                                df_summary = pd.DataFrame(dict([(k,v.to_dict()) for k,v in cat_summary_dict.items()])).fillna('-')
                                st.dataframe(df_summary)
                            except Exception as e:
                                 st.warning(f"Could not display combined categorical summary for {label}: {e}. Showing individually:")
                                 for k, v in cat_summary_dict.items():
                                     st.write(f"**{k}**")
                                     st.dataframe(v)
                        else:
                            st.caption("No categorical columns found for summary.")
                    else:
                        st.write(f"No data for {label}.")

            display_summary(summary_col1, df1, "Condition 1", numeric_cols_df1, cat_cols_df1)
            display_summary(summary_col2, df2, "Condition 2", numeric_cols_df2, cat_cols_df2)

            st.divider()

            # --- Display Detailed Row Results in Tabs ---
            st.subheader("Detailed Row Data")
            tab_r1, tab_r2, tab_c, tab_d1, tab_d2 = st.tabs([
                f"Cond 1 ({len(df1)})", f"Cond 2 ({len(df2)})", f"Common ({len(common_rows_display)})",
                f"Only C1 ({len(only_in_1)})", f"Only C2 ({len(only_in_2)})"
            ])

            def display_df_tab(tab, data):
                 with tab:
                    cols_to_drop = ["Month_Num", "Quarter", "index", "_merge_id"] # Drop helpers
                    st.dataframe(data.drop(columns=cols_to_drop, errors='ignore'), use_container_width=True)

            display_df_tab(tab_r1, df1)
            display_df_tab(tab_r2, df2)
            display_df_tab(tab_c, common_rows_display)
            display_df_tab(tab_d1, only_in_1)
            display_df_tab(tab_d2, only_in_2)

        else:
             st.info("Adjust filters in the columns above and click 'Compare Conditions' to see results.")


except FileNotFoundError:
    # Error handled in load_data
    pass
except Exception as e:
    st.error("An unexpected error occurred in the main application.")
    st.exception(e) # Display detailed traceback for debugging
