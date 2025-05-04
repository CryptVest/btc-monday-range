import streamlit as st
import pandas as pd
import numpy as np
import traceback # For better error reporting

st.set_page_config(page_title="BTC Monday Filter", layout="wide")

# --- Configuration ---
# Explicitly list columns to EXCLUDE from filtering UI
EXCLUDED_FILTER_COLS = ['Touches', 'Month_Num', 'Quarter', 'Date'] # Date has its own widget
# Explicitly list columns known to be NUMERIC for slider/range filters
NUMERIC_COLS_FOR_FILTER = ['Monday Size'] # Add other numeric columns like 'High', 'Low' if they exist and you want range filters
# Explicitly list columns with a limited, known set of categorical values for Selectbox/Radio
# Example: If 'Some Col' only has 'A', 'B', 'C'
# KNOWN_LIMITED_CATEGORICAL = {'Some Col': ['A', 'B', 'C']}
KNOWN_LIMITED_CATEGORICAL = {'Other Side Taken': ['Yes', 'No']} # Assuming these are the only values + 'All'

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("btc_data.csv", delimiter=";")
    except FileNotFoundError:
        st.error("❌ 'btc_data.csv' not found. Please make sure the file is in the same directory as the script.")
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
        st.error(f"❌ Error parsing 'Date' column: {e}. Expected format 'Month Day, Year' (e.g., 'January 01, 2023'). Please check the CSV.")
        st.stop()

    # --- Derive Helper Columns ---
    try:
        if "Month" in df.columns:
            df["Month_Num"] = pd.to_datetime(df["Month"], format="%B").dt.month
            df["Quarter"] = ((df["Month_Num"] - 1) // 3 + 1)
        else:
            st.warning("⚠️ 'Month' column not found. Cannot derive Month Number or Quarter.")
            df["Month_Num"] = np.nan
            df["Quarter"] = np.nan

        if "Year" not in df.columns:
             st.warning("⚠️ 'Year' column not found.")
             # Attempt to derive from Date if missing
             df['Year'] = df['Date'].dt.year
        else:
             df["Year"] = pd.to_numeric(df["Year"], errors='coerce').astype('Int64') # Use nullable Int

    except Exception as e:
         st.warning(f"⚠️ Error deriving helper columns (Month_Num, Quarter, Year): {e}")


    # --- Parse/Clean OTHER Columns Dynamically ---
    potential_numeric = ['Monday Size'] # Start with known numeric
    # You might need to add others manually if auto-detection fails
    # e.g., potential_numeric.extend(['High', 'Low', 'Open', 'Close'])

    for col in df.columns:
        if col in ['Date', 'Month_Num', 'Quarter']: # Already handled or derived
            continue

        # Attempt numeric conversion for specified columns
        if col in potential_numeric:
            try:
                original_dtype = df[col].dtype
                # Handle percentage strings specifically
                if col == 'Monday Size' and pd.api.types.is_string_dtype(original_dtype):
                     df[col] = df[col].astype(str).str.rstrip('%').str.replace(',', '.').astype(float)
                else:
                    # General numeric conversion, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Report if conversion failed for many rows
                if df[col].isnull().sum() > 0 and not pd.api.types.is_numeric_dtype(original_dtype):
                     st.warning(f"⚠️ Column '{col}': Some values could not be converted to numeric and are now NaN.")
            except Exception as e:
                st.warning(f"⚠️ Could not convert column '{col}' to numeric: {e}. Leaving as is.")
                # Ensure it's treated as object/string if conversion fails
                if pd.api.types.is_numeric_dtype(df[col].dtype): # Check if pandas made it numeric despite error
                    df[col] = df[col].astype(object) # Revert to object

        # Ensure other non-numeric, non-date columns are treated as strings/objects
        elif not pd.api.types.is_numeric_dtype(df[col].dtype) and not pd.api.types.is_datetime64_any_dtype(df[col].dtype):
             df[col] = df[col].astype(str).fillna('NaN') # Convert to string, handle potential NaNs from loading

    # Sort by Date for clarity
    df = df.sort_values(by="Date").reset_index(drop=True)
    return df

# --- Main App ---
try:
    df = load_data()
    # Define columns available for filtering dynamically
    ALL_AVAILABLE_COLUMNS = [col for col in df.columns if col not in EXCLUDED_FILTER_COLS]

    st.title("📊 BTC Monday Range Filter Tool")

    # Add tabs for single filter vs comparison mode
    tab1, tab2 = st.tabs(["Single Filter", "Comparison Mode"])

    # ========================== Tab 1: Single Filter ==========================
    with tab1:
        with st.sidebar:
            st.header("🔍 Filter Options (Single)")
            # Use the dynamic UI generator function here as well for consistency
            # (Or keep the manual sidebar if preferred for Tab 1)

            # --- Manual Sidebar Example (as before) ---
            min_date_dt, max_date_dt = df["Date"].min(), df["Date"].max()
            min_date, max_date = min_date_dt.date(), max_date_dt.date()
            date_range_single = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date, key="date_single")

            filter_params_single = {"date_range": date_range_single} # Start collecting params

            for col in ALL_AVAILABLE_COLUMNS:
                 # Add widgets based on column type (similar logic to create_filter_ui)
                 widget_key = f"{col.lower().replace(' ', '_')}_single"
                 try:
                    if col in NUMERIC_COLS_FOR_FILTER:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        if pd.isna(min_val) or pd.isna(max_val):
                            st.caption(f"{col}: Cannot create slider (contains NaN)")
                            filter_params_single[col] = None # Indicate no filter possible
                            continue
                        if min_val == max_val: max_val += 0.1 # Ensure range > 0
                        default_val = (min_val, max_val)
                        filter_params_single[col] = st.slider(f"{col} Range", min_val, max_val, default_val, key=widget_key)
                    # Add selectbox for specific known categoricals
                    elif col in KNOWN_LIMITED_CATEGORICAL:
                         options = ["All"] + sorted([str(v) for v in KNOWN_LIMITED_CATEGORICAL[col]])
                         filter_params_single[col] = st.selectbox(f"{col}", options, key=widget_key)
                    # Default to multiselect for other (assumed) categorical columns
                    elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or col == 'Year': # Treat Year as categorical for filtering
                        unique_vals = sorted(df[col].unique().astype(str))
                        # Limit options if too many unique values for performance/UI
                        if len(unique_vals) > 100:
                             st.caption(f"{col}: Too many unique values ({len(unique_vals)}) to display as multiselect. Filter skipped.")
                             filter_params_single[col] = None # Indicate no filter applied
                        elif len(unique_vals) == 0:
                             st.caption(f"{col}: No values found.")
                             filter_params_single[col] = None
                        else:
                             filter_params_single[col] = st.multiselect(f"{col}", unique_vals, default=unique_vals, key=widget_key)
                    # Skip columns that don't fit categories (e.g., complex objects if any)
                    else:
                         st.caption(f"Skipping filter for column: {col} (unhandled type: {df[col].dtype})")
                         filter_params_single[col] = None

                 except Exception as e:
                     st.warning(f"⚠️ Error creating filter for {col}: {e}")
                     filter_params_single[col] = None # Skip filter on error

            run_filter_single = st.button("🔎 Search", key="search_single")

        # --- Filter logic for Tab 1 ---
        st.subheader("Filter Results")
        if not run_filter_single:
            st.dataframe(df.drop(columns=["Month_Num", "Quarter"], errors='ignore'), use_container_width=True)
            st.info("Adjust filters in the sidebar and click 'Search' to refine results.")
        else:
            # Use the generic apply_filters function
            filtered_df_single = apply_filters(df, filter_params_single) # Pass original df and params

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

        # --- Define function to create filter UI dynamically ---
        def create_filter_ui(container, label, key_suffix, base_df):
            filters = {}
            with container:
                st.subheader(f"{label} Filters")

                # --- Date Range (Special Case) ---
                min_date_dt, max_date_dt = base_df["Date"].min(), base_df["Date"].max()
                min_date, max_date = min_date_dt.date(), max_date_dt.date()
                filters["date_range"] = st.date_input(f"Date Range {label}", [min_date, max_date],
                                                    min_value=min_date, max_value=max_date,
                                                    key=f"date_{key_suffix}")

                # --- Dynamic Filters for other columns ---
                for col in ALL_AVAILABLE_COLUMNS:
                    widget_key = f"{col.lower().replace(' ', '_')}_{key_suffix}" # Unique key
                    try:
                        # --- Numeric Columns -> Slider ---
                        if col in NUMERIC_COLS_FOR_FILTER:
                            if pd.api.types.is_numeric_dtype(base_df[col]):
                                min_val = float(base_df[col].min())
                                max_val = float(base_df[col].max())
                                if pd.isna(min_val) or pd.isna(max_val):
                                    st.caption(f"{col}: Cannot create slider (contains NaN)")
                                    filters[col] = None
                                    continue
                                if min_val == max_val: max_val += 0.1 # Handle edge case
                                default_val = (min_val, max_val)
                                filters[col] = st.slider(f"{col} Range", min_val, max_val, default_val, key=widget_key)
                            else:
                                st.caption(f"{col}: Expected numeric, found {base_df[col].dtype}. Filter skipped.")
                                filters[col] = None

                        # --- Specific Categorical -> Selectbox ---
                        elif col in KNOWN_LIMITED_CATEGORICAL:
                             # Ensure options are strings for consistency
                             options = ["All"] + sorted([str(v) for v in KNOWN_LIMITED_CATEGORICAL[col]])
                             filters[col] = st.selectbox(f"{col}", options, key=widget_key)

                        # --- Other String/Object/Year Columns -> Multiselect ---
                        # (Treat Year as categorical for multi-selection)
                        elif pd.api.types.is_object_dtype(base_df[col]) or pd.api.types.is_string_dtype(base_df[col]) or col == 'Year':
                            unique_vals = sorted(base_df[col].unique().astype(str))
                             # Limit options if too many unique values for performance/UI
                            if len(unique_vals) > 100:
                                st.caption(f"{col}: Too many unique values ({len(unique_vals)}) to display. Filter skipped.")
                                filters[col] = None # Indicate no filter applied
                            elif len(unique_vals) == 0:
                                st.caption(f"{col}: No values found.")
                                filters[col] = None
                            else:
                                filters[col] = st.multiselect(f"{col}", unique_vals, default=unique_vals, key=widget_key)

                        # --- Skip other types ---
                        else:
                            # Optional: Log or display skipped columns
                            # st.caption(f"Skipping filter for column: {col} (type: {base_df[col].dtype})")
                            filters[col] = None # Explicitly mark as not filtered

                    except Exception as e:
                        st.warning(f"⚠️ Error creating filter widget for {col}: {e}")
                        filters[col] = None # Skip filter on error

            return filters

        # --- Create filter UIs for both conditions ---
        condition1_params = create_filter_ui(col1, "Condition 1", "c1", df)
        condition2_params = create_filter_ui(col2, "Condition 2", "c2", df)

        # --- Function to apply filters dynamically ---
        # Takes the original DataFrame and the parameters dictionary
        def apply_filters(base_df, filter_params):
            filtered = base_df.copy()
            for col, filter_value in filter_params.items():
                if filter_value is None: # Skip if filter was not created or skipped
                    continue

                try:
                    # --- Date Range Filter ---
                    if col == "date_range":
                        if filter_value and len(filter_value) == 2:
                            start_date = pd.to_datetime(filter_value[0])
                            end_date = pd.to_datetime(filter_value[1])
                            # Ensure comparison is timezone-naive if base_df['Date'] is
                            if base_df['Date'].dt.tz is None:
                                start_date = start_date.tz_localize(None)
                                end_date = end_date.tz_localize(None)
                            filtered = filtered[
                                (filtered["Date"] >= start_date) &
                                (filtered["Date"] <= end_date)
                            ]
                    # --- Numeric Range (Slider) Filter ---
                    elif col in NUMERIC_COLS_FOR_FILTER:
                        # Check if filter_value is a tuple/list of length 2 (from slider)
                        if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                             min_f, max_f = filter_value
                             # Important: Handle NaNs in the column being filtered
                             filtered = filtered[
                                 filtered[col].between(min_f, max_f) | filtered[col].isnull()
                             ]
                             # If you want to EXCLUDE NaNs:
                             # filtered = filtered[filtered[col].between(min_f, max_f)]
                        else:
                            st.warning(f"Skipping numeric filter for '{col}'. Expected range, got {filter_value}")


                    # --- Categorical (Selectbox) Filter ---
                    elif col in KNOWN_LIMITED_CATEGORICAL:
                        if isinstance(filter_value, str) and filter_value != "All":
                            filtered = filtered[filtered[col].astype(str) == filter_value]

                    # --- Categorical (Multiselect) Filter ---
                    # Check if it's a list (multiselect returns list) AND column exists
                    elif isinstance(filter_value, list) and col in filtered.columns:
                        # Check if the default (all selected) is different from current selection
                        all_options = sorted(base_df[col].unique().astype(str))
                        if set(filter_value) != set(all_options): # Apply filter only if not all options are selected
                            # Ensure comparison is with strings if multiselect options were strings
                             filtered = filtered[filtered[col].astype(str).isin(filter_value)]


                    # Add more elif conditions here if you introduce other widget types

                except Exception as e:
                    st.error(f"Error applying filter for column '{col}' with value '{filter_value}': {e}")
                    # Optionally stop or just continue, potentially with inconsistent results
                    # st.stop()

            return filtered


        # --- Run Comparison ---
        if st.button("🔍 Compare Conditions", key="compare_button"):
            # Apply filters using the dynamic function
            df1 = apply_filters(df, condition1_params)
            df2 = apply_filters(df, condition2_params)

            # --- Row Comparison Logic (remains similar) ---
            merge_col = 'Date' # Assuming Date is the primary key for comparison
            if not df['Date'].is_unique:
                st.warning("⚠️ 'Date' column is not unique. Row comparison might behave unexpectedly. Consider adding a unique ID.")
                # Fallback to index if Date is not unique (less ideal)
                df1 = df1.reset_index()
                df2 = df2.reset_index()
                merge_col = 'index'

            try:
                # Find rows present in both based on the key column
                common_keys = pd.merge(df1[[merge_col]], df2[[merge_col]], how='inner', on=merge_col)
                common_rows_df1 = df1[df1[merge_col].isin(common_keys[merge_col])]
                # For display, common_rows can be just one set (they match on the key)
                common_rows_display = common_rows_df1

                only_in_1 = df1[~df1[merge_col].isin(common_keys[merge_col])]
                only_in_2 = df2[~df2[merge_col].isin(common_keys[merge_col])]

                # Calculate Similarity (Jaccard Index on rows based on the key)
                union_count = len(df1) + len(df2) - len(common_keys)
                similarity_percent = (len(common_keys) / union_count * 100) if union_count > 0 else 0

            except KeyError as e:
                 st.error(f"Merge Error: Column '{e}' not found for row comparison. Check data or merge_col ('{merge_col}').")
                 st.stop()
            except Exception as e:
                 st.error(f"An error occurred during row comparison: {e}")
                 st.stop()

            # --- Show Row Comparison Metrics ---
            st.subheader("Row Comparison Results")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Condition 1 Total Rows", len(df1))
            metric_col2.metric("Condition 2 Total Rows", len(df2))
            metric_col3.metric("Common Rows (by Date/Key)", len(common_keys)) # Changed label slightly
            metric_col4.metric("Similarity (Jaccard Index)", f"{similarity_percent:.1f}%")


            # --- Summary Statistics Comparison (Adjust as needed) ---
            st.subheader("Summary Statistics Comparison")

            # Define columns for summary dynamically or keep manual list
            cols_to_describe_numeric = [col for col in df1.columns if pd.api.types.is_numeric_dtype(df1[col]) and col not in ['Month_Num', 'Quarter', 'index']]
            cols_to_describe_categorical = [col for col in df1.columns if col not in cols_to_describe_numeric and col not in ['Date', 'Month_Num', 'Quarter', 'index']]


            summary_col1, summary_col2 = st.columns(2)

            def display_summary(container, data, label, numeric_cols, cat_cols):
                 with container:
                    st.markdown(f"**{label} Summaries**")
                    if not data.empty:
                        st.write("Numeric:")
                        st.dataframe(data[numeric_cols].describe().T.round(2))
                        st.write("Categorical Frequencies (%):")
                        cat_summary = {}
                        for col in cat_cols:
                            if col in data.columns:
                                counts = data[col].value_counts(normalize=True, dropna=False) * 100
                                # Show top N if too many unique values
                                if len(counts) > 15:
                                     top_n = counts.head(10)
                                     other_perc = counts.iloc[10:].sum()
                                     top_n['Other'] = other_perc
                                     counts = top_n
                                cat_summary[col] = counts.round(1).astype(str) + '%'
                        # Convert dict of Series to DataFrame, handling potential length mismatches
                        try:
                            st.dataframe(pd.DataFrame(dict([(k,v.to_dict()) for k,v in cat_summary.items()])).fillna('-'))
                        except Exception as e:
                             st.warning(f"Could not display categorical summary for {label}: {e}")
                             # Fallback: display individually
                             for k, v in cat_summary.items():
                                 st.write(f"**{k}**")
                                 st.dataframe(v)

                    else:
                        st.write(f"No data for {label}.")

            display_summary(summary_col1, df1, "Condition 1", cols_to_describe_numeric, cols_to_describe_categorical)
            display_summary(summary_col2, df2, "Condition 2", cols_to_describe_numeric, cols_to_describe_categorical)

            st.divider()

            # --- Display Detailed Row Results in Tabs ---
            st.subheader("Detailed Row Data")
            tab_results1, tab_results2, tab_common, tab_diff1, tab_diff2 = st.tabs([
                f"Condition 1 ({len(df1)})",
                f"Condition 2 ({len(df2)})",
                f"Common ({len(common_rows_display)})",
                f"Only in Condition 1 ({len(only_in_1)})",
                f"Only in Condition 2 ({len(only_in_2)})"
            ])

            def display_df(tab, data):
                 with tab:
                    cols_to_drop = ["Month_Num", "Quarter", "index"] # Add 'index' if it was added
                    st.dataframe(data.drop(columns=cols_to_drop, errors='ignore'), use_container_width=True)

            display_df(tab_results1, df1)
            display_df(tab_results2, df2)
            display_df(tab_common, common_rows_display)
            display_df(tab_diff1, only_in_1)
            display_df(tab_diff2, only_in_2)

        else:
             st.info("Adjust filters in the columns above and click 'Compare Conditions' to see results.")


except FileNotFoundError:
    # Error handled in load_data
    pass
except Exception as e:
    st.error("An unexpected error occurred in the main application.")
    st.exception(e) # Display detailed traceback for debugging
