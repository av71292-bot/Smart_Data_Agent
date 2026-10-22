import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai



st.set_page_config(page_title="Smart Data Clean & Query Agent", layout="wide")

st.title("🤖 Smart Data Clean & Query Agent")
st.write("Upload your Excel or CSV file to check data types, clean missing values, ask questions, and download cleaned data!")

# --- Sidebar Section ---

st.sidebar.markdown("### ℹ️ About This App")
st.sidebar.info(
    "This *Smart Data Clean & Query Agent* automatically checks and fixes data types, "
    "handles and fills missing values in your dataset, and allows you to query your data seamlessly using Google Gemini AI."
)

st.sidebar.markdown("---")

api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

st.sidebar.markdown("---")

st.sidebar.markdown(
    "<p style='text-align: center; color: #B0BEC5; font-size: 18px;'>Developed by <br><b>Ankita Vaghela</b></p>", 
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file:
    # Reset session state on new file upload
    if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
        st.session_state["last_uploaded_file"] = uploaded_file.name
        if 'df_cleaned' in st.session_state:
            del st.session_state['df_cleaned']

    # Read uploaded file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, na_values=['None', 'none', 'null', 'NULL', 'NaN', ''])
    else:
        df = pd.read_excel(uploaded_file, na_values=['None', 'none', 'null', 'NULL', 'NaN', ''])

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # --- DATA HEALTH & DATA TYPE CHECK ---
    total_missing = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())

    st.markdown("### 🔍 Dataset Health & Data Types Summary")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Missing Values", total_missing)
    col2.metric("Duplicate Rows", total_duplicates)

    # Show Data Types Table
    dtype_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": df.isnull().sum().values
    })
    
    with st.expander("📋 View Column Data Types & Details", expanded=False):
        st.dataframe(dtype_df, use_container_width=True)

    if total_missing == 0 and total_duplicates == 0:
        st.success("✅ This dataset is completely clean! 0 missing values and 0 duplicate rows found.")
    else:
        st.warning(f"⚠️ This dataset has {total_missing} missing values and {total_duplicates} duplicate rows.")

    # --- AUTO DATA CLEANING & TYPE CONVERSION ---
    if st.button("🧹 Clean Data & Fix Data Types Automatically"):
        df_cleaned = df.drop_duplicates().copy()
        
        # 1. Automatic Data Type Conversion (Dates & Numbers)
        for col in df_cleaned.columns:
            # Try converting to datetime if column name sounds like a date or has date-like text
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='ignore')
                except Exception:
                    pass
            
            # Try converting numeric text to numeric numbers
            if df_cleaned[col].dtype == 'object':
                numeric_converted = pd.to_numeric(df_cleaned[col], errors='coerce')
                # If more than 50% values converted to numeric, convert column
                if numeric_converted.notnull().sum() > 0.5 * len(df_cleaned):
                    df_cleaned[col] = numeric_converted

        # 2. Fill numeric columns with Mean
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
        
        # 3. Fill text/object columns with 'Unknown'
        text_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in text_cols:
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")
            
        st.session_state['df_cleaned'] = df_cleaned
        st.success("Data Cleaned & Data Types Corrected Successfully!")
        st.rerun()

    working_df = st.session_state.get('df_cleaned', df)

    if 'df_cleaned' in st.session_state:
        st.subheader("✨ Cleaned Dataset Preview")
        st.dataframe(working_df.head())

    # --- DOWNLOAD SECTION ---
    st.subheader("📥 Download Cleaned Data")
    csv_data = working_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Dataset (CSV)",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    # --- GEMINI AI QUERY SECTION ---
    st.subheader("💬 Ask Questions About Your Data")
    user_query = st.text_input("Ask a question (e.g., Which columns are numeric or what are the data types?):")

    if user_query:
        if not api_key:
            st.error("⚠️ Please enter your Gemini API Key in the sidebar to ask questions!")
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            summary_info = f"""
            Full Dataset Metadata for Current File ({uploaded_file.name}):
            - Total Rows: {len(df)}
            - Total Columns: {list(df.columns)}
            - Column Data Types: {df.dtypes.to_dict()}
            - Missing Values Per Column: {df.isnull().sum().to_dict()}
            - Total Missing Values: {total_missing}
            - Total Duplicate Rows: {total_duplicates}
            - First 5 Rows Sample:
            {df.head().to_string()}
            """
            
            prompt = f"{summary_info}\n\nUser Question: {user_query}\nProvide an accurate answer based on the current uploaded dataset metadata provided."
            
            with st.spinner("Analyzing with AI..."):
                try:
                    response = model.generate_content(prompt)
                    st.write("### AI Response:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")