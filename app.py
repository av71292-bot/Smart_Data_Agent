import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Data Clean & Query Agent", layout="wide")

# Colorful App Title
st.markdown(
    "<h1 style='color: #008080; text-align: center;'>💻 Smart Data Clean & Query Agent</h1>",
    unsafe_allow_html=True
)
st.write("Upload your Excel or CSV file to check data types, clean missing values, handle outliers, ask questions, and download cleaned data!")

# --- Sidebar Section ---
st.sidebar.markdown("⚙️ *About This App*")
st.sidebar.info(
    "This \"Smart Data Clean & Query Agent\" automatically checks and fixes data types, "
    "handles and fills missing values in your dataset, allows smart outlier inspection, "
    "and allows you to query your data seamlessly using Google Gemini AI."
)
st.sidebar.markdown("---")
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align: center; color: #88DECS; font-size: 25px;'>Developed by <br><b>Ankita Vaghela</b></p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file:
    # Reset session state on new file upload
    if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
        st.session_state["last_uploaded_file"] = uploaded_file.name
        if "df_cleaned" in st.session_state:
            del st.session_state["df_cleaned"]

    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, na_values=['None', 'none', 'null', 'NULL', 'NaN', ''])
    else:
        df = pd.read_excel(uploaded_file, na_values=['None', 'none', 'null', 'NULL', 'NaN', ''])

    # Colorful Subheader
    st.markdown("<h3 style='color: #FB8C00;'>📁 Dataset Preview</h3>", unsafe_allow_html=True)
    st.dataframe(df.head())

    # --- DATA HEALTH & DATA TYPE CHECK ---
    total_missing = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())

    st.markdown("<h3 style='color: #43A047;'>🔍 Dataset Health & Data Types Summary</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Total Missing Values", total_missing)
    col2.metric("Duplicate Rows", total_duplicates)

    # Show Data Types Table
    dtype_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": df.isnull().sum().values
    })

    with st.expander("📂 View Column Data Types & Details", expanded=False):
        st.dataframe(dtype_df, use_container_width=True)

    if total_missing == 0 and total_duplicates == 0:
        st.success("🎉 This dataset is completely clean! 0 missing values and 0 duplicate rows found.")
    else:
        st.warning(f"⚠️ This dataset has {total_missing} missing values and {total_duplicates} duplicate rows.")

    # --- AUTO DATA CLEANING & TYPE CONVERSION ---
    if st.button("🧹 Clean Data & Fix Data Types Automatically"):
        df_cleaned = df.drop_duplicates().copy()

        # 1. Automatic Data Type Conversion (Dates & Numbers)
        for col in df_cleaned.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='ignore')
                except Exception:
                    pass

            if df_cleaned[col].dtype == 'object':
                numeric_converted = pd.to_numeric(df_cleaned[col], errors='coerce')
                if numeric_converted.notnull().sum() > 0.5 * len(df_cleaned):
                    df_cleaned[col] = numeric_converted

        # 2. Fill numeric columns with Mean
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())

        # 3. Fill text/object columns with 'Unknown'
        text_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in text_cols:
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")

        st.session_state["df_cleaned"] = df_cleaned
        st.success("✅ Data Cleaned & Data Types Corrected Successfully!")
        st.rerun()

    working_df = st.session_state.get('df_cleaned', df)

    if 'df_cleaned' in st.session_state:
        st.markdown("<h3 style='color: #8E24AA;'>✨ Cleaned Dataset Preview</h3>", unsafe_allow_html=True)
        st.dataframe(working_df.head())

        # --- SMART OUTLIER INSPECTION & ACTION ---
        st.markdown("<h3 style='color: #FF7043;'>🔍 Smart Outlier Inspection & Action</h3>", unsafe_allow_html=True)

        if st.button("Check Outliers"):
            num_cols = working_df.select_dtypes(include=['number']).columns
            outlier_summary = {}
            
            for col in num_cols:
                Q1 = working_df[col].quantile(0.25)
                Q3 = working_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = working_df[(working_df[col] < lower_bound) | (working_df[col] > upper_bound)].shape[0]
                if outliers_count > 0:
                    outlier_summary[col] = outliers_count

            if outlier_summary:
                st.warning(f"⚠️ Outliers detected in the following columns: {outlier_summary}")
                
                # --- AI SUGGESTION INTEGRATION ---
                if api_key:
                    genai.configure(api_key=api_key)
                    ai_model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    outlier_prompt = f"""
                    Dataset name/info: Uploaded file containing numeric columns.
                    Outlier summary (Column: count of outliers): {outlier_summary}
                    
                    As an expert Data Scientist, analyze these outliers. For each column, briefly suggest whether we should:
                    1. Keep As Is (if they look like valid business/domain outliers like high salaries or asset values)
                    2. Cap / Winsorize (if they are moderate extremes)
                    3. Drop Rows (if they are clear data entry errors)
                    
                    Give a concise, professional recommendation for the user.
                    """
                    
                    with st.spinner("🤖 AI is analyzing outliers and generating suggestions..."):
                        try:
                            ai_response = ai_model.generate_content(outlier_prompt)
                            st.info(f"💡 *AI Recommendation & Insights:*\n\n{ai_response.text}")
                        except Exception as e:
                            st.error(f"Could not fetch AI suggestion: {e}")
                else:
                    st.info("💡 Tip: Review if these values are genuine or errors. Enter Gemini API key in sidebar for automated AI suggestions.")
            else:
                st.success("✅ No significant outliers detected using the IQR method!")

        outlier_action = st.selectbox(
            "Select Action for Outliers:",
            ["Select Action", "Keep As Is (Valid/Legal Outliers)", "Cap Outliers (IQR Capping)", "Drop Outlier Rows"]
        )

        if outlier_action == "Cap Outliers (IQR Capping)":
            if st.button("Apply Outlier Capping"):
                num_cols = working_df.select_dtypes(include=['number']).columns
                for col in num_cols:
                    Q1 = working_df[col].quantile(0.25)
                    Q3 = working_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    working_df[col] = np.clip(working_df[col], lower_bound, upper_bound)
                
                st.session_state["df_cleaned"] = working_df
                st.success("✅ Outliers successfully capped using IQR method!")
                st.rerun()

        elif outlier_action == "Drop Outlier Rows":
            if st.button("Apply Drop Outliers"):
                num_cols = working_df.select_dtypes(include=['number']).columns
                for col in num_cols:
                    Q1 = working_df[col].quantile(0.25)
                    Q3 = working_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    working_df = working_df[(working_df[col] >= lower_bound) & (working_df[col] <= upper_bound)]
                
                st.session_state["df_cleaned"] = working_df
                st.success("✅ Outlier rows dropped successfully!")
                st.rerun()

        # --- FOCUSED & CLEAN FEATURE ENGINEERING ---
    st.markdown("<h3 style='color: #AB47BC;'>🛠️ Smart Feature Engineering</h3>", unsafe_allow_html=True)
    st.write("Extract clean date components and meaningful numerical features to improve your model and EDA.")

    fe_col1, fe_col2 = st.columns(2)

    with fe_col1:
        if st.button("🚀 Extract Clean Date Features"):
            df_fe = working_df.copy()
            date_cols = [col for col in df_fe.columns if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower()]
            
            if date_cols:
                count = 0
                for col in date_cols:
                    try:
                        parsed_dates = pd.to_datetime(df_fe[col], errors='coerce')
                        if parsed_dates.notnull().sum() > 0:
                            df_fe[col + '_year'] = parsed_dates.dt.year
                            df_fe[col + '_month'] = parsed_dates.dt.month
                            df_fe[col + '_day'] = parsed_dates.dt.day
                            count += 3
                    except Exception:
                        pass
                st.session_state["df_cleaned"] = df_fe
                st.success(f"✅ Successfully extracted year, month, and day from date columns ({count} new features added)!")
                st.rerun()
            else:
                st.warning("⚠️ No date/time columns found in this dataset.")

    with fe_col2:
        if st.button("➕ Create Custom Average Feature (Low & High)"):
            df_fe = working_df.copy()
            num_cols = df_fe.select_dtypes(include=['number']).columns.tolist()
            
            # अगर डेटा में लो और हाई जैसे कॉलम्स हैं, तो उनका एवरेज निकालें
            low_cols = [c for c in num_cols if 'low' in c.lower() or 'min' in c.lower()]
            high_cols = [c for c in num_cols if 'high' in c.lower() or 'max' in c.lower()]
            
            if low_cols and high_cols:
                l_col = low_cols[0]
                h_col = high_cols[0]
                df_fe['range_average'] = (df_fe[l_col] + df_fe[h_col]) / 2
                st.session_state["df_cleaned"] = df_fe
                st.success(f"✅ Created 'range_average' using {l_col} and {h_col}!")
                st.rerun()
            else:
                st.info("ℹ️ Could not automatically detect explicit 'low'/'high' columns. You can use AI insights below for custom ideas.")

    # AI Guidance for Feature Engineering
    if st.button("🧠 Ask AI for Specific Feature Ideas"):
        if not api_key:
            st.error("⚠️ Please enter your Gemini API Key in the sidebar!")
        else:
            genai.configure(api_key=api_key)
            fe_model = genai.GenerativeModel('gemini-2.5-flash')
            
            columns_info = f"Columns and Types: {working_df.dtypes.to_dict()}"
            fe_prompt = f"""
            You are an expert Data Scientist. Analyze this dataset structure:
            {columns_info}
            
            Suggest 1 or 2 specific, highly valuable features we can engineer from these exact columns. Give a short, practical explanation.
            """
            
            with st.spinner("🤖 AI is thinking..."):
                try:
                    fe_response = fe_model.generate_content(fe_prompt)
                    st.info(f"💡 *AI Recommendation:*\n\n{fe_response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

       # --- INTERACTIVE EDA & CHARTS SECTION (FULLY DYNAMIC) ---
    st.markdown("<h3 style='color: #29B6F6;'>📊 Automated Exploratory Data Analysis (EDA)</h3>", unsafe_allow_html=True)
    st.write("Explore distributions, relationships, and correlations across all original and newly engineered features.")

    # यह लाइन सुनिश्चित करेगी कि फीचर इंजीनियरिंग के बाद के सभी नए और पुराने न्यूमेरिकल कॉलम्स आ जाएं
    num_cols = working_df.select_dtypes(include=['number']).columns.tolist()

    col_uni, col_bi, col_multi = st.columns(3)

    with col_uni:
        uni_btn = st.button("📈 Univariate Analysis (All Numeric)")

    with col_bi:
        bi_btn = st.button("📉 Bivariate Analysis (All Pairs)")

    with col_multi:
        multi_btn = st.button("🔥 Multivariate Analysis (Full Heatmap)")

    # 1. Univariate Analysis (Dynamic for ALL numeric columns without restriction)
    if uni_btn:
        st.subheader("📈 Univariate Analysis (Distributions for All Features)")
        if len(num_cols) >= 1:
            for col in num_cols:
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.histplot(working_df[col], kde=True, ax=ax, color="#29B6F6")
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns available for analysis.")

    # 2. Bivariate Analysis (Dynamic for all numeric combinations)
    if bi_btn:
        st.subheader("📉 Bivariate Analysis (Feature Relationships)")
        if len(num_cols) >= 2:
            # बिना किसी फिक्स लिमिट के सभी संभव न्यूमेरिकल जोड़ियों (Pairs) के चार्ट्स बनाएं
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.scatterplot(data=working_df, x=num_cols[i], y=num_cols[j], ax=ax, color="#FF7043")
                    ax.set_title(f"{num_cols[i]} vs {num_cols[j]}")
                    st.pyplot(fig)
        else:
            st.warning("⚠️ Need at least 2 numeric columns for Bivariate analysis.")

    # 3. Multivariate Analysis (Full Correlation Heatmap including new features)
    if multi_btn:
        st.subheader("🔥 Multivariate Analysis (Full Correlation Matrix)")
        if len(num_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(working_df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, annot_kws={"size": 8})
            ax.set_title("Correlation Heatmap (Including Engineered Features)")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Need at least 2 numeric columns for Multivariate analysis.")

        # --- DOWNLOAD SECTION (Sabse Last Me) ---
        st.markdown("<h3 style='color: #00897B;'>📥 Download Cleaned Data</h3>", unsafe_allow_html=True)
        csv_data = working_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    # --- GEMINI AI QUERY SECTION ---
    st.markdown("<h3 style='color: #E53935;'>🤖 Ask Questions About Your Data</h3>", unsafe_allow_html=True)
    user_query = st.text_input("Ask a question (e.g., Which columns are numeric or what are the data types?):")

    if user_query:
        if not api_key:
            st.error("⚠️ Please enter your Gemini API Key in the sidebar to ask questions!")
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            summary_info = f"""
            Full Dataset Metadata for Current File ({uploaded_file.name}):
            - Total Rows: {len(working_df)}
            - Total Columns: {list(working_df.columns)}
            - Column Data Types: {working_df.dtypes.to_dict()}
            - Missing Values Per Column: {working_df.isnull().sum().to_dict()}
            - Total Missing Values: {int(working_df.isnull().sum().sum())}
            - Total Duplicate Rows: {int(working_df.duplicated().sum())}
            - First 5 Rows Sample:
            {working_df.head().to_string()}
            """

            prompt = f"{summary_info}\n\nUser Question: {user_query}\n\nProvide an accurate answer based on the current uploaded dataset."

            with st.spinner("Analyzing with AI..."):
                try:
                    response = model.generate_content(prompt)
                    st.write("### 🤖 AI Response:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("📂 Please upload a CSV or Excel file from the left/above to get started.")