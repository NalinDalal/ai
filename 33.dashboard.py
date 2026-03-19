# this is 14 oct program
#Basic Stats Dashboard (load CSV → mean, std, visualize)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Basic Stats Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Data Preview")
    st.dataframe(df.head())

    # Select column
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select a numeric column", numeric_cols)

        # Compute basic stats
        mean_val = df[selected_col].mean()
        std_val = df[selected_col].std()
        min_val = df[selected_col].min()
        max_val = df[selected_col].max()

        st.subheader("📈 Basic Statistics")
        st.write(f"**Mean:** {mean_val:.4f}")
        st.write(f"**Std Dev:** {std_val:.4f}")
        st.write(f"**Min:** {min_val:.4f}")
        st.write(f"**Max:** {max_val:.4f}")

        # Visualization
        st.subheader("📉 Visualization")
        fig, ax = plt.subplots()
        ax.hist(df[selected_col].dropna(), bins=30, edgecolor='black')
        ax.set_title(f"Distribution of {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found in this CSV.")
else:
    st.info("Upload a CSV file to begin.")

# to run: `streamlit run 33.dashboard.py`