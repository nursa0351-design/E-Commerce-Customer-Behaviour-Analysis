import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from datetime import datetime

st.markdown(
    """
    <style>
    /* Hide hamburger menu / settings icon */
    #MainMenu {visibility: hidden !important;}
    /* Hide top header and loading bar */
    header {visibility: hidden !important; height: 0 !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; height: 0 !important;}
    /* Hide the “Hosted with Streamlit” / GitHub footer */
    footer, div.viewerBadge_container__, .viewerBadge_link__qYDs4, .styles_viewerBadge__CvC9N {visibility: hidden !important; display: none !important;}
    /* Hide full-screen icon if present */
    a[href*="github"] {visibility: hidden !important; display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

hide_badge = """
<style>
[data-testid="stFooter"] {display: none !important;}
a[href*="github"] {display: none !important;}
</style>
"""
st.markdown(hide_badge, unsafe_allow_html=True)

st.set_page_config(page_title="E-Commerce Data Analysis", layout="wide")

st.title("🛍️ E-Commerce Data Analysis & Customer Segmentation Dashboard")
st.markdown("""
This Streamlit app combines:
1. **Customer Behavior Analysis** — KPIs, top products, and revenue trends  
2. **RFM Segmentation** — Clustering customers using Recency, Frequency, and Monetary value  
""")

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("📂 Upload your E-Commerce dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        df = pd.read_excel(uploaded_file)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    # -------- Tabs for Behavior vs Segmentation --------
    tab1, tab2 = st.tabs(["📊 Customer Behavior Dashboard", "🤖 RFM Segmentation & Insights"])

    # ===================================================
    # TAB 1: CUSTOMER BEHAVIOR DASHBOARD
    # ===================================================
    with tab1:
        st.header("📊 Customer Behavior Dashboard")

        # Sidebar filters
        st.sidebar.header("🔍 Filter Options")

        # Date filter
        if "InvoiceDate" in df.columns:
            min_date, max_date = df["InvoiceDate"].min(), df["InvoiceDate"].max()
            start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
            if isinstance(start_date, list):
                start_date, end_date = start_date
            df = df[(df["InvoiceDate"] >= pd.to_datetime(start_date)) & (df["InvoiceDate"] <= pd.to_datetime(end_date))]

        # Country filter
        if "Country" in df.columns:
            country_list = sorted(df["Country"].dropna().unique().tolist())
            country = st.sidebar.selectbox("🌍 Select Country", ["All"] + country_list)
            if country != "All":
                df = df[df["Country"] == country]

        # Product search
        if "Description" in df.columns:
            product_search = st.sidebar.text_input("🔎 Search Product (optional)")
            if product_search:
                df = df[df["Description"].str.contains(product_search, case=False, na=False)]

        st.sidebar.markdown("---")
        st.sidebar.markdown("🧠 *Filters update all charts dynamically!*")

        # KPIs
        total_revenue = df["Revenue"].sum() if "Revenue" in df.columns else 0
        total_orders = df["InvoiceNo"].nunique() if "InvoiceNo" in df.columns else 0
        total_customers = df["CustomerID"].nunique() if "CustomerID" in df.columns else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
        col2.metric("🧾 Total Orders", total_orders)
        col3.metric("👥 Unique Customers", total_customers)

        # Top Products
        if "Description" in df.columns:
            st.write("### 🏆 Top 10 Most Sold Products")
            top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_products.plot(kind="barh", ax=ax, color="teal")
            ax.invert_yaxis()
            ax.set_xlabel("Quantity Sold")
            ax.set_title("Top 10 Products by Quantity Sold")
            st.pyplot(fig)

        # Top Countries
        if "Country" in df.columns and "Revenue" in df.columns:
            st.write("### 🌍 Top 10 Countries by Revenue")
            top_countries = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax, palette="Blues_r")
            ax.set_xlabel("Revenue")
            ax.set_ylabel("Country")
            ax.set_title("Top 10 Countries by Revenue")
            st.pyplot(fig)

        # Monthly Revenue Trend
        if "InvoiceDate" in df.columns and "Revenue" in df.columns:
            st.write("### 📈 Monthly Revenue Trend")
            df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
            monthly_sales = df.groupby("Month")["Revenue"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x="Month", y="Revenue", data=monthly_sales, marker="o", ax=ax, color="crimson")
            plt.xticks(rotation=45)
            ax.set_title("Monthly Revenue Trend")
            st.pyplot(fig)

        # Top Customers
        if "CustomerID" in df.columns and "Revenue" in df.columns:
            st.write("### 👤 Top 10 Customers by Revenue")
            top_customers = df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_customers.index.astype(str), y=top_customers.values, ax=ax, palette="viridis")
            ax.set_xlabel("Customer ID")
            ax.set_ylabel("Revenue")
            ax.set_title("Top 10 Customers by Revenue")
            st.pyplot(fig)

    # ===================================================
    # TAB 2: RFM SEGMENTATION
    # ===================================================
    with tab2:
        st.header("🤖 RFM Segmentation & AI Insights")

        df_rfm = df.copy()
        df_rfm.dropna(subset=["CustomerID"], inplace=True)
        if "InvoiceNo" in df_rfm.columns:
            df_rfm = df_rfm[~df_rfm["InvoiceNo"].astype(str).str.startswith("C")]
        if "InvoiceDate" in df_rfm.columns:
            df_rfm["InvoiceDate"] = pd.to_datetime(df_rfm["InvoiceDate"])
        if "Quantity" in df_rfm.columns and "UnitPrice" in df_rfm.columns:
            df_rfm["TotalPrice"] = df_rfm["Quantity"] * df_rfm["UnitPrice"]

        st.write(f"✅ Cleaned data contains **{df_rfm['CustomerID'].nunique()}** unique customers.")

        # RFM Calculation
        latest_date = df_rfm["InvoiceDate"].max() + pd.Timedelta(days=1)
        rfm = df_rfm.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (latest_date - x.max()).days,
            "InvoiceNo": "count",
            "TotalPrice": "sum"
        }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})

        # K-Means Clustering
        try:
            rfm_log = np.log1p(rfm)
            kmeans = KMeans(n_clusters=4, random_state=42)
            rfm["Cluster"] = kmeans.fit_predict(rfm_log)

            cluster_summary = rfm.groupby("Cluster").agg({
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean",
                "Cluster": "count"
            }).rename(columns={"Cluster": "Num_Customers"}).round(2)

            st.write("### 📈 Cluster Summary")
            st.dataframe(cluster_summary)

            # Boxplots
            col1, col2 = st.columns(2)
            with col1:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x="Cluster", y="Recency", data=rfm)
                st.pyplot(plt.gcf())
                plt.clf()
            with col2:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x="Cluster", y="Frequency", data=rfm)
                st.pyplot(plt.gcf())
                plt.clf()

            st.write("**Monetary Value Distribution by Cluster**")
            plt.figure(figsize=(8, 5))
            sns.boxplot(x="Cluster", y="Monetary", data=rfm)
            st.pyplot(plt.gcf())
            plt.clf()

            # Insights
            st.subheader("💡 Automated Insights Summary")

            insights = []
            for c in cluster_summary.index:
                rec = cluster_summary.loc[c, "Recency"]
                freq = cluster_summary.loc[c, "Frequency"]
                mon = cluster_summary.loc[c, "Monetary"]

                if rec < cluster_summary["Recency"].mean() and freq > cluster_summary["Frequency"].mean() and mon > cluster_summary["Monetary"].mean():
                    label = "⭐ Loyal Customers"
                    note = "They buy frequently, spend more, and purchase recently. Reward them."
                elif rec > cluster_summary["Recency"].mean() and freq < cluster_summary["Frequency"].mean():
                    label = "⚠️ At-Risk Customers"
                    note = "Haven’t purchased recently. Use win-back offers."
                elif freq < cluster_summary["Frequency"].mean() and mon < cluster_summary["Monetary"].mean():
                    label = "🆕 New / Low-Value Customers"
                    note = "New or low spenders. Encourage more purchases."
                else:
                    label = "💤 Potential Regulars"
                    note = "Average spenders — can be converted to loyal customers."

                insights.append({"Cluster": c, "Segment": label, "Insight": note})

            insight_df = pd.DataFrame(insights)
            st.dataframe(insight_df)

            st.markdown("### 🧭 Summary Report")
            for _, row in insight_df.iterrows():
                st.markdown(f"**Cluster {row['Cluster']} → {row['Segment']}**: {row['Insight']}")

            # Download segmented data
            csv = rfm.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Segmented Customer Data",
                               data=csv, file_name="RFM_Clusters.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error during clustering: {e}")

else:
    st.info("👆 Please upload your dataset to start analysis.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Python, and Machine Learning.")
