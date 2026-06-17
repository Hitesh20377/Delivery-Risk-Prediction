import sys
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.kpi_calculator import calculate_kpis
from src.risk_scoring import risk_category
from src.feature_config import get_model_features

st.set_page_config(
    page_title="APL Delivery Risk Prediction",
    page_icon="🚚",
    layout="wide"
)
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data_sample.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "delivery_risk_model.pkl")
FIGURE_DIR = os.path.join(BASE_DIR, "reports", "figures")
CSS_PATH = os.path.join(BASE_DIR, "dashboard", "style.css")


def load_css():
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)



def load_model():
    return joblib.load(MODEL_PATH)


def safe_unique(dataframe, col):
    if col in dataframe.columns:
        return sorted(dataframe[col].dropna().unique())
    return []


def recommendation_from_risk(prob):
    if prob >= 0.70:
        return "Immediate intervention required"
    elif prob >= 0.30:
        return "Monitor shipment closely"
    return "Normal processing"


def prepare_model_input(input_df):
    X = get_model_features(input_df).copy()

    expected_cols = getattr(model, "feature_names_in_", None)

    if expected_cols is not None:
        X = X.reindex(columns=expected_cols)

    X = X.replace({pd.NA: np.nan})

    return X


def predict_delay_probability(input_df):
    X = prepare_model_input(input_df)

    X = X.replace({pd.NA: np.nan})

    for col in X.columns:
        if str(X[col].dtype) in ["Int64", "Int32", "Float64", "Float32"]:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    try:
        return model.predict_proba(X)[:, 1]

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.exception(e)
        st.stop()

df = load_data()
model = load_model()
load_css()

st.markdown(
    "<div class='main-title'>🚚 Predictive Delivery Risk Intelligence System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>APL Logistics | Late Delivery Risk Prediction Dashboard</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Executive Overview",
        "Executive Summary",
        "Delay Risk Analysis",
        "Region & Market Analysis",
        "Shipping Mode Analysis",
        "Order Prediction",
        "Model Performance",
        "Operations Action Panel"
    ]
)

st.sidebar.header("Interactive Filters")

markets = st.sidebar.multiselect(
    "Market",
    options=safe_unique(df, "Market"),
    default=safe_unique(df, "Market")
)

shipping_modes = st.sidebar.multiselect(
    "Shipping Mode",
    options=safe_unique(df, "Shipping Mode"),
    default=safe_unique(df, "Shipping Mode")
)

customer_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=safe_unique(df, "Customer Segment"),
    default=safe_unique(df, "Customer Segment")
)

order_regions = st.sidebar.multiselect(
    "Order Region",
    options=safe_unique(df, "Order Region"),
    default=safe_unique(df, "Order Region")
)

order_statuses = st.sidebar.multiselect(
    "Order Status",
    options=safe_unique(df, "Order Status"),
    default=safe_unique(df, "Order Status")
)

risk_threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05
)

top_n = st.sidebar.slider(
    "Top N Records",
    min_value=5,
    max_value=50,
    value=15,
    step=5
)

filtered_df = df.copy()

if "Market" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Market"].isin(markets)]

if "Shipping Mode" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Shipping Mode"].isin(shipping_modes)]

if "Customer Segment" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Customer Segment"].isin(customer_segments)]

if "Order Region" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Order Region"].isin(order_regions)]

if "Order Status" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Order Status"].isin(order_statuses)]

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

kpis = calculate_kpis(filtered_df)
delayed_df = filtered_df[filtered_df["Late_delivery_risk"] == 1]

sales_at_risk = delayed_df["Sales"].sum()
profit_at_risk = delayed_df["Order Profit Per Order"].sum()
avg_shipping_days = round(filtered_df["Days for shipping (real)"].mean(), 2)


if page == "Executive Overview":

    st.header("📊 Executive Overview")

    st.markdown(
        """
        <div class='insight-box'>
        This dashboard identifies late delivery risk, risky regions,
        risky shipping modes, high-risk orders, and operational action priorities.
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["KPI Summary", "Risk Overview", "Business Insights"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Orders", f"{kpis['Total Orders']:,}")
        c2.metric("Delayed Orders", f"{kpis['Delayed Orders']:,}")
        c3.metric("Delay Rate", f"{kpis['Delay Rate']}%")
        c4.metric("Avg Shipping Days", avg_shipping_days)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Total Sales", f"${kpis['Total Sales']:,.0f}")
        c6.metric("Total Profit", f"${kpis['Total Profit']:,.0f}")
        c7.metric("Sales at Risk", f"${sales_at_risk:,.0f}")
        c8.metric("Profit at Risk", f"${profit_at_risk:,.0f}")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            risk_labels = filtered_df["Late_delivery_risk"].map({
                0: "Not Delayed",
                1: "Delayed"
            })

            fig = px.pie(
                names=risk_labels,
                title="Delayed vs Non-Delayed Orders"
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            segment_risk = (
                filtered_df.groupby("Customer Segment")["Late_delivery_risk"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                segment_risk,
                x="Customer Segment",
                y="Late_delivery_risk",
                title="Delay Risk by Customer Segment",
                labels={"Late_delivery_risk": "Delay Risk Rate"}
            )
            st.plotly_chart(fig, width="stretch")

        region_risk = (
            filtered_df.groupby("Order Region")["Late_delivery_risk"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        fig = px.bar(
            region_risk,
            x="Late_delivery_risk",
            y="Order Region",
            orientation="h",
            title=f"Top {top_n} Regions by Delay Risk",
            labels={"Late_delivery_risk": "Delay Risk Rate"}
        )
        st.plotly_chart(fig, width="stretch")

    with tab3:
        most_risky_region = (
            filtered_df.groupby("Order Region")["Late_delivery_risk"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )

        most_risky_mode = (
            filtered_df.groupby("Shipping Mode")["Late_delivery_risk"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )

        st.markdown(
            f"""
            <div class='warn-box'>
            <b>Key Insight:</b> The most risky region is <b>{most_risky_region}</b>
            and the most risky shipping mode is <b>{most_risky_mode}</b>.
            Operations should prioritize high-value orders from these segments.
            </div>
            """,
            unsafe_allow_html=True
        )

        summary_table = pd.DataFrame({
            "Business Area": [
                "Delay monitoring",
                "Sales protection",
                "Operational priority",
                "Customer retention"
            ],
            "Recommendation": [
                "Track high-risk regions daily",
                "Prioritize high-sales orders with delay risk",
                "Use risk threshold to build action queue",
                "Communicate early with customers for risky orders"
            ]
        })

        st.dataframe(summary_table, width="stretch")


elif page == "Executive Summary":

    st.header("📋 Executive Summary")

    delay_rate = round(filtered_df["Late_delivery_risk"].mean() * 100, 2)

    most_risky_region = (
        filtered_df.groupby("Order Region")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    most_risky_mode = (
        filtered_df.groupby("Shipping Mode")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Orders Analysed", f"{len(filtered_df):,}")
    c2.metric("Delay Rate", f"{delay_rate}%")
    c3.metric("Sales at Risk", f"${sales_at_risk:,.0f}")
    c4.metric("Profit at Risk", f"${profit_at_risk:,.0f}")

    st.markdown(
        f"""
        <div class='insight-box'>
        <b>Summary:</b> The system analysed <b>{len(filtered_df):,}</b> logistics orders
        and found a delay rate of <b>{delay_rate}%</b>. The most risky region is
        <b>{most_risky_region}</b>, while the most risky shipping mode is
        <b>{most_risky_mode}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    rec_df = pd.DataFrame({
        "Area": [
            "High-risk orders",
            "Shipping mode",
            "Regional operations",
            "Customer communication"
        ],
        "Recommendation": [
            "Prioritize orders above the selected risk threshold.",
            "Review risky shipping modes and allocate backup capacity.",
            "Monitor regions with consistently high delay probability.",
            "Notify customers early for high-risk deliveries."
        ]
    })

    st.subheader("Key Business Recommendations")
    st.dataframe(rec_df, width="stretch")


elif page == "Delay Risk Analysis":

    st.header("⚠️ Delay Risk Analysis")

    tab1, tab2, tab3 = st.tabs(["Shipping Analysis", "Sales Impact", "Delay Gap"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                filtered_df,
                x="Days for shipping (real)",
                color="Late_delivery_risk",
                title="Real Shipping Days vs Delay Risk"
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            fig = px.box(
                filtered_df,
                x="Late_delivery_risk",
                y="Days for shipment (scheduled)",
                title="Scheduled Days Distribution"
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(
                filtered_df,
                x="Late_delivery_risk",
                y="Sales",
                title="Sales Distribution by Delay Risk"
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            fig = px.box(
                filtered_df,
                x="Late_delivery_risk",
                y="Order Profit Per Order",
                title="Profit Distribution by Delay Risk"
            )
            st.plotly_chart(fig, width="stretch")

    with tab3:
        if "Delay_Gap" in filtered_df.columns:
            fig = px.histogram(
                filtered_df,
                x="Delay_Gap",
                color="Late_delivery_risk",
                title="Delay Gap Distribution"
            )
            st.plotly_chart(fig, width="stretch")


elif page == "Region & Market Analysis":

    st.header("🌍 Region & Market Analysis")

    tab1, tab2, tab3 = st.tabs(["Market Risk", "Country Risk", "Risk Heatmap"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            market_risk = (
                filtered_df.groupby("Market")["Late_delivery_risk"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                market_risk,
                x="Market",
                y="Late_delivery_risk",
                title="Delay Risk by Market",
                labels={"Late_delivery_risk": "Delay Risk Rate"}
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            market_sales = (
                filtered_df.groupby("Market")["Sales"]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                market_sales,
                x="Market",
                y="Sales",
                title="Sales by Market"
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        country_risk = (
            filtered_df.groupby("Order Country")["Late_delivery_risk"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        fig = px.bar(
            country_risk,
            x="Late_delivery_risk",
            y="Order Country",
            orientation="h",
            title=f"Top {top_n} Risky Countries",
            labels={"Late_delivery_risk": "Delay Risk Rate"}
        )
        st.plotly_chart(fig, width="stretch")

    with tab3:
        heatmap_df = pd.pivot_table(
            filtered_df,
            values="Late_delivery_risk",
            index="Market",
            columns="Shipping Mode",
            aggfunc="mean"
        )

        fig = px.imshow(
            heatmap_df,
            text_auto=True,
            aspect="auto",
            title="Market vs Shipping Mode Risk Heatmap"
        )
        st.plotly_chart(fig, width="stretch")


elif page == "Shipping Mode Analysis":

    st.header("🚛 Shipping Mode Analysis")

    tab1, tab2 = st.tabs(["Mode Risk", "Sales Analysis"])

    with tab1:
        mode_risk = (
            filtered_df.groupby("Shipping Mode")["Late_delivery_risk"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            mode_risk,
            x="Shipping Mode",
            y="Late_delivery_risk",
            title="Average Delay Risk by Shipping Mode",
            labels={"Late_delivery_risk": "Delay Risk Rate"}
        )
        st.plotly_chart(fig, width="stretch")

        fig = px.box(
            filtered_df,
            x="Shipping Mode",
            y="Days for shipping (real)",
            title="Shipping Days by Shipping Mode"
        )
        st.plotly_chart(fig, width="stretch")

    with tab2:
        sales_mode = (
            filtered_df.groupby("Shipping Mode")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            sales_mode,
            x="Shipping Mode",
            y="Sales",
            title="Sales by Shipping Mode"
        )
        st.plotly_chart(fig, width="stretch")

        profit_mode = (
            filtered_df.groupby("Shipping Mode")["Order Profit Per Order"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            profit_mode,
            x="Shipping Mode",
            y="Order Profit Per Order",
            title="Profit by Shipping Mode"
        )
        st.plotly_chart(fig, width="stretch")


elif page == "Order Prediction":

    st.header("🎯 Order-Level Risk Prediction")

    tab1, tab2 = st.tabs(["Manual Prediction", "Sample Order Prediction"])

    with tab1:
        sample_order = filtered_df.sample(1, random_state=42).copy()

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_market = st.selectbox("Market", safe_unique(df, "Market"))
            selected_region = st.selectbox("Order Region", safe_unique(df, "Order Region"))
            selected_country = st.selectbox("Order Country", safe_unique(df, "Order Country"))

        with col2:
            selected_shipping = st.selectbox("Shipping Mode", safe_unique(df, "Shipping Mode"))
            selected_segment = st.selectbox("Customer Segment", safe_unique(df, "Customer Segment"))
            selected_status = st.selectbox("Order Status", safe_unique(df, "Order Status"))

        with col3:
            sales_value = st.number_input(
                "Sales",
                min_value=0.0,
                value=float(filtered_df["Sales"].median())
            )

            quantity_value = st.number_input(
                "Order Item Quantity",
                min_value=1,
                value=int(filtered_df["Order Item Quantity"].median())
            )

            product_price_value = st.number_input(
                "Product Price",
                min_value=0.0,
                value=float(filtered_df["Product Price"].median())
            )

        sample_order.loc[:, "Market"] = selected_market
        sample_order.loc[:, "Order Region"] = selected_region
        sample_order.loc[:, "Order Country"] = selected_country
        sample_order.loc[:, "Shipping Mode"] = selected_shipping
        sample_order.loc[:, "Customer Segment"] = selected_segment
        sample_order.loc[:, "Order Status"] = selected_status
        sample_order.loc[:, "Sales"] = sales_value
        sample_order.loc[:, "Order Item Quantity"] = quantity_value
        sample_order.loc[:, "Product Price"] = product_price_value

        if st.button("Predict Manual Order Risk", key="manual_pred_btn"):
            prob = predict_delay_probability(sample_order)[0]
            category = risk_category(prob)
            recommendation = recommendation_from_risk(prob)

            c1, c2, c3 = st.columns(3)
            c1.metric("Delay Probability", f"{prob * 100:.2f}%")
            c2.metric("Risk Category", category)
            c3.metric("Recommended Action", recommendation)

    with tab2:
        if "Order Id" in filtered_df.columns:
            order_ids = filtered_df["Order Id"].dropna().unique()
        else:
            order_ids = filtered_df.index

        selected_order = st.selectbox(
            "Select Existing Order",
            options=order_ids
        )

        if "Order Id" in filtered_df.columns:
            sample = filtered_df[filtered_df["Order Id"] == selected_order].head(1)
        else:
            sample = filtered_df.loc[[selected_order]]

        preview_cols = [
            "Order Id",
            "Market",
            "Order Region",
            "Order Country",
            "Shipping Mode",
            "Customer Segment",
            "Sales",
            "Order Item Quantity",
            "Product Price"
        ]

        preview_cols = [col for col in preview_cols if col in sample.columns]

        st.dataframe(sample[preview_cols], width="stretch")

        if st.button("Predict Selected Order Risk", key="selected_pred_btn"):
            prob = predict_delay_probability(sample)[0]
            category = risk_category(prob)
            recommendation = recommendation_from_risk(prob)

            c1, c2, c3 = st.columns(3)
            c1.metric("Delay Probability", f"{prob * 100:.2f}%")
            c2.metric("Risk Category", category)
            c3.metric("Recommended Action", recommendation)


elif page == "Model Performance":

    st.header("🤖 Model Performance")

    tab1, tab2, tab3 = st.tabs(["Metrics", "Confusion Matrix & ROC", "Model Comparison"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", "67.94%")
        c2.metric("Precision", "73.33%")
        c3.metric("Recall", "65.27%")
        c4.metric("F1 Score", "69.07%")

        st.markdown(
            """
            <div class='insight-box'>
            Random Forest was selected as the final model because it provides
            a stronger balance between recall and F1 score for late delivery risk prediction.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            confusion_path = os.path.join(FIGURE_DIR, "confusion_matrix.png")
            if os.path.exists(confusion_path):
                st.image(confusion_path, width="stretch")
            else:
                st.warning("confusion_matrix.png not found. Run python main.py first.")

        with col2:
            st.subheader("ROC Curve")
            roc_path = os.path.join(FIGURE_DIR, "roc_curve.png")
            if os.path.exists(roc_path):
                st.image(roc_path, width="stretch")
            else:
                st.warning("roc_curve.png not found. Run python main.py first.")

        st.subheader("Feature Importance")
        feature_path = os.path.join(FIGURE_DIR, "feature_importance.png")
        if os.path.exists(feature_path):
            st.image(feature_path, width="stretch")
        else:
            st.warning("feature_importance.png not found.")

    with tab3:
        comparison_path = os.path.join(BASE_DIR, "models", "model_comparison.csv")
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
            st.subheader("Model Benchmark Comparison")
            st.dataframe(comparison_df, width="stretch")
        else:
            st.warning("model_comparison.csv not found. Run python main.py first.")


elif page == "Operations Action Panel":

    st.header("🚨 Operations Action Panel")

    tab1, tab2, tab3 = st.tabs(["Action Queue", "Risk Distribution", "Recommendations"])

    probabilities = predict_delay_probability(filtered_df)

    action_df = filtered_df.copy()
    action_df["Delay Probability"] = probabilities
    action_df["Risk Category"] = action_df["Delay Probability"].apply(risk_category)
    action_df["Recommendation"] = action_df["Delay Probability"].apply(recommendation_from_risk)

    high_risk_df = action_df[action_df["Delay Probability"] >= risk_threshold]

    with tab1:
        c1, c2, c3 = st.columns(3)

        c1.metric("Orders Requiring Attention", f"{len(high_risk_df):,}")

        if len(high_risk_df) > 0:
            c2.metric("Avg Delay Probability", f"{high_risk_df['Delay Probability'].mean() * 100:.2f}%")
            c3.metric("Sales at Risk", f"${high_risk_df['Sales'].sum():,.0f}")
        else:
            c2.metric("Avg Delay Probability", "0%")
            c3.metric("Sales at Risk", "$0")

        display_cols = [
            "Order Id",
            "Market",
            "Order Region",
            "Order Country",
            "Order Status",
            "Shipping Mode",
            "Customer Segment",
            "Sales",
            "Order Profit Per Order",
            "Delay Probability",
            "Risk Category",
            "Recommendation"
        ]

        display_cols = [col for col in display_cols if col in high_risk_df.columns]

        st.subheader("Top High-Risk Orders")

        st.dataframe(
            high_risk_df[display_cols]
            .sort_values("Delay Probability", ascending=False)
            .head(top_n),
            width="stretch"
        )

        st.download_button(
            label="📥 Download High-Risk Orders CSV",
            data=high_risk_df[display_cols].to_csv(index=False),
            file_name="high_risk_orders.csv",
            mime="text/csv"
        )

    with tab2:
        risk_dist = action_df["Risk Category"].value_counts().reset_index()
        risk_dist.columns = ["Risk Category", "Count"]

        fig = px.pie(
            risk_dist,
            names="Risk Category",
            values="Count",
            title="Risk Category Distribution"
        )
        st.plotly_chart(fig, width="stretch")

        fig = px.histogram(
            action_df,
            x="Delay Probability",
            color="Risk Category",
            title="Delay Probability Distribution"
        )
        st.plotly_chart(fig, width="stretch")

    with tab3:
        rec_summary = (
            action_df.groupby("Recommendation")
            .agg(
                Orders=("Recommendation", "count"),
                Sales_Impact=("Sales", "sum"),
                Avg_Risk=("Delay Probability", "mean")
            )
            .reset_index()
        )

        st.dataframe(rec_summary, width="stretch")

        st.markdown(
            """
            <div class='warn-box'>
            <b>Action Rule:</b> Orders above the selected risk threshold should be reviewed first.
            High-risk shipments may require rerouting, priority handling, or early customer communication.
            </div>
            """,
            unsafe_allow_html=True
        )
