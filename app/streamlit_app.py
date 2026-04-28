from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow Streamlit to import modules from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.analytics.kpi_engine import calculate_inventory_kpis
from src.data.validation import ValidationResult, validate_inventory_data


SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "sample_inventory.csv"


def configure_page() -> None:
    """
    Configure the main Streamlit page.
    """

    st.set_page_config(
        page_title="StockSense AI",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_css() -> None:
    """
    Apply custom CSS to make the Streamlit dashboard look like a polished SaaS product.
    """

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
            }
            .block-container {
                color: #0f172a;
            }

            h1, h2, h3, h4, h5, h6, p, span, div {
                color: inherit;
            }

            section[data-testid="stSidebar"] {
                background: #0f172a;
                border-right: 1px solid #1e293b;
            }

            section[data-testid="stSidebar"] * {
                color: #f8fafc;
            }

            .main-header {
                padding: 1.6rem 1.8rem;
                border-radius: 24px;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
                color: white;
                margin-bottom: 1.5rem;
                box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
            }

            .main-header h1 {
                font-size: 2.4rem;
                font-weight: 800;
                margin-bottom: 0.4rem;
                letter-spacing: -0.03em;
            }

            .main-header p {
                font-size: 1.02rem;
                color: #cbd5e1;
                margin-bottom: 0;
                max-width: 900px;
            }

            .section-title {
                font-size: 1.25rem;
                font-weight: 750;
                color: #0f172a;
                margin-top: 0.6rem;
                margin-bottom: 0.7rem;
            }

            .section-subtitle {
                font-size: 0.92rem;
                color: #64748b;
                margin-top: -0.4rem;
                margin-bottom: 1rem;
            }

            .metric-card {
                padding: 1.1rem 1.2rem;
                border-radius: 20px;
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(226, 232, 240, 0.95);
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
                min-height: 128px;
            }

            .metric-label {
                font-size: 0.78rem;
                font-weight: 700;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.4rem;
            }

            .metric-value {
                font-size: 1.75rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.2;
                margin-bottom: 0.35rem;
            }

            .metric-help {
                font-size: 0.82rem;
                color: #64748b;
                line-height: 1.35;
            }

            .insight-card {
            padding: 1.1rem 1.25rem;
            border-radius: 18px;
            background: #ffffff;
            border-left: 5px solid #2563eb;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.75rem;
            color: #0f172a !important;
            font-size: 1rem;
            line-height: 1.6;
            }

            .insight-card strong {
                color: #0f172a !important;
                font-weight: 800;
            }   

            .warning-card {
                padding: 1rem 1.2rem;
                border-radius: 18px;
                background: #fff7ed;
                border-left: 5px solid #f97316;
                color: #7c2d12;
                margin-bottom: 0.75rem;
            }

            .success-card {
                padding: 1rem 1.2rem;
                border-radius: 18px;
                background: #ecfdf5;
                border-left: 5px solid #10b981;
                color: #064e3b;
                margin-bottom: 0.75rem;
            }

            .error-card {
                padding: 1rem 1.2rem;
                border-radius: 18px;
                background: #fef2f2;
                border-left: 5px solid #ef4444;
                color: #7f1d1d;
                margin-bottom: 0.75rem;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid #e2e8f0;
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
            }

            div[data-testid="stMetric"] {
                background: white;
                padding: 1rem;
                border-radius: 18px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }

            .small-muted {
                font-size: 0.82rem;
                color: #64748b;
            }

            .footer-note {
                margin-top: 2rem;
                padding: 1rem;
                color: #64748b;
                text-align: center;
                font-size: 0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    """
    Format a number as currency.
    """

    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format a decimal as a percentage.
    """

    return f"{value * 100:.1f}%"


def render_header() -> None:
    """
    Render main dashboard header.
    """

    st.markdown(
        """
        <div class="main-header">
            <h1>📦 StockSense AI</h1>
            <p>
                Enterprise-style inventory intelligence platform for automated data validation,
                KPI analytics, demand insights, stock risk detection, and AI-ready recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[pd.DataFrame | None, str]:
    """
    Render sidebar upload controls and return selected dataframe.
    """

    st.sidebar.markdown("## StockSense AI")
    st.sidebar.markdown("Inventory Intelligence Platform")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Upload Inventory Data")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload inventory data with required columns such as date, product, stock, sales, waste, and price.",
    )

    use_sample_data = st.sidebar.checkbox(
        "Use sample demo dataset",
        value=uploaded_file is None,
        help="Use synthetic data for demo and development.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Version")
    st.sidebar.markdown("**MVP Module:** Dashboard + KPI Analytics")
    st.sidebar.markdown("**Status:** Development")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            return df, f"Uploaded file: {uploaded_file.name}"

        except Exception as exc:
            st.sidebar.error(f"Could not read uploaded file: {exc}")
            return None, "Upload failed"

    if use_sample_data:
        if SAMPLE_DATA_PATH.exists():
            df = pd.read_csv(SAMPLE_DATA_PATH)
            return df, "Sample demo dataset"

        st.sidebar.error("Sample dataset not found. Run scripts/generate_sample_data.py first.")
        return None, "Sample dataset missing"

    return None, "No dataset selected"


def render_empty_state() -> None:
    """
    Render empty state when no data is selected.
    """

    st.markdown(
        """
        <div class="insight-card">
            <h3>Start by uploading inventory data</h3>
            <p>
                Upload a CSV or Excel file from the sidebar, or enable the sample demo dataset.
                StockSense AI will validate the data, calculate KPIs, identify risks,
                and prepare insights for decision-making.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_validation_result(validation_result: ValidationResult, data_source: str) -> None:
    """
    Render data validation status.
    """

    st.markdown('<div class="section-title">Data Intake Status</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1, 2])

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Source</div>
                <div class="metric-value">{data_source}</div>
                <div class="metric-help">Current dataset used for analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        score = validation_result.data_quality_score
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Quality Score</div>
                <div class="metric-value">{score}/100</div>
                <div class="metric-help">Based on schema, missing values, invalid values, and business rules</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        if validation_result.is_valid:
            st.markdown(
                """
                <div class="success-card">
                    <strong>Validation passed.</strong><br>
                    The dataset is ready for KPI analytics, dashboarding, forecasting, and recommendations.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="error-card">
                    <strong>Validation failed.</strong><br>
                    Please fix the critical issues before continuing with analytics.
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metric_card(label: str, value: str, help_text: str) -> None:
    """
    Render a custom metric card.
    """

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_summary(summary: Dict[str, Any], risk_summary: Dict[str, int]) -> None:
    """
    Render executive KPI cards.
    """

    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">High-level business performance indicators from the selected inventory dataset.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            "Total Revenue",
            format_currency(summary["total_revenue"]),
            "Revenue generated from sold inventory",
        )

    with col2:
        render_metric_card(
            "Units Sold",
            f'{summary["total_units_sold"]:,}',
            "Total quantity sold during the selected period",
        )

    with col3:
        render_metric_card(
            "Waste Value",
            format_currency(summary["total_waste_value"]),
            "Estimated value lost due to wasted stock",
        )

    with col4:
        render_metric_card(
            "Inventory Value",
            format_currency(summary["latest_inventory_value"]),
            "Latest closing inventory value, not cumulative",
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        render_metric_card(
            "Inventory Health",
            f'{summary["inventory_health_score"]}/100',
            "Overall inventory condition score",
        )

    with col6:
        render_metric_card(
            "Sell-Through Rate",
            format_percentage(summary["average_sell_through_rate"]),
            "Share of available stock that was sold",
        )

    with col7:
        render_metric_card(
            "Waste Rate",
            format_percentage(summary["average_waste_rate"]),
            "Share of product movement that became waste",
        )

    with col8:
        render_metric_card(
            "High-Risk Products",
            f'{risk_summary["high_stockout_risk_products"] + risk_summary["high_overstock_risk_products"]}',
            "Products with high stockout or overstock risk",
        )


def render_ai_style_insights(
    product_performance: pd.DataFrame,
    category_performance: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:
    """
    Render simple rule-based executive insights.

    Later, this section can be enhanced using the AI recommendation engine.
    """

    st.markdown('<div class="section-title">Executive Insights</div>', unsafe_allow_html=True)

    top_product = product_performance.iloc[0]
    highest_waste = product_performance.sort_values("total_waste_value", ascending=False).iloc[0]
    best_category = category_performance.iloc[0]

    insights = [
        (
            f"Top revenue product is <strong>{top_product['product_name']}</strong>, "
            f"generating <strong>{format_currency(top_product['total_revenue'])}</strong> in revenue."
        ),
        (
            f"Best-performing category is <strong>{best_category['category']}</strong>, "
            f"with <strong>{format_currency(best_category['total_revenue'])}</strong> total revenue."
        ),
        (
            f"Highest waste value is from <strong>{highest_waste['product_name']}</strong>, "
            f"causing an estimated loss of <strong>{format_currency(highest_waste['total_waste_value'])}</strong>."
        ),
        (
            f"Overall inventory health score is <strong>{summary['inventory_health_score']}/100</strong>, "
            f"based on sell-through, waste, stockout, and overstock risk."
        ),
    ]

    for insight in insights:
        st.markdown(
            f"""
            <div class="insight-card">
                {insight}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_charts(
    enriched_data: pd.DataFrame,
    product_performance: pd.DataFrame,
    category_performance: pd.DataFrame,
) -> None:
    """
    Render interactive Plotly charts.
    """

    st.markdown('<div class="section-title">Performance Analytics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        daily_revenue = (
            enriched_data.groupby("date", as_index=False)
            .agg(revenue=("revenue", "sum"))
            .sort_values("date")
        )

        fig = px.line(
            daily_revenue,
            x="date",
            y="revenue",
            title="Daily Revenue Trend",
            markers=True,
        )
        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            template="plotly_white",
            font=dict(color="#0f172a"),
            title_font=dict(color="#0f172a", size=18),
            xaxis=dict(
                title_font=dict(color="#475569"),
                tickfont=dict(color="#64748b"),
                gridcolor="#e2e8f0",
            ),
            yaxis=dict(
                title_font=dict(color="#475569"),
                tickfont=dict(color="#64748b"),
                gridcolor="#e2e8f0",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_products = product_performance.head(8)

        fig = px.bar(
            top_products,
            x="total_revenue",
            y="product_name",
            orientation="h",
            title="Top Products by Revenue",
            labels={"total_revenue": "Revenue", "product_name": "Product"},
        )
        fig.update_layout(
            height=380,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.bar(
            category_performance,
            x="category",
            y="total_revenue",
            title="Revenue by Category",
            labels={"total_revenue": "Revenue", "category": "Category"},
        )
        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        waste_by_product = product_performance.sort_values(
            "total_waste_value", ascending=False
        ).head(8)

        fig = px.bar(
            waste_by_product,
            x="total_waste_value",
            y="product_name",
            orientation="h",
            title="Highest Waste Value by Product",
            labels={"total_waste_value": "Waste Value", "product_name": "Product"},
        )
        fig.update_layout(
            height=380,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_product_table(product_performance: pd.DataFrame) -> None:
    """
    Render product performance table.
    """

    st.markdown('<div class="section-title">Product Performance</div>', unsafe_allow_html=True)

    display_df = product_performance[
        [
            "product_name",
            "category",
            "total_revenue",
            "total_units_sold",
            "total_units_wasted",
            "current_stock",
            "sell_through_rate",
            "waste_rate",
            "stockout_risk",
            "overstock_risk",
            "product_health_score",
        ]
    ].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "product_name": "Product",
            "category": "Category",
            "total_revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
            "total_units_sold": st.column_config.NumberColumn("Units Sold"),
            "total_units_wasted": st.column_config.NumberColumn("Units Wasted"),
            "current_stock": st.column_config.NumberColumn("Current Stock"),
            "sell_through_rate": st.column_config.ProgressColumn(
                "Sell-Through Rate",
                format="%.1f",
                min_value=0,
                max_value=1,
            ),
            "waste_rate": st.column_config.ProgressColumn(
                "Waste Rate",
                format="%.1f",
                min_value=0,
                max_value=1,
            ),
            "stockout_risk": "Stockout Risk",
            "overstock_risk": "Overstock Risk",
            "product_health_score": st.column_config.ProgressColumn(
                "Health Score",
                min_value=0,
                max_value=100,
            ),
        },
    )


def render_risk_tables(product_performance: pd.DataFrame) -> None:
    """
    Render stockout and overstock risk tables.
    """

    st.markdown('<div class="section-title">Inventory Risk Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Products that may need immediate business attention.</div>',
        unsafe_allow_html=True,
    )

    stockout_df = product_performance[
        product_performance["stockout_risk"].isin(["High", "Medium"])
    ].copy()

    overstock_df = product_performance[
        product_performance["overstock_risk"].isin(["High", "Medium"])
    ].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Stockout Risk")
        if stockout_df.empty:
            st.markdown(
                """
                <div class="success-card">
                    No high or medium stockout risk products detected.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(
                stockout_df[
                    [
                        "product_name",
                        "category",
                        "current_stock",
                        "avg_daily_demand",
                        "days_of_stock_left",
                        "stockout_risk",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "product_name": "Product",
                    "category": "Category",
                    "current_stock": "Current Stock",
                    "avg_daily_demand": st.column_config.NumberColumn(
                        "Avg Daily Demand", format="%.2f"
                    ),
                    "days_of_stock_left": st.column_config.NumberColumn(
                        "Days Left", format="%.1f"
                    ),
                    "stockout_risk": "Risk",
                },
            )

    with col2:
        st.markdown("#### Overstock Risk")
        if overstock_df.empty:
            st.markdown(
                """
                <div class="success-card">
                    No high or medium overstock risk products detected.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(
                overstock_df[
                    [
                        "product_name",
                        "category",
                        "current_stock",
                        "avg_daily_demand",
                        "days_of_stock_left",
                        "overstock_risk",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "product_name": "Product",
                    "category": "Category",
                    "current_stock": "Current Stock",
                    "avg_daily_demand": st.column_config.NumberColumn(
                        "Avg Daily Demand", format="%.2f"
                    ),
                    "days_of_stock_left": st.column_config.NumberColumn(
                        "Days Left", format="%.1f"
                    ),
                    "overstock_risk": "Risk",
                },
            )


def render_category_table(category_performance: pd.DataFrame) -> None:
    """
    Render category performance table.
    """

    st.markdown('<div class="section-title">Category Performance</div>', unsafe_allow_html=True)

    display_df = category_performance[
        [
            "category",
            "total_revenue",
            "total_units_sold",
            "total_units_wasted",
            "total_waste_value",
            "product_count",
            "sell_through_rate",
            "waste_rate",
        ]
    ].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "category": "Category",
            "total_revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
            "total_units_sold": "Units Sold",
            "total_units_wasted": "Units Wasted",
            "total_waste_value": st.column_config.NumberColumn("Waste Value", format="$%.2f"),
            "product_count": "Product Count",
            "sell_through_rate": st.column_config.ProgressColumn(
                "Sell-Through Rate",
                min_value=0,
                max_value=1,
            ),
            "waste_rate": st.column_config.ProgressColumn(
                "Waste Rate",
                min_value=0,
                max_value=1,
            ),
        },
    )


def render_data_quality_details(validation_result: ValidationResult) -> None:
    """
    Render detailed validation errors and warnings.
    """

    st.markdown('<div class="section-title">Data Quality Report</div>', unsafe_allow_html=True)

    if validation_result.errors:
        st.markdown("#### Critical Errors")
        for error in validation_result.errors:
            st.markdown(
                f"""
                <div class="error-card">
                    {error}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="success-card">
                No critical validation errors found.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if validation_result.warnings:
        st.markdown("#### Warnings")
        for warning in validation_result.warnings:
            st.markdown(
                f"""
                <div class="warning-card">
                    {warning}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="success-card">
                No validation warnings found.
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """
    Main Streamlit application entry point.
    """

    configure_page()
    apply_custom_css()
    render_header()

    raw_df, data_source = render_sidebar()

    if raw_df is None:
        render_empty_state()
        return

    validation_result = validate_inventory_data(raw_df)

    render_validation_result(validation_result, data_source)

    if not validation_result.is_valid:
        render_data_quality_details(validation_result)
        st.stop()

    kpi_result = calculate_inventory_kpis(validation_result.cleaned_data)

    summary = kpi_result["summary_metrics"]
    risk_summary = kpi_result["risk_summary"]
    product_performance = kpi_result["product_performance"]
    category_performance = kpi_result["category_performance"]
    enriched_data = kpi_result["enriched_data"]

    dashboard_tab, products_tab, risks_tab, categories_tab, quality_tab = st.tabs(
        [
            "Overview",
            "Products",
            "Risk Center",
            "Categories",
            "Data Quality",
        ]
    )

    with dashboard_tab:
        render_executive_summary(summary, risk_summary)
        st.markdown("")
        render_ai_style_insights(product_performance, category_performance, summary)
        st.markdown("")
        render_charts(enriched_data, product_performance, category_performance)

    with products_tab:
        render_product_table(product_performance)

    with risks_tab:
        render_risk_tables(product_performance)

    with categories_tab:
        render_category_table(category_performance)

    with quality_tab:
        render_data_quality_details(validation_result)
        st.markdown("#### Preview of Cleaned Data")
        st.dataframe(
            validation_result.cleaned_data.head(50),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """
        <div class="footer-note">
            StockSense AI MVP · Built for inventory intelligence, forecasting, recommendations, and MLOps readiness.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()