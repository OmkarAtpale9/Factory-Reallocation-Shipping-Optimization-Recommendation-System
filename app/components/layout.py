from __future__ import annotations

import streamlit as st


def set_page_config(app_name: str) -> None:
    st.set_page_config(
        page_title=app_name,
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def kpi_row(*, avg_lead_time: float, delay_frequency: float, total_shipments: int) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Lead Time (days)", f"{avg_lead_time:.2f}")
    c2.metric("Delay Frequency", f"{delay_frequency:.1%}")
    c3.metric("Total Shipments", f"{total_shipments:,}")

