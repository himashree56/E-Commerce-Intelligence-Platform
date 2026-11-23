"""
KPI Metric Cards Component.

Provides consistent, styled visual cards for key commercial, operational, and ML metrics.
"""
from typing import Any, Dict, List, Optional
import streamlit as st
from src.dashboard.config import PALETTE


def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
    border: bool = True,
) -> None:
    """Renders a single KPI metric card."""
    with st.container(border=border):
        st.metric(
            label=label,
            value=value,
            delta=delta,
            delta_color=delta_color,
            help=help_text,
        )


def render_kpi_row(cards: List[Dict[str, Any]]) -> None:
    """
    Renders a row of metric cards evenly distributed across columns.

    Each card dict can contain:
    - label: str
    - value: str
    - delta: Optional[str]
    - delta_color: 'normal' | 'inverse' | 'off'
    - help_text: Optional[str]
    """
    if not cards:
        return
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            render_kpi_card(
                label=card.get("label", ""),
                value=card.get("value", ""),
                delta=card.get("delta"),
                delta_color=card.get("delta_color", "normal"),
                help_text=card.get("help_text"),
                border=card.get("border", True),
            )
