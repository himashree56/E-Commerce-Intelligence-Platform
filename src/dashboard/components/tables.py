"""
Interactive Data Table Component.

Provides styled table displays, column formatters, and CSV export capabilities.
"""
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st


def render_table(
    df: pd.DataFrame,
    title: Optional[str] = None,
    download_filename: Optional[str] = None,
    column_config: Optional[Dict] = None,
    height: Optional[int] = None,
    hide_index: bool = True,
) -> None:
    """
    Renders an interactive DataFrame with optional header and CSV download button.
    """
    if title:
        st.subheader(title)

    if df.empty:
        st.info("No records found matching the specified criteria.")
        return

    # Table display
    st.dataframe(
        df,
        column_config=column_config,
        height=height,
        hide_index=hide_index,
        use_container_width=True,
    )

    # Download button if filename provided
    if download_filename:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {download_filename} (CSV)",
            data=csv,
            file_name=download_filename,
            mime="text/csv",
            key=f"dl_{download_filename}_{hash(title or download_filename)}",
        )
