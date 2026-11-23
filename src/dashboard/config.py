"""
Dashboard Configuration and Styling Constants.

Defines themes, currency formats, color palettes, and pagination defaults.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DashboardConfig:
    """Dashboard runtime configuration."""
    APP_TITLE: str = "E-Commerce Intelligence Platform"
    APP_SUBTITLE: str = "Enterprise Lakehouse Analytics & Predictive Machine Learning"
    APP_ICON: str = "📊"
    LAYOUT: str = "wide"
    CURRENCY_SYMBOL: str = "£"
    DEFAULT_PAGE_SIZE: int = 25
    CACHE_TTL_SECONDS: int = 3600  # 1 hour cache TTL for analytics


# Palette mapping
PALETTE = {
    "primary": "#1E3A8A",      # Deep Navy
    "secondary": "#0D9488",    # Teal
    "accent": "#F59E0B",       # Amber
    "success": "#10B981",      # Emerald
    "danger": "#EF4444",       # Crimson / Red
    "warning": "#F97316",      # Orange
    "info": "#3B82F6",         # Blue
    "neutral_dark": "#1F2937", # Dark Slate
    "neutral_light": "#F3F4F6",# Off-white / light gray
}

# Transaction status color mapping
STATUS_COLORS: Dict[str, str] = {
    "VALID_PURCHASE": "#10B981",       # Emerald
    "CANCELLATION": "#F59E0B",         # Amber
    "DUPLICATE": "#6B7280",            # Gray
    "INVENTORY_WRITE_OFF": "#EF4444",  # Red
    "ZERO_PRICE_ENTRY": "#9CA3AF",     # Light Gray
}

# Cluster color mapping (Phase 4 K-Means)
CLUSTER_COLORS: Dict[int, str] = {
    0: "#1E3A8A",  # High-Value Loyal Champions (Navy)
    1: "#0D9488",  # Active Recent Shoppers (Teal)
    2: "#3B82F6",  # Lapsed High-Potential (Blue)
    3: "#6B7280",  # Lost Low-Value Buyers (Gray)
    4: "#EF4444",  # Net Returners & Adjusters (Red)
}

CLUSTER_NAMES: Dict[int, str] = {
    0: "High-Value Loyal Champions",
    1: "Active Recent Shoppers",
    2: "Lapsed High-Potential",
    3: "Lost Low-Value Buyers",
    4: "Net Returners & Adjusters",
}


def format_gbp(value: float, decimals: int = 2) -> str:
    """Formats float value into GBP currency string (e.g. £19,012,910.85)."""
    if value is None:
        return "£0.00"
    if value < 0:
        return f"-£{abs(value):,.{decimals}f}"
    return f"£{value:,.{decimals}f}"


def format_number(value: float, decimals: int = 0) -> str:
    """Formats integer or float with thousands separator."""
    if value is None:
        return "0"
    if decimals == 0:
        return f"{int(round(value)):,}"
    return f"{value:,.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Formats ratio as percentage string (e.g. 19.68%)."""
    if value is None:
        return "0.00%"
    return f"{value * 100:.{decimals}f}%"
