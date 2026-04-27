"""Simulator-first synthetic market package for the piroth2 branch."""

from .config import DiagnosticsConfig, SimulatorConfig, SymbolSpec
from .real_data import RealMarketDataLoader, load_market_days
from .simulator import SyntheticDay, SyntheticMarketGenerator

__all__ = [
    "DiagnosticsConfig",
    "RealMarketDataLoader",
    "SimulatorConfig",
    "SymbolSpec",
    "SyntheticDay",
    "SyntheticMarketGenerator",
    "load_market_days",
]
