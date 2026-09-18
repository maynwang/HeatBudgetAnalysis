from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from heat_budget.config import (
    HeatBudgetParameters,
    SeasonConfig,
)

from heat_budget.plotting import (
    plot_heat_fluxes,
    plot_residual,
)


# ============================================================
# CONFIGURATION
# ============================================================

season = SeasonConfig(
    year=2024,
    start="2024-01-26",
    end="2024-04-15",
    basal_plot_start="2024-02-02",
)

params = HeatBudgetParameters()


# ============================================================
# PATHS
# ============================================================

processed_dir = Path("processed/")
figure_dir = Path("figures/")

figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD PROCESSED HEAT-BUDGET DATA
# ============================================================

daily = xr.open_dataset(
    processed_dir / "HeatBudget_inputs_2024.nc"
)

fluxes = xr.open_dataset(
    processed_dir / "HeatBudget_fluxes_2024.nc"
)


# ============================================================
# FIG. 6 — HEAT FLUXES
# ============================================================

fig6, ax6 = plot_heat_fluxes(
    daily,
    fluxes,
    season,
    params,
)

fig6.savefig(
    figure_dir / "Fig6_HeatFluxes.png",
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# FIG. 7 — RESIDUAL
# ============================================================

fig7, ax7 = plot_residual(
    daily,
    fluxes,
    season,
)

fig7.savefig(
    figure_dir / "Fig7_Residual.png",
    dpi=300,
    bbox_inches="tight",
)


plt.show()