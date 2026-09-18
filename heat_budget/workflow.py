from typing import Optional, Tuple

import xarray as xr

from .calculations import (
    calculate_basal_balance,
    calculate_conduction,
    calculate_radiation,
    calculate_surface_balance,
    calculate_turbulent_fluxes,
    calculate_uncertainties,
)
from .config import HeatBudgetParameters, SeasonConfig
from .loaders import load_raw_data, preprocess_daily
from .residual import calculate_residual


def run_heat_budget(
    cfg: SeasonConfig,
    params: Optional[HeatBudgetParameters] = None,
    include_residual: bool = True,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Run the complete daily heat-budget workflow."""

    if params is None:
        params = HeatBudgetParameters()

    raw = load_raw_data(cfg)
    daily = preprocess_daily(raw, cfg, params)

    conduction = calculate_conduction(daily, params)
    radiation = calculate_radiation(daily, params)
    turbulent = calculate_turbulent_fluxes(daily, params)

    surface = calculate_surface_balance(
        conduction,
        radiation,
        turbulent,
    )

    basal = calculate_basal_balance(
        daily,
        conduction,
        params,
    )

    uncertainty = calculate_uncertainties(
        daily,
        conduction,
        radiation,
        turbulent,
        params,
    )

    fluxes = xr.merge(
        [
            conduction,
            radiation,
            turbulent,
            surface,
            basal,
            uncertainty,
        ],
        compat="override",
    )

    if include_residual:
        residual = calculate_residual(
            daily,
            fluxes,
            raw,
            params,
        )
        fluxes = xr.merge(
            [fluxes, residual],
            compat="override",
        )

    fluxes.attrs.update(
        {
            "year": cfg.year,
            "analysis_start": cfg.start,
            "analysis_end": cfg.end,
            "time_resolution": "daily",
            "sign_convention": "positive into the ice",
            "L_ice_J_kg": params.L_ice,
        }
    )

    return daily, fluxes
