import numpy as np
import xarray as xr

from .config import HeatBudgetParameters


SECONDS_PER_DAY = 86400.0


def calculate_residual(
    daily: xr.Dataset,
    fluxes: xr.Dataset,
    raw: dict,
    p: HeatBudgetParameters,
) -> xr.Dataset:
    """
    Calculate melt terms and residual heat flux.

    This preserves the calculation used in the original 2024
    residual analysis while returning daily output.
    """

    if "snow_ice_smoothed" not in raw:
        raise KeyError(
            "Residual calculation requires "
            "SIMBA/snow-ice-smoothed.pickle."
        )

    # ========================================================
    # 1. SNOW MELT
    # ========================================================

    # Change in snow-air interface
    dH_snow_loss = daily["snow_air"].diff("time").where(
        daily["H_snow"].diff("time") < 0,
        0,
    )

    # Fraction of snow loss attributed to melt.
    # Original assumption:
    # if wind >= 7.7 m/s, only 12% of snow loss is melt.
    melt_fraction = xr.where(
        daily["wind_10m_max"]
        >= p.wind_transport_threshold,
        p.snow_melt_fraction_during_transport,
        1.0,
    )

    # --------------------------------------------------------
    # Reproduce original time alignment
    # --------------------------------------------------------
    #
    # Original code used:
    #
    # dH_snow_loss.where(
    #     F_net_surf[0:-1].values < 0, 0
    # )
    #
    # So the heat-flux criterion is applied positionally using
    # the preceding F_net_surface timestamp.
    #

    dH_snow_melt = dH_snow_loss.where(
        fluxes["F_net_surface"][:-1].values < 0,
        0,
    )

    # xarray aligns melt_fraction to the timestamps of
    # dH_snow_melt here, matching the original calculation.
    dH_snow_melt = (
        dH_snow_melt
        * melt_fraction
    )

    # --------------------------------------------------------
    # Convert snow loss to heat flux
    # --------------------------------------------------------

    F_snow_melt_calc = (
        p.rho_s
        * p.L_ice
        * dH_snow_melt
        / 86400.0
    )

    # The original code explicitly shifted these values onto
    # snow_air[0:-1]. Reproduce that behavior.
    F_snow_melt = xr.zeros_like(
        daily["snow_air"]
    )

    F_snow_melt[:-1] = (
        F_snow_melt_calc.values
    )


    # ========================================================
    # 2. ICE / SNOW-ICE MELT
    # ========================================================

    # IMPORTANT:
    # Preserve native (6-hourly) snow-ice resolution here.
    start = daily.time.values[0]
    end = daily.time.values[-1] + np.timedelta64(1, "D")

    snow_ice_smoothed = (
        raw["snow_ice_smoothed"]
        .sel(time=slice(start, end))
    )

    dH_ice_melt = (
        snow_ice_smoothed.diff("time")
    )

    dH_ice_melt = dH_ice_melt.where(
        dH_ice_melt < 0,
        0,
    )

    # Original snow-ice timestep = 6 hours
    dt_ice = 6 * 60 * 60

    F_ice_melt = (
        p.rho_snow_ice
        * p.L_ice
        * dH_ice_melt
        / dt_ice
    )

    # THEN average the flux to daily.
    # This is intentionally different from differentiating
    # daily-mean snow-ice thickness.
    F_ice_melt = (
        F_ice_melt
        .resample(time="1D")
        .mean("time")
    )

    # Reproduce original 2024 criterion exactly.
    F_ice_melt = F_ice_melt.where(
        daily["H_snow"] == 0,
        0,
    )

    # Make sure it uses the common daily coordinate.
    F_ice_melt = F_ice_melt.reindex(
        time=daily.time
    )


    # ========================================================
    # 3. TOTAL MELT
    # ========================================================

    F_melt = (
        F_snow_melt
        + F_ice_melt
    )


    # ========================================================
    # 4. RESIDUAL
    # ========================================================

    residual = (
        fluxes["F_net_surface"]
        - F_ice_melt
        - F_snow_melt
    )


    # ========================================================
    # 5. ROLLING RESIDUAL
    # ========================================================

    # Preserve the exact structure of the original calculation
    # rather than rolling the already-calculated residual.
    window = p.residual_rolling_days

    residual_roll = (
        fluxes["F_net_surface"]
        .rolling(
            time=window,
            center=True,
        )
        .mean()

        -

        F_ice_melt
        .rolling(
            time=window,
            center=True,
        )
        .mean()

        -

        F_snow_melt
        .rolling(
            time=window,
            center=True,
        )
        .mean()
    )


    return xr.Dataset(
        {
            "F_snow_melt": F_snow_melt,
            "F_ice_melt": F_ice_melt,
            "F_melt": F_melt,
            "residual": residual,
            "residual_roll": residual_roll,
        }
    )

    
def summarize_rain_residual(
    daily: xr.Dataset,
    fluxes: xr.Dataset,
) -> dict:
    """
    Summarize overlap between rain events and residual heat flux
    that falls outside the surface heat-flux uncertainty.
    """

    if "rain_mm" not in daily:
        raise KeyError(
            "Rain summary requires daily['rain_mm']. "
            "Set rain_file in SeasonConfig."
        )

    # Residual values that fall below the uncertainty envelope
    outside_residual = fluxes["residual_roll"].where(
        fluxes["residual_roll"]
        < -fluxes["sig_F_surface_roll"]
    )

    # Only retain those residuals on rain days
    rain_residual = outside_residual.where(
        daily["rain_mm"] > 0
    )

    # --------------------------------------------------------
    # Fraction of residual exceedance days that coincide with rain
    # --------------------------------------------------------

    n_outside = int(
        outside_residual.count()
    )

    n_rain_outside = int(
        rain_residual.count()
    )

    if n_outside > 0:
        fraction_days = (
            n_rain_outside / n_outside
        )
    else:
        fraction_days = np.nan

    # --------------------------------------------------------
    # Integrated residual energy
    # --------------------------------------------------------

    seconds_per_day = 86400
    joules_to_MJ = 1e6

    residual_energy_MJ_m2 = float(
        -outside_residual.sum(skipna=True)
        * seconds_per_day
        / joules_to_MJ
    )

    rain_residual_energy_MJ_m2 = float(
        -rain_residual.sum(skipna=True)
        * seconds_per_day
        / joules_to_MJ
    )

    if residual_energy_MJ_m2 != 0:
        fraction_energy = (
            rain_residual_energy_MJ_m2
            / residual_energy_MJ_m2
        )
    else:
        fraction_energy = np.nan

    return {
        "n_outside_uncertainty_days":
            n_outside,

        "n_rain_and_outside_days":
            n_rain_outside,

        "fraction_outside_days_with_rain":
            fraction_days,

        "residual_energy_MJ_m2":
            residual_energy_MJ_m2,

        "rain_residual_energy_MJ_m2":
            rain_residual_energy_MJ_m2,

        "fraction_residual_energy_with_rain":
            fraction_energy,
    }