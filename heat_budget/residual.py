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
    """

    # ========================================================
    # 1. SNOW MELT
    # ========================================================

    # --------------------------------------------------------
    # Snow-air interface change
    # --------------------------------------------------------

    dH_snow_air = daily["snow_air"].diff("time")
    dH_snow = daily["H_snow"].diff("time")

    # Only retain periods when total dry-snow thickness decreases
    dH_snow_loss = dH_snow_air.where(
        dH_snow < 0,
        0,
    )

    # --------------------------------------------------------
    # Align other variables explicitly with the snow-loss interval
    # --------------------------------------------------------

    # diff() labels each interval with the later timestamp.
    # For example:
    # Feb 28 = snow_air(Feb 28) - snow_air(Feb 27)

    interval_time = dH_snow_loss.time

    # Surface heat flux corresponding to the preceding day
    F_net_previous = xr.DataArray(
        fluxes["F_net_surface"].isel(time=slice(None, -1)).values,
        coords={"time": interval_time},
        dims=["time"],
    )

    # Only count snow loss as melt when surface energy is toward melt
    dH_snow_melt = dH_snow_loss.where(
        F_net_previous < 0,
        0,
    )

    # --------------------------------------------------------
    # Wind redistribution
    # --------------------------------------------------------

    wind_for_interval = daily["wind_10m_max"].sel(
        time=interval_time
    )

    melt_fraction = xr.where(
        wind_for_interval >= p.wind_transport_threshold,
        p.snow_melt_fraction_during_transport,
        1.0,
    )

    dH_snow_melt = dH_snow_melt * melt_fraction

    # --------------------------------------------------------
    # Convert snow loss to heat flux
    # --------------------------------------------------------

    F_snow_melt_calc = (
        p.rho_s
        * p.L_ice
        * dH_snow_melt
        / 86400.0
    )

    F_snow_melt = xr.zeros_like(
        daily["snow_air"]
    )

    F_snow_melt[:-1] = (
        F_snow_melt_calc.values
    )


    # ========================================================
    # 2. ICE / SNOW-ICE MELT
    # ========================================================

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
    F_ice_melt = (
        F_ice_melt
        .resample(time="1D")
        .mean("time")
    )

    # Only melt ice if the snow depth is 0
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

    # Smooth out the residual
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
