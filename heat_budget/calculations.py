import numpy as np
import xarray as xr

from functions import IMS_toolbox as IMS

from .config import HeatBudgetParameters


def calculate_conduction(ds: xr.Dataset, p: HeatBudgetParameters) -> xr.Dataset:
    """Calculate conductive heat-flux diagnostics on the daily grid."""

    ref_lower = ds["H_bottom"] + p.ref_layer_bottom
    ref_upper = ds["H_bottom"] + p.ref_layer_top

    temp_ice_ref = ds["temp_ice"].where(
        (ds["temp_ice"].z < ref_upper)
        & (ds["temp_ice"].z > ref_lower),
        other=np.nan,
    )

    F_cond_ice_2d = -p.k_i * ds["temp_ice"].differentiate("z")

    F_cond_ref = (
        -p.k_i * temp_ice_ref.differentiate("z")
    ).mean("z")

    temp_ice_snow = ds["temperature"].where(
        (ds["temperature"].z < ds["snow_air"])
        & (ds["temperature"].z > ds["H_bottom"])
    )
    dT_dz_full = temp_ice_snow.differentiate("z")

    surface_lower = ds["snow_air"] - p.surface_layer_depth
    surface_upper = ds["snow_air"]

    dT_dz_surface = dT_dz_full.where(
        (dT_dz_full.z < surface_upper)
        & (dT_dz_full.z > surface_lower),
        other=np.nan,
    )

    k_surface = xr.where(
        ds["H_snow"] < p.snow_cover_threshold,
        p.k_i,
        p.k_s,
    )

    F_cond_surface = (
        -k_surface * dT_dz_surface
    ).mean("z")

    temp_snow = ds["temperature"].where(
        (ds["temperature"].z <= ds["snow_air"])
        & (ds["temperature"].z >= ds["snow_ice"])
    )
    F_cond_snow_2d = -p.k_s * temp_snow.differentiate("z")

    return xr.Dataset(
        {
            # Keep only quantities needed by figures or later calculations.
            "F_cond_ice_2d": F_cond_ice_2d,
            "F_cond_snow_2d": F_cond_snow_2d,
            "F_cond_ref": F_cond_ref,
            "F_cond_surface": F_cond_surface,
            "dT_dz_surface": dT_dz_surface,
        }
    )


def calculate_radiation(ds: xr.Dataset, p: HeatBudgetParameters) -> xr.Dataset:
    """Calculate albedo and radiative surface-flux terms."""

    albedo = xr.where(
        ds["SW_in"] > 0,
        ds["SW_out"] / ds["SW_in"],
        np.nan,
    )

    I0_surface = xr.where(
        ds["H_snow"] > p.snow_cover_threshold,
        0.0,
        p.I0,
    )

    F_sw = (
        -(1.0 - albedo)
        * (1.0 - I0_surface)
        * ds["SW_in"]
    )

    F_lw = ds["F_lw_measured"].where(
        ds["F_lw_measured"] > 0,
        0,
    )

    return xr.Dataset(
        {
            "albedo": albedo,
            "F_sw": F_sw,
            "F_lw": F_lw,
        }
    )


def calculate_turbulent_fluxes(
    ds: xr.Dataset,
    p: HeatBudgetParameters,
) -> xr.Dataset:
    """Calculate sensible and latent heat fluxes."""

    sig_Ta = xr.zeros_like(ds["T_air"]) + p.sig_Ta
    sig_wind = ds["wind_10m"] * p.relative_sig_wind

    sig_Tsurface_relative = np.sqrt(
        p.relative_sig_radiation**2
        + p.relative_sig_radiation**2
        + (p.sig_emissivity / p.epsilon) ** 2
    )
    sig_Tsurface = ds["T_surface"] * sig_Tsurface_relative

    F_sens, sig_Fsens = IMS.sensible_heat_flux(
        ds["wind_10m"],
        ds["T_surface"],
        ds["T_air"],
        sig_wind,
        sig_Tsurface,
        sig_Ta,
    )

    F_lat = IMS.latent_heat_flux(
        ds["wind_10m"],
        ds["RH"],
        ds["pressure"],
        ds["T_surface"],
        ds["T_air"],
    )

    return xr.Dataset(
        {
            "F_sens": F_sens,
            "F_lat": F_lat,
            "sig_Fsens": sig_Fsens,
            "sig_Tsurface": sig_Tsurface,
            "sig_Ta": sig_Ta,
            "sig_wind_10m": sig_wind,
        }
    )


def calculate_surface_balance(
    conduction: xr.Dataset,
    radiation: xr.Dataset,
    turbulent: xr.Dataset,
) -> xr.Dataset:
    """Combine the daily surface heat-flux components."""

    F_surface = (
        radiation["F_lw"]
        + radiation["F_sw"]
        + turbulent["F_sens"]
        + turbulent["F_lat"]
    )

    F_net_surface = F_surface - conduction["F_cond_surface"]

    return xr.Dataset(
        {
            "F_surface": F_surface,
            "F_net_surface": F_net_surface,
        }
    )


def calculate_basal_balance(
    ds: xr.Dataset,
    conduction: xr.Dataset,
    p: HeatBudgetParameters,
) -> xr.Dataset:
    """Calculate the daily basal phase-change term and ocean heat flux."""

    dH_bottom_dt = ds["H_bottom"].differentiate(
        "time",
        datetime_unit="s",
    )

    F_phase_change_basal = (
        p.L_ice
        * p.rho_i
        * dH_bottom_dt
    )

    # Matches the active equation in the original 2024 script.
    F_w = conduction["F_cond_ref"] + F_phase_change_basal

    return xr.Dataset(
        {
            "F_phase_change_basal": F_phase_change_basal,
            "F_w": F_w,
        }
    )


def calculate_uncertainties(
    ds: xr.Dataset,
    conduction: xr.Dataset,
    radiation: xr.Dataset,
    turbulent: xr.Dataset,
    p: HeatBudgetParameters,
) -> xr.Dataset:
    """Calculate uncertainty of the net surface heat flux."""

    sig_RH = ds["RH"] * p.relative_sig_RH
    sig_pressure = ds["pressure"] * p.relative_sig_pressure

    sig_netLW = np.abs(radiation["F_lw"]) * p.relative_sig_radiation
    sig_netSW = np.abs(radiation["F_sw"]) * p.relative_sig_radiation

    sig_Flat_relative = np.sqrt(
        (turbulent["sig_wind_10m"] / ds["wind_10m"]) ** 2
        + (sig_RH / ds["RH"]) ** 2
        + (sig_pressure / ds["pressure"]) ** 2
        + (turbulent["sig_Tsurface"] / ds["T_surface"]) ** 2
        + (turbulent["sig_Ta"] / ds["T_air"]) ** 2
    )
    sig_Flat = np.abs(sig_Flat_relative * turbulent["F_lat"])

    sig_dT_avg = (
        np.sqrt(
            p.sig_temperature_sensor**2
            + p.sig_temperature_sensor**2
        )
        / np.sqrt(p.n_surface_temperature_sensors)
    )

    dT = (conduction["dT_dz_surface"] * p.dz).mean("z")

    sig_Fcond_relative = np.sqrt(
        (p.sig_k / p.k_s) ** 2
        + (sig_dT_avg / dT) ** 2
        + (p.sig_dz / p.dz) ** 2
    )
    sig_Fcond = np.abs(
        sig_Fcond_relative * conduction["F_cond_surface"]
    )

    sig_F_surface = np.sqrt(
        sig_netLW**2
        + sig_netSW**2
        + turbulent["sig_Fsens"]**2
        + sig_Flat**2
        + sig_Fcond**2
    )

    sig_F_surface_roll = sig_F_surface.rolling(
        time=p.uncertainty_rolling_days,
        center=True,
    ).mean()

    return xr.Dataset(
        {
            "sig_F_surface": sig_F_surface,
            "sig_F_surface_roll": sig_F_surface_roll,
        }
    )
