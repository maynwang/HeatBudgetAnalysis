"""Generate figures for the 2024 Kaipokok Bay heat-budget paper.

This script replaces the original notebook-style plotting workflow with explicit
file loading and one function per figure. It is designed for a publication
repository: running this file does not require variables to have been created in
an interactive namespace first.

The script uses the current processed SIMBA product produced by
``run_simba_interface_2024.py``:

    data/2024/SIMBA/processed/SIMBA_interfaces_2024.nc

That file stores native-resolution variables on ``time_6h`` and daily variables
on ``time_daily``. They are normalized to a local ``time`` dimension when loaded.

Heat-budget plotting expects a NetCDF containing the final quantities used in
Figures 6 and 7. Canonical variable names are listed in ``HEAT_BUDGET_ALIASES``.
Legacy names from the original analysis are also accepted, which makes it easier
to transition the plotting workflow while preserving the published calculation.

Figures represented in the original plotting script:
    Figure 2       SIMBA temperature and interfaces
    Figure 3       Atmospheric conditions
    Figure 3 inset Wind-direction compass legend
    Figure 4a-d    Ocean time series
    Figure 4e-g    Weekly CTD profiles and T-S diagram
    Figure 6       Surface, conductive, and basal heat fluxes
    Figure 7       Surface heat-budget residual
    Figure 8       Simple-model results from unified scenario NetCDF
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cmocean
import colormaps as cmaps
import gsw
import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from functions import plotting as pl

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class FigureConfig:
    root: Path = Path(".")
    year: int = 2024
    start: str = "2024-01-26"
    end: str = "2024-04-15"
    fontsize: int = 12
    dpi: int = 300
    output_dir: Path = Path("figures/paper")

    @property
    def year_dir(self) -> Path:
        return self.root / "data" / str(self.year)

    @property
    def simba_file(self) -> Path:
        return self.year_dir / "SIMBA" / "processed" / f"SIMBA_interfaces_{self.year}.nc"

    @property
    def weekly_ice_file(self) -> Path:
        return self.year_dir / "SiteVisits" / "SiteVisits_Weekly_IceSnowWater.nc"

    @property
    def weather_file(self) -> Path:
        return self.year_dir / "WeatherStation" / "WeatherVars.nc"

    @property
    def era5_climatology_file(self) -> Path:
        return self.root / "data" / "ERA5" / "ERA5_t2m_climatology_daily_1979-2021.nc"

    @property
    def era5_quantiles_file(self) -> Path:
        return self.root / "data" / "ERA5" / "ERA5_t2m_clim_quantiles.nc"

    @property
    def ocean_file(self) -> Path:
        return self.year_dir / "RBR_IMS" / f"PlottingVars_{self.year}.nc"

    @property
    def tcm_hourly_file(self) -> Path:
        return self.year_dir / "TCM-1" / "TCM_currents_hourly.nc"

    @property
    def tcm_detided_file(self) -> Path:
        return self.year_dir / "TCM-1" / "TCM_currents_detided.nc"

    @property
    def ctd_file(self) -> Path:
        return self.year_dir / "CTD_profiles" / "CTD_IMS_SiteVisits.nc"

    @property
    def heat_budget_inputs_file(self) -> Path:
        return (
            self.year_dir
            / "HeatBudget"
            / "processed"
            / f"HeatBudget_inputs_{self.year}.nc"
        )

    @property
    def heat_budget_fluxes_file(self) -> Path:
        return (
            self.year_dir
            / "HeatBudget"
            / "processed"
            / f"HeatBudget_fluxes_{self.year}.nc"
        )

    @property
    def model_scenarios_file(self) -> Path:
        return (
            self.year_dir
            / "Model"
            / "processed"
            / f"model_scenarios_{self.year}.nc"
        )

# Canonical new name -> accepted names. The legacy aliases are intentionally
# retained so the plotting script can read an existing NetCDF exported from the
# original variables while the processing package is being standardized.
HEAT_BUDGET_ALIASES = {
    # Current refactored names are listed first; historical names remain as fallbacks.
    "F_net_surface": ("F_net_surface", "F_net_surf"),
    "F_net_shortwave": ("F_sw", "F_net_shortwave", "F_sw_net"),
    "F_net_longwave": ("F_lw", "F_net_longwave", "F_lw_net_clean"),
    "F_sensible": ("F_sens", "F_sensible"),
    "F_latent": ("F_lat", "F_latent"),
    "F_conductive_surface": ("F_cond_surface", "F_conductive_surface", "F_c_surf"),
    "F_conductive_ice_2d": ("F_cond_ice_2d", "F_conductive_ice_2d", "F_c_2d_daily"),
    "F_conductive_snow_2d": ("F_cond_snow_2d", "F_conductive_snow_2d", "F_c_snow"),
    "reference_layer_lower": ("reference_layer_lower", "ref_layer_lower"),
    "reference_layer_upper": ("reference_layer_upper", "ref_layer_upper"),
    "surface_layer_lower": ("surface_layer_lower", "surf_layer_lower"),
    "surface_layer_upper": ("surface_layer_upper", "surf_layer_upper"),
    "F_growth": ("F_phase_change_basal", "F_growth", "F_l"),
    "F_conductive_bottom": ("F_cond_ref", "F_conductive_bottom", "F_c_daily"),
    "F_ocean": ("F_w", "F_ocean"),
    "F_snow_melt": ("F_snow_melt", "F_snowmelt"),
    "F_ice_melt": ("F_ice_melt", "F_icemelt"),
    "surface_residual": ("residual", "surface_residual", "DIFF"),
    "surface_residual_smoothed": ("residual_roll", "surface_residual_smoothed", "DIFF_roll"),
    "surface_uncertainty_smoothed": ("sig_F_surface_roll", "surface_uncertainty_smoothed", "sig_F_surf_roll"),
    "rain_mm": ("rain_mm", "rain"),
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required figure input not found: {path}")
    return path


def _rename_native_time(da: xr.DataArray) -> xr.DataArray:
    """Normalize a SIMBA DataArray's native time dimension to ``time``."""
    if "time_6h" in da.dims:
        return da.rename({"time_6h": "time"})
    if "time_daily" in da.dims:
        return da.rename({"time_daily": "time"})
    return da


def _get_first(ds: xr.Dataset, names: Iterable[str], label: str | None = None) -> xr.DataArray:
    for name in names:
        if name in ds:
            return ds[name]
    shown = label or "/".join(names)
    raise KeyError(f"Could not find '{shown}' in {list(ds.data_vars)}")


def hb_var(ds: xr.Dataset, canonical_name: str) -> xr.DataArray:
    return _get_first(ds, HEAT_BUDGET_ALIASES[canonical_name], canonical_name)


def weekly_var(ds: xr.Dataset, *names: str) -> xr.DataArray:
    return _get_first(ds, names)


def save_figure(fig: mpl.figure.Figure, path: Path, dpi: int, transparent: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    print(f"Saved {path}")


def custom_cmap_trimmed(orig_cmap, skew: float, vmin: float, vmax: float):
    """Reproduce the trimmed/skewed colormap used for CTD depth."""
    positions = vmin + (vmax - vmin) * (np.linspace(0, 1, 256) ** skew)
    return ListedColormap(orig_cmap(positions))


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------


def load_simba_products(cfg: FigureConfig) -> dict[str, xr.DataArray]:
    ds = xr.open_dataset(_require_file(cfg.simba_file))
    variables = [
        "temperature",
        "temp_ice",
        "H_ice",
        "H_bottom",
        "snow_air",
        "snow_ice",
        "H_snow",
        "snow_ice_smoothed",
    ]
    return {name: _rename_native_time(ds[name]) for name in variables if name in ds}


def load_weather(cfg: FigureConfig) -> xr.Dataset:
    return xr.open_dataset(_require_file(cfg.weather_file)).sel(time=slice(cfg.start, cfg.end))


def load_heat_budget(cfg: FigureConfig) -> xr.Dataset:
    """Load and combine the current heat-budget outputs for plotting."""

    inputs = xr.open_dataset(
        _require_file(cfg.heat_budget_inputs_file)
    )

    fluxes = xr.open_dataset(
        _require_file(cfg.heat_budget_fluxes_file)
    )

    return xr.merge(
        [inputs, fluxes],
        compat="override",
        join="outer",
    )


# -----------------------------------------------------------------------------
# Figure 2: SIMBA temperature and interfaces
# -----------------------------------------------------------------------------


def figure_2_simba(
    cfg: FigureConfig,
    simba: dict[str, xr.DataArray],
    weekly: xr.Dataset,
):

    temperature = simba["temperature"].sel(time=slice(cfg.start, cfg.end))
    snow_air = simba["snow_air"].sel(time=slice(cfg.start, cfg.end))
    snow_ice_smoothed = simba["snow_ice_smoothed"].sel(time=slice(cfg.start, cfg.end))
    H_bottom = simba["H_bottom"].sel(time=slice(cfg.start, cfg.end))

    # Original Figure 2 colormap
    orig = cmaps.WhBlGrYeRe

    cmap = pl.custom_cmap_trimmed(
        orig,
        skew=3,
        vmin=0.2,
        vmax=1,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        facecolor="white",
    )

    im = ax.pcolormesh(
        temperature.time.values,
        temperature.z,
        temperature.transpose("z", "time"),
        cmap=cmap,
        alpha=0.8,
        vmin=-15,
        vmax=0,
        shading="auto",
    )
    ax.set_ylim(-2.9, 1.7)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, extend="both")
    cbar.set_label(r"Temperature ($^\circ$C)", size=cfg.fontsize)
    cbar.ax.tick_params(labelsize=cfg.fontsize)

    ax.plot(snow_air.time, snow_air, c="gainsboro", label="Air-snow")

    hi = weekly_var(weekly, "hi", "Hice")
    hs = weekly_var(weekly, "hs_snowStake_mean", "Hsnow")
    ax.plot(weekly.time[1:], -hi[1:], "o", c="blue", markeredgecolor="k", clip_on=False,
            label="Weekly ice thickness")
    ax.plot(snow_ice_smoothed.time, snow_ice_smoothed, c="k", label="Snow-ice")
    ax.plot(weekly.time[1:], hs[1:], "^", c="white", markeredgecolor="k", clip_on=False,
            label="Weekly snow depth")
    ax.plot(H_bottom.time, H_bottom, c="b", label="Ice-water")

    ax.axhline(0, color="k", linestyle="--")
    ax.set_ylabel(r"$z$ (m)", fontsize=cfg.fontsize)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=cfg.fontsize)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(bbox_to_anchor=(0, 1.15), loc="upper left", fontsize=cfg.fontsize, ncol=3)
    ax.set_xlim(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end) + pd.Timedelta(days=1))

    event_dates = pd.to_datetime([
        "2024-02-13", "2024-02-19", "2024-02-29", "2024-03-09",
        "2024-03-23", "2024-03-27", "2024-04-08",
    ])
    ax.xaxis.set_minor_formatter(ax.xaxis.get_major_formatter())
    ax.set_xticks(event_dates, minor=True)
    ax.tick_params(
        axis="x", which="minor", bottom=True, top=False, direction="in",
        length=6, width=1, labelbottom=True, pad=-32, rotation=35,
        labelsize=cfg.fontsize,
    )

    save_figure(fig, cfg.output_dir / "Fig2_SIMBAIceSnow.png", cfg.dpi)


# -----------------------------------------------------------------------------
# Figure 3: atmospheric summary + wind compass
# -----------------------------------------------------------------------------


def _prepare_era5_climatology(cfg: FigureConfig):
    clim = xr.open_dataset(_require_file(cfg.era5_climatology_file))
    quant = xr.open_dataset(_require_file(cfg.era5_quantiles_file))

    if "dayofyear" in quant.dims or "dayofyear" in quant.coords:
        quant = quant.assign_coords(
            dayofyear=pd.date_range(f"{cfg.year}-01-01", periods=quant.sizes["dayofyear"])
        ).rename(dayofyear="time")
    if "time" not in clim.coords or not np.issubdtype(clim.time.dtype, np.datetime64):
        clim = clim.assign_coords(time=pd.date_range(f"{cfg.year}-01-01", periods=clim.sizes["time"], freq="D"))
    else:
        # Climatologies are often stored with an arbitrary source year. Re-map
        # by day number to the analysis year so plotting aligns with 2024.
        if len(clim.time) in (365, 366):
            clim = clim.assign_coords(time=pd.date_range(f"{cfg.year}-01-01", periods=len(clim.time), freq="D"))

    quant = quant.sel(time=slice(cfg.start, cfg.end)) - 273.15
    clim = clim.sel(time=slice(cfg.start, cfg.end)) - 273.15
    return clim, quant.sel(quantile=0.2), quant.sel(quantile=0.8)


def figure_3_atmosphere(cfg: FigureConfig, weather: xr.Dataset):
    clim, p20, p80 = _prepare_era5_climatology(cfg)
    hourly = weather.resample(time="1h").mean()

    wind_speed = hourly["WS_ms_Avg"]
    wind_dir = (hourly["WindDir"] + 180) % 360

    sw_in = weather["SWUpper_Avg"].where(weather["SWUpper_Avg"] > 0, 0).resample(time="1D").mean()
    sw_out = weather["SWLower_Avg"].where(weather["SWLower_Avg"] > 0, 0).resample(time="1D").mean()
    albedo = (sw_out / sw_in).where(sw_in > 0)

    fig, axes = plt.subplots(7, 1, figsize=(10, 10), sharex=True, facecolor="w", layout="constrained")
    t = hourly.time.values

    norm = mcolors.Normalize(vmin=0, vmax=360)
    cmap_wind = plt.get_cmap("twilight_shifted")
    t_num = mdates.date2num(t)
    points = np.array([t_num, wind_speed.values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap_wind, norm=norm, linewidth=2)
    lc.set_array(wind_dir.values[:-1])
    axes[0].add_collection(lc)
    axes[0].set_xlim(t.min(), t.max())
    axes[0].set_ylim(float(wind_speed.min()), float(wind_speed.max()) + 2)
    axes[0].set_ylabel(r"m s$^{-1}$")
    axes[0].set_title("Wind velocity")
    axes[0].yaxis.set_major_locator(MultipleLocator(5))

    axes[1].plot(t, hourly["BP_mbar_Avg"], "black")
    axes[1].set_ylabel("mbar")
    axes[1].set_title("Barometric pressure")

    t2m_name = "t2m"
    axes[2].fill_between(p20.time.values, p20[t2m_name], p80[t2m_name], alpha=0.3, color="grey")
    axes[2].plot(clim.time, clim[t2m_name], c="grey", alpha=0.8, label="30-year mean")
    axes[2].plot(hourly.time, hourly["AirT_C_Avg"], "m", label="IMS 2024")
    axes[2].set_ylabel(r"$^\circ$C")
    axes[2].set_title("Air temperature")
    axes[2].set_yticks([-30, -20, -10, 0])
    axes[2].legend(loc="lower right")

    axes[3].plot(hourly.time, hourly["RH"], "black")
    axes[3].set_ylabel("%")
    axes[3].set_title("Relative humidity")

    axes[4].plot(t, abs(hourly["SWUpper_Avg"]), "orange", alpha=0.5)
    axes[4].plot(t, abs(hourly["LWUpperCo_Avg"]), "red")
    axes[4].legend(["Incoming SWR", "Incoming LWR"], loc="upper left")
    axes[4].set_ylabel(r"W m$^{-2}$")
    axes[4].set_title("Incoming radiation")

    axes[5].plot(hourly.time, hourly["RsNet_Avg"], "peru")
    axes[5].plot(hourly.time, hourly["RlNet_Avg"], "firebrick")
    axes[5].legend(["Net SWR", "Net LWR"], loc="upper left")
    axes[5].set_ylabel(r"W m$^{-2}$")
    axes[5].set_title("Net radiation")

    axes[6].plot(albedo.time, albedo, c="navy")
    axes[6].set_title("Albedo")
    axes[6].set_ylim(0, 1)
    axes[6].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    for i, ax in enumerate(axes):
        ax.text(0.01, 1.15, f"({chr(97 + i)})", fontsize=cfg.fontsize, fontweight="bold",
                transform=ax.transAxes, verticalalignment="top")
        ax.tick_params(axis="both", labelsize=cfg.fontsize)
        ax.grid()
    axes[-1].set_xlim(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))

    save_figure(fig, cfg.output_dir / "Fig3_AtmosphereSummary.png", 600)

    # Separate compass legend 
    fig_leg, ax_leg = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})
    theta = np.linspace(0, 2 * np.pi, 360)
    compass_colors = cmap_wind(norm(np.linspace(0, 360, 360)))
    ax_leg.bar(theta, np.ones_like(theta), width=0.05, color=compass_colors, edgecolor="none")
    ax_leg.set_theta_offset(np.pi / 2)
    ax_leg.set_theta_direction(-1)
    ax_leg.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax_leg.set_xticklabels(["N", "E", "S", "W"], fontsize=36)
    ax_leg.tick_params(axis="both", which="major", pad=15)
    ax_leg.set_yticklabels([])
    ax_leg.set_yticks([])
    save_figure(fig_leg, cfg.output_dir / "Fig3_WindDirectionLegend.png", 600, transparent=True)


# -----------------------------------------------------------------------------
# Figure 4: ocean observations
# -----------------------------------------------------------------------------


def figure_4_ocean(cfg: FigureConfig):
    ocean = xr.open_dataset(_require_file(cfg.ocean_file))
    tcm_hourly = xr.open_dataset(_require_file(cfg.tcm_hourly_file)).sel(time=slice(cfg.start, cfg.end))
    tcm_detided = xr.open_dataset(_require_file(cfg.tcm_detided_file)).sel(time=slice(cfg.start, cfg.end))

    T_depths = ocean["T_depths_detided"].sel(time=slice(cfg.start, cfg.end))
    SA = ocean["SA_surf_detided"].sel(depth=[-1.88, -2.38], time=slice(cfg.start, cfg.end))
    CT = ocean["CT_surf_detided"].sel(depth=[-1.88, -2.38], time=slice(cfg.start, cfg.end))

    sensors_deep = T_depths.sel(depth=slice(-56, -8.94)).depth
    sensors_shallow = T_depths.sel(depth=slice(-8.94, -1.88)).depth
    depths = T_depths.depth
    t = T_depths.time.values

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(5, 1, hspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)

    ax0.plot(tcm_hourly.time, tcm_hourly.sel(z=1.5)["Speed"], c="navy", alpha=0.5)
    ax0.plot(tcm_hourly.time, tcm_hourly.sel(z=2)["Speed"], c="royalblue", alpha=0.5)
    ax0.plot(tcm_detided.time, tcm_detided.sel(z=1.5)["Speed"], label="-1.5 m", c="royalblue")
    ax0.plot(tcm_detided.time, tcm_detided.sel(z=2)["Speed"], label="-2 m", c="navy")
    ax0.legend(loc="upper left")
    ax0.set_ylabel("m/s")
    ax0.set_title("Current speed (near-surface)")

    ax1.plot(SA.time, SA.sel(depth=-1.88), label="-1.88 m", c="darkseagreen")
    ax1.plot(SA.time, SA.sel(depth=-2.38), label="-2.38 m", c="forestgreen")
    ax1.legend(loc="lower left")
    ax1.set_ylabel("g/kg")
    ax1.set_title("Absolute salinity (near-surface)")

    ax2.plot(CT.time, CT.sel(depth=-1.88), label="-1.88 m", c="indianred")
    ax2.plot(CT.time, CT.sel(depth=-2.38), label="-2.38 m", c="darkred")
    ax2.legend(loc="upper left")
    ax2.set_ylabel(r"$^\circ$C")
    ax2.set_title("Conservative temperature (near-surface)")

    levels = np.linspace(-1.8, -1.4, 9)
    shallow = T_depths.sel(depth=slice(-8.94, -1.88))
    deep = T_depths.sel(depth=slice(float(depths.min()), -8.94))
    im3 = ax3.contourf(shallow.time.values, shallow.depth, shallow.transpose("depth", "time"),
                       levels=levels, cmap=cmocean.cm.thermal)
    ax3.plot(np.repeat(t[0], len(sensors_shallow)), sensors_shallow, c="white", marker="o",
             markersize=7, markeredgecolor="black", linestyle="none", clip_on=False)
    ax3.set_title("Temperature (full depth)")
    ax3.set_ylim(-10, -2)
    ax3.set_yticks([-2, -4, -6, -8])

    im4 = ax4.contourf(deep.time.values, deep.depth, deep.transpose("depth", "time"),
                       levels=levels, cmap=cmocean.cm.thermal)
    ax4.plot(np.repeat(t[0], len(sensors_deep)), sensors_deep, c="white", marker="o",
             markersize=7, markeredgecolor="black", linestyle="none", clip_on=False)
    ax4.set_ylabel(r"$z$ (m)")
    ax4.set_ylim(float(depths.min()), -10)
    ax4.axhline(-10, c="white", linestyle="--")

    for i, ax in enumerate([ax0, ax1, ax2, ax3]):
        ax.text(0.01, 1.16, f"({chr(97 + i)})", fontweight="bold", transform=ax.transAxes,
                verticalalignment="top")
    for ax in [ax0, ax1, ax2]:
        ax.grid()
    for ax in [ax0, ax1, ax2, ax3]:
        ax.tick_params(axis="x",labelbottom=False)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax4.set_xlabel("")
    ax0.set_xlim(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))
    cbar = fig.colorbar(im4, ax=[ax3, ax4], pad=0.02, format=mpl.ticker.FormatStrFormatter("%.2f"))
    cbar.set_label(r"Temperature ($^\circ$C)", size=cfg.fontsize)

    save_figure(fig, cfg.output_dir / "Fig4_OceanSummary.png", 600)


def figure_4_ctd(cfg: FigureConfig):
    ctd = xr.open_dataset(_require_file(cfg.ctd_file))
    T_range = np.linspace(-2, 1)
    S_range = np.linspace(0, 32)
    Tg, Sg = np.meshgrid(T_range, S_range)
    sigma_theta = gsw.density.sigma0(Sg, Tg)

    profile_colors = cmaps.cool_dark(np.linspace(0, 1, len(ctd.date)))
    variables = ["Conservative_Temperature", "Absolute_Salinity"]
    xlabels = [r"Temperature ($^\circ$C)", "Salinity (g/kg)"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), gridspec_kw={"width_ratios": [1, 1, 1]},
                             constrained_layout=True)
    for i, ax in enumerate(axes[:2]):
        ax.set_prop_cycle("color", list(profile_colors))
        for date in ctd.date:
            profile = ctd.sel(date=date)
            ax.plot(profile[variables[i]], -profile["Depth"])
        ax.set_xlabel(xlabels[i])
        ax.set_ylim(-10, 0)
    axes[0].set_ylabel(r"$z$ (m)")
    axes[1].set_yticklabels([])

    cmap_dates = ListedColormap(profile_colors)
    bounds = np.linspace(0, 1, len(ctd.date) + 1)
    norm = BoundaryNorm(bounds, cmap_dates.N)
    sm = plt.cm.ScalarMappable(cmap=cmap_dates, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=1, pad=0.04, boundaries=bounds,
                        ticks=0.5 * (bounds[1:] + bounds[:-1]))
    cbar.set_ticklabels([pd.to_datetime(str(t)).strftime("%m-%d") for t in ctd.date.values])
    for i, label in enumerate(cbar.ax.yaxis.get_ticklabels()):
        if i % 2 != 0:
            label.set_visible(False)

    depth_cmap = custom_cmap_trimmed(cmaps.berlin, skew=1.4, vmin=0, vmax=0.8)
    sc = axes[2].scatter(ctd["Absolute_Salinity"], ctd["Conservative_Temperature"],
                         c=ctd["Depth"], marker=".", cmap=depth_cmap, vmin=0, vmax=8, zorder=100)
    cs = axes[2].contour(Sg, Tg, sigma_theta, levels=[0, 6, 12, 18, 24], colors="lightgrey", zorder=0)
    axes[2].clabel(cs, fontsize=10, inline=True, fmt="%.1f")
    cbar2 = fig.colorbar(sc, ax=axes[2], fraction=1, pad=0.04, extend="max")
    cbar2.set_label("Depth (m)", size=cfg.fontsize)
    axes[2].set_ylabel(r"Temperature ($^\circ$C)")
    axes[2].set_xlabel("Salinity (g/kg)")
    axes[2].set_ylim(-2, 0)
    axes[2].set_xlim(0, 34)

    for label, ax in zip(["(e)", "(f)", "(g)"], axes):
        ax.text(0.01, 1.09, label, fontweight="bold", transform=ax.transAxes, verticalalignment="top")
        ax.tick_params(axis="both", labelsize=cfg.fontsize)

    save_figure(fig, cfg.output_dir / "Fig4_CTDProfiles.png", 600)


# -----------------------------------------------------------------------------
# Figure 6: heat fluxes
# -----------------------------------------------------------------------------


def figure_6_heat_fluxes(cfg: FigureConfig, simba: dict[str, xr.DataArray], hb: xr.Dataset):
    F_net = hb_var(hb, "F_net_surface")
    F_sw = hb_var(hb, "F_net_shortwave")
    F_lw = hb_var(hb, "F_net_longwave")
    F_sens = hb_var(hb, "F_sensible")
    F_lat = hb_var(hb, "F_latent")
    F_c_surf = hb_var(hb, "F_conductive_surface")
    F_c_2d = hb_var(hb, "F_conductive_ice_2d")
    F_c_snow = hb_var(hb, "F_conductive_snow_2d")
    # These plotting bounds are derived diagnostics. Older workflows kept them
    # only in memory, so reconstruct them if they were not saved.
    if any(name in hb for name in HEAT_BUDGET_ALIASES["reference_layer_lower"]):
        ref_lower = hb_var(hb, "reference_layer_lower")
        ref_upper = hb_var(hb, "reference_layer_upper")
    else:
        H_bottom_daily = simba["H_bottom"].resample(time="1D").mean()
        ref_lower = H_bottom_daily + 0.08
        ref_upper = H_bottom_daily + 0.20

    if any(name in hb for name in HEAT_BUDGET_ALIASES["surface_layer_lower"]):
        surf_lower = hb_var(hb, "surface_layer_lower")
        surf_upper = hb_var(hb, "surface_layer_upper")
    else:
        snow_air_daily = simba["snow_air"].resample(time="1D").mean()
        surf_lower = snow_air_daily - 0.10
        surf_upper = snow_air_daily
    F_growth = hb_var(hb, "F_growth").resample(time="1D").mean()
    F_c_bottom = hb_var(hb, "F_conductive_bottom").resample(time="1D").mean()
    F_ocean = hb_var(hb, "F_ocean").resample(time="1D").mean()

    snow_ice = simba["snow_ice"].sel(time=slice(cfg.start, cfg.end))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, facecolor="w", layout="constrained")
    t = F_net.time

    axes[0].axhline(0, c="dimgrey", alpha=0.8)
    axes[0].plot(t, F_sw, color="peru", label=r"$F_\mathrm{netSWR}$")
    axes[0].plot(t, F_lw, c="firebrick", label=r"$F_\mathrm{netLWR}$")
    axes[0].plot(t, F_sens, c="forestgreen", label=r"$F_\mathrm{SH}$")
    axes[0].plot(t, F_c_surf, c="blue", label=r"$F_\mathrm{c,s}$")
    axes[0].plot(t, F_lat, c="violet", label=r"$F_\mathrm{LH}$")
    axes[0].plot(t, F_net, c="black", linewidth=1.5, label=r"$F_\mathrm{net,s}$")
    axes[0].set_ylim(-130, 100)
    axes[0].set_ylabel(r"W m$^{-2}$")
    axes[0].legend(bbox_to_anchor=(1.001, 1.001), fontsize=cfg.fontsize)
    axes[0].grid()

    im = axes[1].pcolormesh(F_c_2d.time.values, F_c_2d.z, F_c_2d.transpose("z", "time"),
                            cmap="RdBu_r", vmin=-40, vmax=40, shading="auto")
    axes[1].pcolormesh(F_c_snow.time.values, F_c_snow.z, F_c_snow.transpose("z", "time"),
                       cmap="RdBu_r", vmin=-40, vmax=40, shading="auto")
    axes[1].plot(snow_ice.time, snow_ice, c="k", linewidth=1.5)
    axes[1].plot(ref_lower.time, ref_lower, "--", c="grey")
    axes[1].plot(ref_upper.time, ref_upper, "--", c="grey")
    axes[1].plot(surf_lower.time, surf_lower, "--", c="grey")
    axes[1].plot(surf_upper.time, surf_upper, "--", c="grey")
    axes[1].set_ylabel(r"$z$ (m)")
    axes[1].set_ylim(-0.75, 1)
    cbar = fig.colorbar(im, ax=axes[1], pad=0.01)
    cbar.set_label(r"$F_\mathrm{c}$ (W m$^{-2}$)")

    basal_start = "2024-02-02"
    F_growth = F_growth.sel(time=slice(basal_start, None))
    F_c_bottom = F_c_bottom.sel(time=slice(basal_start, None))
    F_ocean = F_ocean.sel(time=slice(basal_start, None))
    axes[2].plot(F_growth.time, F_growth, c="brown", label=r"$F_\mathrm{g}$")
    axes[2].plot(F_c_bottom.time, F_c_bottom, c="blue", label=r"$F_\mathrm{c,b}$")
    axes[2].plot(F_ocean.time, F_ocean, c="forestgreen", label=r"$F_\mathrm{w}$")
    axes[2].axvline(basal_start, c="grey")
    axes[2].set_ylabel(r"W m$^{-2}$")
    axes[2].set_ylim(-40, 40)
    axes[2].legend(bbox_to_anchor=(1.001, 1.001), loc="upper left", fontsize=cfg.fontsize)
    axes[2].grid()
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[2].set_xlim(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))

    for i, ax in enumerate(axes):
        ax.text(0.01, 0.96, f"({chr(97 + i)})", fontsize=cfg.fontsize + 4, fontweight="bold",
                transform=ax.transAxes, verticalalignment="top")
        ax.tick_params(axis="both", labelsize=cfg.fontsize)

    save_figure(fig, cfg.output_dir / "Fig6_HeatFluxes.png", cfg.dpi)


# -----------------------------------------------------------------------------
# Figure 7: surface residual
# -----------------------------------------------------------------------------


def figure_7_residual(cfg: FigureConfig, hb: xr.Dataset):
    F_net = hb_var(hb, "F_net_surface")
    F_sw = hb_var(hb, "F_net_shortwave")
    F_lw = hb_var(hb, "F_net_longwave")
    F_c = hb_var(hb, "F_conductive_surface")
    F_sens = hb_var(hb, "F_sensible")
    F_lat = hb_var(hb, "F_latent")
    F_snowmelt = hb_var(hb, "F_snow_melt")
    F_icemelt = hb_var(hb, "F_ice_melt")
    residual = hb_var(hb, "surface_residual")
    residual_roll = hb_var(hb, "surface_residual_smoothed")
    uncertainty = hb_var(hb, "surface_uncertainty_smoothed")

    F_c_pos, F_c_neg = F_c.where(F_c > 0, 0), F_c.where(F_c < 0, 0)
    F_sens_pos, F_sens_neg = F_sens.where(F_sens > 0, 0), F_sens.where(F_sens < 0, 0)
    F_lat_pos, F_lat_neg = F_lat.where(F_lat > 0, 0), F_lat.where(F_lat < 0, 0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, facecolor="w", layout="constrained")
    t = F_net.time
    bar_width = np.timedelta64(22, "h")

    axes[0].bar(t, F_lw, width=bar_width, color="firebrick", label=r"$F_{\mathrm{netLWR}}$")
    axes[0].bar(t, F_sw, width=bar_width, color="peru", label=r"$F_\mathrm{netSWR}$")
    axes[0].bar(t, F_c_neg, width=bar_width, bottom=F_sw.where(F_c < 0, 0),
                color="steelblue", label=r"$F_\mathrm{c,s}$")
    axes[0].bar(t, F_c_pos, width=bar_width, bottom=F_lw.where(F_c > 0, 0), color="steelblue")
    axes[0].bar(t, F_sens_neg, width=bar_width, bottom=F_c_neg + F_sw.where(F_sens < 0, 0),
                color="forestgreen", label=r"$F_\mathrm{SH}$")
    axes[0].bar(t, F_sens_pos, width=bar_width, bottom=F_c_pos + F_lw.where(F_sens > 0, 0),
                color="forestgreen")
    axes[0].bar(t, F_lat_neg, width=bar_width,
                bottom=F_c_neg + F_sw.where(F_lat < 0, 0) + F_sens_neg,
                color="violet", label=r"$F_\mathrm{LH}$")
    axes[0].bar(t, F_lat_pos, width=bar_width,
                bottom=F_c_pos + F_lw.where(F_lat > 0, 0) + F_sens_pos, color="violet")
    axes[0].bar(t, F_snowmelt, width=bar_width,
                bottom=F_c_neg + F_sw.where(F_snowmelt < 0, 0) + F_sens_neg + F_lat_neg,
                color="lawngreen", label=r"$F_{\mathrm{m,sn}}$")
    axes[0].bar(t, F_icemelt, width=bar_width,
                bottom=F_c_neg + F_sw.where(F_icemelt < 0, 0) + F_sens_neg + F_lat_neg + F_snowmelt,
                color="gold", label=r"$F_{\mathrm{m,ic}}$")
    axes[0].plot(t, residual_roll, "k--", label="Residual\n(smoothed)")
    axes[0].set_ylabel(r"W m$^{-2}$")
    axes[0].legend(bbox_to_anchor=(1, 1.001), loc="upper left", fontsize=cfg.fontsize)
    axes[0].grid()

    axes[1].plot(t, F_net, "k-", label=r"$F_{\mathrm{net,s}}$")
    axes[1].plot(t, F_icemelt + F_snowmelt, c="red", label=r"$F_{\mathrm{melt}}$")
    axes[1].plot(t, residual, c="darkgrey", linestyle="--", label="Residual\n(unsmoothed)")
    axes[1].plot(t, residual_roll, "k--", label="Residual\n(smoothed)")
    axes[1].set_ylabel(r"W m$^{-2}$")
    axes[1].legend(bbox_to_anchor=(1, 1.001), loc="upper left", fontsize=cfg.fontsize)
    axes[1].grid()

    axes[2].plot(t, residual_roll, "k--", label="Residual\n(smoothed)")
    axes[2].fill_between(t.values, -uncertainty.values, uncertainty.values, alpha=0.3,
                         facecolor="black", label="Uncertainty")

    if any(name in hb for name in HEAT_BUDGET_ALIASES["rain_mm"]):
        rain = hb_var(hb, "rain_mm")
        rain_days = np.unique(rain.where(rain > 0, drop=True).time.dt.floor("D").values.astype("datetime64[D]"))
        for i, day in enumerate(rain_days):
            axes[2].axvspan(day, day + np.timedelta64(1, "D"), color="lightblue", alpha=0.3,
                            lw=0, label="Rain days" if i == 0 else None)

    axes[2].set_ylabel(r"W m$^{-2}$")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[2].legend(bbox_to_anchor=(1, 1.001), loc="upper left", fontsize=cfg.fontsize)
    axes[2].set_xlim(t[0].values, t[-1].values)

    for i, ax in enumerate(axes):
        ax.text(0.01, 0.96, f"({chr(97 + i)})", fontsize=cfg.fontsize + 4, fontweight="bold",
                transform=ax.transAxes, verticalalignment="top")
        ax.tick_params(axis="both", labelsize=cfg.fontsize)

    save_figure(fig, cfg.output_dir / "Fig7_Residual.png", cfg.dpi)


# -----------------------------------------------------------------------------
# Figure 8: simple-model results
# -----------------------------------------------------------------------------


def _model_var(
    ds: xr.Dataset,
    quantity: str,
    scenario: str,
) -> xr.DataArray:
    """Return one model-scenario variable with a clear error if absent."""

    name = f"{quantity}_{scenario}"

    if name not in ds:
        available = sorted(ds.data_vars)
        raise KeyError(
            f"Figure 8 requires '{name}' in "
            f"{ds.encoding.get('source', 'the model dataset')}.\n"
            f"Available variables are: {available}\n"
            "Add the required scenario to SCENARIOS in "
            "run_semtner_model_general.py and rerun the model."
        )

    return ds[name]


def figure_8_model(
    cfg: FigureConfig,
    simba: dict[str, xr.DataArray],
):
    """
    Plot the simple-model comparison from the unified scenario NetCDF.

        H_i_no_Fw
        H_si_no_Fw
        H_i_no_Fw_no_snow_ice
        H_si_no_Fw_no_snow_ice

    With flooding disabled but rain retained, H_si_no_Fw_no_snow_ice contains
    the modeled meltwater/SWE + rain contribution.
    """

    ds_model = xr.open_dataset(
        _require_file(cfg.model_scenarios_file)
    )

    # ------------------------------------------------------------
    # Model scenarios
    # ------------------------------------------------------------

    H_i_control = _model_var(
        ds_model,
        "H_i",
        "no_Fw",
    )

    H_si_control = _model_var(
        ds_model,
        "H_si",
        "no_Fw",
    )

    H_i_no_snow_ice = _model_var(
        ds_model,
        "H_i",
        "no_Fw_no_snow_ice",
    )

    H_si_melt_rain = _model_var(
        ds_model,
        "H_si",
        "no_Fw_no_snow_ice",
    )

    model_time = ds_model.time

    # ------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------

    H_bottom = simba["H_bottom"].sel(
        time=slice(cfg.start, cfg.end)
    )

    H_ice = simba["H_ice"].sel(
        time=slice(cfg.start, cfg.end)
    )

    snow_air = simba["snow_air"].sel(
        time=slice(cfg.start, cfg.end)
    )

    # Preserve the observational snow-ice diagnostic used in the
    # original Figure 8.
    H_si_obs = -(H_ice - H_bottom)

    # Preserve the original hourly interpolation of the observed
    # snow-air interface, including nearest-value extrapolation at
    # the model-period edges.
    H_s_obs = (
        snow_air
        .sel(
            time=slice(
                pd.Timestamp(model_time.values[0]),
                pd.Timestamp(model_time.values[-1]),
            )
        )
        .resample(time="1h")
        .interpolate("linear")
        .interpolate_na(
            dim="time",
            method="nearest",
            fill_value="extrapolate",
        )
        .interp(time=model_time)
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        layout="constrained",
        height_ratios=[1.5, 1],
    )

    t = model_time.values

    # ============================================================
    # (a) Vertical evolution of snow and ice
    # ============================================================

    axes[0].plot(
        t,
        -H_i_control,
        color="#1f78b4",
    )

    hIce = axes[0].fill_between(
        t,
        -H_i_control,
        0,
        color="#1f78b4",
        alpha=0.5,
    )

    SnowAir, = axes[0].plot(
        model_time,
        H_s_obs,
        color="gray",
        label="Snow-air interface",
    )

    hSnow = axes[0].fill_between(
        t,
        H_si_control,
        H_s_obs,
        color="gray",
        alpha=0.2,
    )

    axes[0].plot(
        t,
        H_si_control,
        color="#a6cee3",
    )

    hFlood = axes[0].fill_between(
        t,
        0,
        H_si_control,
        color="#a6cee3",
        alpha=0.5,
    )

    # With flooding disabled, the remaining H_si comes from
    # surface melt and rain only.
    axes[0].plot(
        t,
        H_si_melt_rain,
        color="cornflowerblue",
    )

    hSImeltRain = axes[0].fill_between(
        t,
        0,
        H_si_melt_rain,
        edgecolor="cornflowerblue",
        facecolor="none",
        hatch="..",
        alpha=0.6,
    )

    axes[0].set_ylabel(
        "Vertical distance (m)",
        fontsize=cfg.fontsize,
    )

    # Observations.
    hObs, = axes[0].plot(
        H_bottom.time,
        H_bottom,
        "r--",
        label="Ice bottom",
    )

    hFloodObs, = axes[0].plot(
        H_si_obs.time,
        H_si_obs,
        linestyle="--",
        c="green",
        label="Snow-ice\ninterface",
    )

    # These field observations are specific to the 2024 paper.
    slob_handle = None

    if cfg.year == 2024:
        slob_dates = [
            "2024-03-19",
            "2024-03-26",
            "2024-04-02",
        ]

        for date in slob_dates:
            axes[0].scatter(
                np.datetime64(date).astype(datetime),
                0,
                marker="^",
                c="k",
                s=80,
                transform=axes[0].get_xaxis_transform(),
                clip_on=False,
                zorder=10,
            )

        slob_handle = Line2D(
            [],
            [],
            color="k",
            marker="^",
            linestyle="None",
            label="Slush/snow-ice observed",
        )

    # Hatch the observational period before the model begins.
    H_bottom_daily = H_bottom.resample(time="1D").mean()
    snow_air_daily = snow_air.resample(time="1D").mean()

    fill_mask = (
        H_bottom_daily.time
        < model_time.values[0]
    )

    axes[0].fill_between(
        H_bottom_daily.time.values,
        H_bottom_daily,
        snow_air_daily,
        where=fill_mask,
        facecolor="none",
        hatch="xx",
        edgecolor="darkgrey",
    )

    axes[0].set_xlim(
        pd.Timestamp(cfg.start),
        pd.Timestamp(cfg.end),
    )

    l1 = axes[0].legend(
        [
            hSnow,
            hFlood,
            hSImeltRain,
            hIce,
        ],
        [
            "Dry snow",
            "Snow ice",
            "SWE + rain",
            "Sea ice",
        ],
        bbox_to_anchor=(1, 0.99),
        loc="upper left",
        title="Model",
        fontsize=cfg.fontsize,
        title_fontsize=cfg.fontsize,
    )

    obs_handles = [
        SnowAir,
        hObs,
        hFloodObs,
    ]

    if slob_handle is not None:
        obs_handles.append(slob_handle)

    l2 = axes[0].legend(
        handles=obs_handles,
        bbox_to_anchor=(1, 0.6),
        loc="upper left",
        title="Observations",
        fontsize=cfg.fontsize,
        title_fontsize=cfg.fontsize,
    )

    l1.get_title().set_fontweight("bold")
    l2.get_title().set_fontweight("bold")
    axes[0].add_artist(l1)

    # ============================================================
    # (b) Congelation-ice sensitivity to flooding/snow-ice
    # ============================================================

    axes[1].plot(
        model_time,
        H_i_control,
        c="red",
        label="Control",
    )

    axes[1].plot(
        model_time,
        H_i_no_snow_ice,
        c="blue",
        label="No flooding",
    )

    axes[1].plot(
        H_bottom.time,
        -H_bottom,
        "k-",
        label="Observations",
    )

    axes[1].legend(
        bbox_to_anchor=(1, 0.97),
        fontsize=cfg.fontsize,
    )

    axes[1].xaxis.set_major_formatter(
        mdates.DateFormatter("%m-%d")
    )

    axes[1].grid(
        True,
        alpha=0.5,
    )

    axes[1].set_ylabel(
        "Sea ice thickness (m)",
        fontsize=cfg.fontsize,
    )

    # Preserve the historical paper limits for the 2024 figure.
    if cfg.year == 2024:
        axes[1].set_ylim(
            0.45,
            0.64,
        )

    for i, ax in enumerate(axes):
        ax.text(
            0.01,
            0.97,
            f"({chr(97 + i)})",
            fontsize=cfg.fontsize + 4,
            fontweight="bold",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.tick_params(
            axis="both",
            labelsize=cfg.fontsize,
        )

    save_figure(
        fig,
        cfg.output_dir / "Fig8_Model.png",
        600,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def make_all_figures(cfg: FigureConfig) -> None:
    cfg.output_dir = cfg.root / cfg.output_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    simba = load_simba_products(cfg)
    weekly = xr.open_dataset(_require_file(cfg.weekly_ice_file))
    weather = load_weather(cfg)

    figure_2_simba(cfg, simba, weekly)
    figure_3_atmosphere(cfg, weather)
    figure_4_ocean(cfg)
    figure_4_ctd(cfg)

    # Figures 6 and 7 use the final heat-budget products. Keeping these in one
    # NetCDF makes the plotting workflow independent of interactive variables.
    hb = load_heat_budget(cfg)
    figure_6_heat_fluxes(cfg, simba, hb)
    figure_7_residual(cfg, hb)

    figure_8_model(cfg, simba)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from processed data.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/paper"),
                        help="Output directory relative to the repository root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FigureConfig(root=args.root, output_dir=args.output_dir)
    make_all_figures(cfg)


if __name__ == "__main__":
    main()
