"""
Explore processed IMS outputs across one or more years.

Figures
-------
1. All numeric 1-D weather variables, one variable per subplot.
2. SIMBA temperature with snow-air, snow-ice, and ice-water interfaces.
   For a single year, this follows Figure 2 of the paper plotting code.
   For multiple years, one SIMBA temperature panel is shown per year.
3. Heat-budget fluxes, one flux per subplot.

Run from the repository root:

    python explore_outputs.py

Edit YEARS_TO_PLOT near the top to choose the years:

    YEARS_TO_PLOT = [2024]
    YEARS_TO_PLOT = [2024, 2025]
    YEARS_TO_PLOT = "all"
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import colormaps as cmaps
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from functions import plotting as pl


# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT = Path(".")

# Options:
#   [2024]
#   [2024, 2025]
#   "all"
YEARS_TO_PLOT = "all"

OUTPUT_DIR = Path("figures/explore_outputs/all")

SAVE_FIGURES = True
SHOW_FIGURES = False
DPI = 250

# Weather is reduced to hourly means for plotting.
WEATHER_RESAMPLE = "1h"

# If multiple years are selected, line plots are aligned by month/day using
# this leap year as a plotting-only reference year.
COMPARISON_REFERENCE_YEAR = 2000

FONTSIZE = 11


# =============================================================================
# FILE PATHS
# =============================================================================


def year_dir(year: int) -> Path:
    return ROOT / "data" / str(year)


def weather_file(year: int) -> Path:
    return year_dir(year) / "WeatherStation" / "WeatherVars.nc"


def simba_file(year: int) -> Path:
    return (
        year_dir(year)
        / "SIMBA"
        / "processed"
        / f"SIMBA_interfaces_{year}.nc"
    )


def weekly_ice_file(year: int) -> Path:
    return (
        year_dir(year)
        / "SiteVisits"
        / "SiteVisits_Weekly_IceSnowWater.nc"
    )


def heat_budget_dir(year: int) -> Path:
    return year_dir(year) / "HeatBudget" / "processed"


# =============================================================================
# VARIABLES TO PLOT
# =============================================================================

# Variables in HeatBudget_fluxes_{year}.nc
HEAT_FLUX_PANELS = [
    ("F_lw", "Net longwave", r"$F_{\mathrm{netLWR}}$"),
    ("F_sw", "Net shortwave", r"$F_{\mathrm{netSWR}}$"),
    ("F_sens", "Sensible", r"$F_{\mathrm{SH}}$"),
    ("F_lat", "Latent", r"$F_{\mathrm{LH}}$"),
    ("F_cond_surface", "Surface conductive", r"$F_{\mathrm{c,s}}$"),
    ("F_net_surface", "Net surface", r"$F_{\mathrm{net,s}}$"),
    ("F_phase_change_basal", "Basal phase change", r"$F_{\mathrm{g}}$"),
    ("F_cond_ref", "Basal conductive", r"$F_{\mathrm{c,b}}$"),
    ("F_w", "Ocean", r"$F_{\mathrm{w}}$"),
    ("F_snow_melt", "Snow melt", r"$F_{\mathrm{m,sn}}$"),
    ("F_ice_melt", "Ice melt", r"$F_{\mathrm{m,ic}}$"),
]

HEAT_FLUX_INPUTS = [
    ("T_air", "Air temperature", r"$T_\mathrm{air}$"),
    ("wind_10m", "10 m wind speed", r"$U_\mathrm{wd}$"),
    ("wind_10m_max", "Max 10 m wind speed", r"$U_\mathrm{wd,max}$"),
    ("F_lw_net","Net longwave", r"$F_{\mathrm{netLWR}}$"),
    ("SW_in", "Incoming shortwave", r"$SW_{\mathrm{in}}$"),
    ("SW_out", "Outgoing shortwave", r"$SW_{\mathrm{out}}$"),
    ("albedo_calculated", "Albedo", r"$\alpha = SW_{\mathrm{out}}/SW_{\mathrm{in}}$"),
    ("T_surface", "Surface temp", r"$T_\mathrm{s}$"),
]

THICKNESS_PANELS = [
    ("H_ice", "Total thickness"),
    ("H_bottom","Sea ice thickness (congelation)"),
    ("snow_ice", "Snow ice thickness"),
    ("H_snow", "Snow thickness"),
    ("rain_mm","Rain mm")
]

# =============================================================================
# GENERIC HELPERS
# =============================================================================


def resolve_years() -> list[int]:
    """Resolve YEARS_TO_PLOT into an explicit sorted list."""

    if YEARS_TO_PLOT == "all":
        data_dir = ROOT / "data"

        years = sorted(
            int(path.name)
            for path in data_dir.iterdir()
            if path.is_dir()
            and path.name.isdigit()
            and len(path.name) == 4
        )

        if not years:
            raise FileNotFoundError(
                f"No year directories were found in {data_dir}"
            )

        return years

    years = sorted(int(year) for year in YEARS_TO_PLOT)

    if not years:
        raise ValueError("YEARS_TO_PLOT is empty.")

    return years


def comparison_mode(years: list[int]) -> bool:
    return len(years) > 1


def comparison_time(time, compare: bool) -> pd.DatetimeIndex:
    """
    Return real timestamps for one year, or map timestamps from different years
    onto the same leap reference year for month/day comparison.
    """

    t = pd.DatetimeIndex(pd.to_datetime(np.asarray(time)))

    if not compare:
        return t

    return pd.DatetimeIndex(
        [
            timestamp.replace(year=COMPARISON_REFERENCE_YEAR)
            for timestamp in t
        ]
    )


def panel_label(index: int) -> str:
    return f"({chr(97 + index)})"


def format_time_axis(ax, compare: bool):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    if compare:
        ax.set_xlabel("Month-day")


def save_and_show(fig, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_FIGURES:
        path = OUTPUT_DIR / filename
        fig.savefig(
            path,
            dpi=DPI,
            bbox_inches="tight",
        )
        print(f"Saved {path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def get_first(
    ds: xr.Dataset,
    names: Iterable[str],
) -> xr.DataArray | None:
    for name in names:
        if name in ds:
            return ds[name]

    return None


def normalize_time_dimension(da: xr.DataArray) -> xr.DataArray:
    """Normalize SIMBA native time dimension names to `time`."""

    if "time_6h" in da.dims:
        return da.rename({"time_6h": "time"})

    if "time_daily" in da.dims:
        return da.rename({"time_daily": "time"})

    return da


def numeric_time_series(da: xr.DataArray) -> xr.DataArray | None:
    """Return a numeric 1-D time series if possible."""

    if "time" not in da.dims:
        return None

    if not np.issubdtype(da.dtype, np.number):
        return None

    da = da.squeeze(drop=True)

    if da.dims != ("time",):
        return None

    return da


# =============================================================================
# DATA LOADERS
# =============================================================================


def load_weather(year: int) -> xr.Dataset:
    path = weather_file(year)

    if not path.exists():
        raise FileNotFoundError(
            f"Weather file not found for {year}: {path}"
        )

    return xr.open_dataset(path)


def load_simba(year: int) -> dict[str, xr.DataArray]:
    path = simba_file(year)

    if not path.exists():
        raise FileNotFoundError(
            f"SIMBA file not found for {year}: {path}"
        )

    ds = xr.open_dataset(path)

    names = [
        "temperature",
        "temp_ice",
        "H_ice",
        "H_bottom",
        "snow_air",
        "snow_ice",
        "H_snow",
        "snow_ice_smoothed",
    ]

    return {
        name: normalize_time_dimension(ds[name])
        for name in names
        if name in ds
    }


def load_weekly_ice(year: int) -> xr.Dataset | None:
    path = weekly_ice_file(year)

    if not path.exists():
        print(
            f"Weekly site-visit file not found for {year}; "
            "weekly markers will be skipped."
        )
        return None

    return xr.open_dataset(path)


def load_heat_budget_fluxes(
    year: int,
) -> xr.Dataset:

    path = (
        heat_budget_dir(year)
        / f"HeatBudget_fluxes_{year}.nc"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Heat-budget flux file not found for {year}: {path}"
        )

    return xr.open_dataset(path)

def load_heat_budget_inputs(
    year: int,
) -> xr.Dataset:

    path = (
        heat_budget_dir(year)
        / f"HeatBudget_inputs_{year}.nc"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Heat-budget inputs file not found for {year}: {path}"
        )

    return xr.open_dataset(path)

# =============================================================================
# FIGURE 1: ALL WEATHER VARIABLES
# =============================================================================


def figure_1_weather(years: list[int]):
    """
    Plot every numeric 1-D weather variable.

    Each weather variable gets its own subplot. If multiple years are selected,
    the years are overlaid and aligned by month/day.
    """

    compare = comparison_mode(years)
    weather = {}

    for year in years:
        try:
            weather[year] = (
                load_weather(year)
                .resample(time=WEATHER_RESAMPLE)
                .mean()
            )
        except FileNotFoundError as exc:
            print(exc)

    if not weather:
        print("Skipping Figure 1: no weather files found.")
        return

    variable_names = []

    for ds in weather.values():
        for name in ds.data_vars:
            if (
                name not in variable_names
                and numeric_time_series(ds[name]) is not None
            ):
                variable_names.append(name)

    if not variable_names:
        print(
            "Skipping Figure 1: no numeric 1-D weather variables found."
        )
        return

    fig, axes = plt.subplots(
        len(variable_names),
        1,
        figsize=(11, max(2.2 * len(variable_names), 5)),
        sharex=True,
        facecolor="white",
        layout="constrained",
    )

    axes = np.atleast_1d(axes)

    for i, (ax, variable) in enumerate(zip(axes, variable_names)):
        for year in years:
            if year not in weather or variable not in weather[year]:
                continue

            da = numeric_time_series(weather[year][variable])

            if da is None:
                continue

            x = comparison_time(da.time.values, compare)

            ax.plot(
                x,
                da.values,
                linewidth=1,
                label=str(year),
            )

            # If albedo, set limits y limits to be between 0 and 1
            if variable == 'Albedo_Avg':
                ax.set_ylim(0,1)

        long_name = variable
        units = ""

        for year in years:
            if year in weather and variable in weather[year]:
                long_name = weather[year][variable].attrs.get(
                    "long_name",
                    variable,
                )
                units = weather[year][variable].attrs.get("units", "")
                break

        ax.set_title(
            str(long_name),
            loc="left",
            fontsize=FONTSIZE,
        )

        if units:
            ax.set_ylabel(units, fontsize=FONTSIZE)

        ax.text(
            0.01,
            0.92,
            panel_label(i),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=FONTSIZE,
            va="top",
        )

        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=FONTSIZE)

    if len(years) > 1:
        axes[0].legend(title="Year", loc="upper right")

    format_time_axis(axes[-1], compare)

    fig.suptitle(
        "Weather station variables (hourly)",
        fontsize=FONTSIZE + 2,
    )

    save_and_show(fig, "Figure1_weather.png")


# =============================================================================
# FIGURE 2: SIMBA TEMPERATURE + INTERFACES
# =============================================================================


def weekly_var(
    weekly: xr.Dataset | None,
    names: Iterable[str],
) -> xr.DataArray | None:
    if weekly is None:
        return None

    return get_first(weekly, names)


def plot_simba_panel(
    ax,
    year: int,
    simba: dict[str, xr.DataArray],
    weekly: xr.Dataset | None,
    compare: bool,
):
    """Plot one SIMBA panel using the paper Figure 2 styling."""

    required = [
        "temperature",
        "snow_air",
        "snow_ice_smoothed",
        "H_bottom",
    ]

    missing = [name for name in required if name not in simba]

    if missing:
        raise KeyError(
            f"{year} SIMBA file is missing: {missing}"
        )

    temperature = simba["temperature"]
    snow_air = simba["snow_air"]
    snow_ice_smoothed = simba["snow_ice_smoothed"]
    H_bottom = simba["H_bottom"]

    # Original Figure 2 colormap.
    orig = cmaps.WhBlGrYeRe
    cmap = pl.custom_cmap_trimmed(
        orig,
        skew=3,
        vmin=0.2,
        vmax=1,
    )

    x_temp = comparison_time(
        temperature.time.values,
        compare,
    )

    im = ax.pcolormesh(
        x_temp,
        temperature.z,
        temperature.transpose("z", "time"),
        cmap=cmap,
        alpha=0.8,
        vmin=-15,
        vmax=0,
        shading="auto",
    )

    ax.plot(
        comparison_time(snow_air.time.values, compare),
        snow_air.values,
        c="gainsboro",
        label="Air-snow",
    )

    ax.plot(
        comparison_time(snow_ice_smoothed.time.values, compare),
        snow_ice_smoothed.values,
        c="k",
        label="Snow-ice",
    )

    ax.plot(
        comparison_time(H_bottom.time.values, compare),
        H_bottom.values,
        c="b",
        label="Ice-water",
    )

    hi = weekly_var(
        weekly,
        (
            "hi",
            "Hice",
            "H_ice",
            "ice_thickness",
        ),
    )

    hs = weekly_var(
        weekly,
        (
            "hs_snowStake_mean",
            "Hsnow",
            "hs",
            "H_snow",
            "snow_depth",
        ),
    )

    if weekly is not None and "time" in weekly.coords:
        weekly_time = comparison_time(
            weekly.time.values,
            compare,
        )

        if hi is not None:
            ax.plot(
                weekly_time[1:],
                -hi.values[1:],
                "o",
                c="blue",
                markeredgecolor="k",
                clip_on=False,
                label="Weekly ice thickness",
            )

        if hs is not None:
            ax.plot(
                weekly_time[1:],
                hs.values[1:],
                "^",
                c="white",
                markeredgecolor="k",
                clip_on=False,
                label="Weekly snow depth",
            )

    ax.axhline(0, color="k", linestyle="--")
    ax.set_ylim(-2.9, 1.7)
    ax.set_ylabel(r"$z$ (m)", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_title(str(year), loc="left", fontsize=FONTSIZE + 1)

    # Preserve the special 2024 event ticks from the paper Figure 2 in
    # single-year mode.
    if year == 2024 and not compare:
        event_dates = pd.to_datetime(
            [
                "2024-02-13",
                "2024-02-19",
                "2024-02-29",
                "2024-03-09",
                "2024-03-23",
                "2024-03-27",
                "2024-04-08",
            ]
        )

        ax.xaxis.set_minor_formatter(ax.xaxis.get_major_formatter())
        ax.set_xticks(event_dates, minor=True)
        ax.tick_params(
            axis="x",
            which="minor",
            bottom=True,
            top=False,
            direction="in",
            length=6,
            width=1,
            labelbottom=True,
            pad=-32,
            rotation=35,
            labelsize=FONTSIZE,
        )

    return im


def figure_2_simba(
    years: list[int],
):
    """
    SIMBA temperature and interfaces.

    One year:
        reproduces the paper Figure 2 layout/styling.

    Multiple years:
        one vertically stacked temperature/interface panel per year,
        with dates aligned by month/day.
    """

    compare = comparison_mode(years)

    simba_by_year = {}
    weekly_by_year = {}

    for year in years:
        try:
            simba_by_year[year] = (
                load_simba(year)
            )

            weekly_by_year[year] = (
                load_weekly_ice(year)
            )

        except FileNotFoundError as exc:
            print(exc)

    plot_years = [
        year
        for year in years
        if year in simba_by_year
    ]

    if not plot_years:
        print(
            "Skipping Figure 2: no SIMBA files found."
        )
        return

# ---------------------------------------------------------
# Common x-axis limits across all years
# Include both SIMBA and weekly observation times
# ---------------------------------------------------------

    all_times = []

    for year in plot_years:

        # SIMBA temperature times
        temperature = (
            simba_by_year[year][
                "temperature"
            ]
        )

        x_temp = comparison_time(
            temperature.time.values,
            compare,
        )

        all_times.extend(
            x_temp
        )


        # Weekly observation times
        weekly = weekly_by_year[
            year
        ]

        if (
            weekly is not None
            and "time" in weekly.coords
        ):

            x_weekly = comparison_time(
                weekly.time.values,
                compare,
            )

            all_times.extend(
                x_weekly
            )


    x_min = min(
        all_times
    )

    x_max = max(
        all_times
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        len(plot_years),
        1,
        figsize=(
            10,
            6 * len(plot_years),
        ),
        facecolor="white",
        squeeze=False,
        sharex=True,
        layout="constrained",
    )

    axes = axes[:, 0]

    for i, (
        ax,
        year,
    ) in enumerate(
        zip(
            axes,
            plot_years,
        )
    ):

        im = plot_simba_panel(
            ax=ax,
            year=year,
            simba=simba_by_year[year],
            weekly=weekly_by_year[
                year
            ],
            compare=compare,
        )

        # Same x-axis for every subplot
        ax.set_xlim(
            x_min,
            x_max,
        )

        # Individual but identical colorbar
        cbar = fig.colorbar(
            im,
            ax=ax,
            pad=0.02,
            extend="both",
            ticks=[
                -15,
                -12,
                -9,
                -6,
                -3,
                0,
            ],
        )

        cbar.set_label(
            r"Temperature ($^\circ$C)",
            size=FONTSIZE,
        )

        cbar.ax.tick_params(
            labelsize=FONTSIZE,
        )

        if i == 0:
            ax.legend(
                bbox_to_anchor=(
                    0,
                    1.18,
                ),
                loc="upper left",
                fontsize=FONTSIZE,
                ncol=3,
            )

    format_time_axis(
        axes[-1],
        compare,
    )

    save_and_show(
        fig,
        "Figure2_SIMBA_temperature_interfaces.png",
    )


# =============================================================================
# FIGURE 3: INDIVIDUAL HEAT FLUXES
# =============================================================================


def figure_3_heat_fluxes(
    years: list[int],
):

    compare = comparison_mode(years)

    fluxes_by_year = {}

    for year in years:

        try:
            fluxes_by_year[year] = (
                load_heat_budget_fluxes(year)
            )

        except FileNotFoundError as exc:
            print(exc)


    if not fluxes_by_year:

        print(
            "Skipping Figure 3: "
            "no heat-budget flux files found."
        )

        return


    fig, axes = plt.subplots(
        len(HEAT_FLUX_PANELS),
        1,
        figsize=(
            11,
            2.2 * len(HEAT_FLUX_PANELS),
        ),
        sharex=True,
        facecolor="white",
        layout="constrained",
    )

    axes = np.atleast_1d(
        axes
    )


    for i, (
        ax,
        (
            variable,
            title,
            symbol,
        ),
    ) in enumerate(
        zip(
            axes,
            HEAT_FLUX_PANELS,
        )
    ):

        for year in years:

            if year not in fluxes_by_year:
                continue


            ds = fluxes_by_year[
                year
            ]


            if variable not in ds:

                print(
                    f"{year}: "
                    f"{variable} not found "
                    "in heat-budget flux file."
                )

                continue


            da = (
                ds[variable]
                .squeeze(
                    drop=True
                )
            )


            if da.dims != ("time",):

                print(
                    f"{year}: skipping "
                    f"{variable}; dimensions "
                    f"are {da.dims}."
                )

                continue


            x = comparison_time(
                da.time.values,
                compare,
            )


            ax.plot(
                x,
                da.values,
                linewidth=1.2,
                label=str(year),
            )


        ax.axhline(
            0,
            color="0.5",
            linewidth=0.8,
        )


        ax.set_title(
            f"{title}  {symbol}",
            loc="left",
            fontsize=FONTSIZE,
        )


        ax.set_ylabel(
            r"W m$^{-2}$",
            fontsize=FONTSIZE,
        )


        ax.text(
            0.01,
            0.92,
            panel_label(i),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=FONTSIZE,
            va="top",
        )


        ax.grid(
            alpha=0.3
        )


        ax.tick_params(
            labelsize=FONTSIZE
        )


    if len(years) > 1:

        axes[0].legend(
            title="Year",
            loc="upper right",
        )


    format_time_axis(
        axes[-1],
        compare,
    )


    fig.suptitle(
        "Heat-budget fluxes (daily)",
        fontsize=FONTSIZE + 2,
    )


    save_and_show(
        fig,
        "Figure3_heat_fluxes.png",
    )

# =============================================================================
# FIGURE 4: ICE, SNOW-ICE, AND SNOW THICKNESS
# =============================================================================


def figure_4_thicknesses(
    years: list[int],
):

    compare = comparison_mode(years)

    inputs_by_year = {}


    for year in years:

        try:

            inputs_by_year[year] = (
                load_heat_budget_inputs(year)
            )

        except FileNotFoundError as exc:

            print(exc)


    if not inputs_by_year:

        print(
            "Skipping Figure 4: "
            "no heat-budget input files found."
        )

        return


    fig, axes = plt.subplots(
        len(THICKNESS_PANELS),
        1,
        figsize=(10, 8),
        sharex=True,
        facecolor="white",
        layout="constrained",
    )


    axes = np.atleast_1d(
        axes
    )


    for i, (
        ax,
        (
            variable,
            title,
        ),
    ) in enumerate(
        zip(
            axes,
            THICKNESS_PANELS,
        )
    ):

        for year in years:

            if year not in inputs_by_year:
                continue


            ds = inputs_by_year[
                year
            ]


            if variable not in ds:

                print(
                    f"{year}: "
                    f"{variable} not found "
                    "in heat-budget inputs."
                )

                continue


            da = (
                ds[variable]
                .squeeze(
                    drop=True
                )
            )


            if da.dims != ("time",):

                print(
                    f"{year}: skipping "
                    f"{variable}; dimensions "
                    f"are {da.dims}."
                )

                continue


            x = comparison_time(
                da.time.values,
                compare,
            )


            ax.plot(
                x,
                da.values,
                linewidth=1.5,
                label=str(year),
            )


        ax.set_title(
            title,
            loc="left",
            fontsize=FONTSIZE,
        )


        ax.set_ylabel(
            "m",
            fontsize=FONTSIZE,
        )


        ax.text(
            0.01,
            0.92,
            panel_label(i),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=FONTSIZE,
            va="top",
        )


        ax.grid(
            alpha=0.3
        )


        ax.tick_params(
            labelsize=FONTSIZE
        )


    if len(years) > 1:

        axes[0].legend(
            title="Year",
            loc="upper right",
        )


    format_time_axis(
        axes[-1],
        compare,
    )


    fig.suptitle(
        "Snow and ice thickness",
        fontsize=FONTSIZE + 2,
    )


    save_and_show(
        fig,
        "Figure4_thicknesses.png",
    )


def figure_5_inputs(
    years: list[int],
):

    compare = comparison_mode(years)

    fluxes_by_year = {}

    for year in years:

        try:
            fluxes_by_year[year] = (
                load_heat_budget_inputs(year)
            )

        except FileNotFoundError as exc:
            print(exc)


    if not fluxes_by_year:

        print(
            "Skipping Figure 5: "
            "no heat-budget inputs files found."
        )

        return


    fig, axes = plt.subplots(
        len(HEAT_FLUX_INPUTS),
        1,
        figsize=(
            11,
            2.2 * len(HEAT_FLUX_INPUTS),
        ),
        sharex=True,
        facecolor="white",
        layout="constrained",
    )

    axes = np.atleast_1d(
        axes
    )


    for i, (
        ax,
        (
            variable,
            title,
            symbol,
        ),
    ) in enumerate(
        zip(
            axes,
            HEAT_FLUX_INPUTS,
        )
    ):

        for year in years:

            if year not in fluxes_by_year:
                continue


            ds = fluxes_by_year[
                year
            ]


            if variable not in ds:

                print(
                    f"{year}: "
                    f"{variable} not found "
                    "in heat-budget inputs file."
                )

                continue


            da = (
                ds[variable]
                .squeeze(
                    drop=True
                )
            )


            if da.dims != ("time",):

                print(
                    f"{year}: skipping "
                    f"{variable}; dimensions "
                    f"are {da.dims}."
                )

                continue


            x = comparison_time(
                da.time.values,
                compare,
            )


            ax.plot(
                x,
                da.values,
                linewidth=1.2,
                label=str(year),
            )


        ax.axhline(
            0,
            color="0.5",
            linewidth=0.8,
        )


        ax.set_title(
            f"{title}  {symbol}",
            loc="left",
            fontsize=FONTSIZE,
        )


        ax.set_ylabel(
            r"W m$^{-2}$",
            fontsize=FONTSIZE,
        )


        ax.text(
            0.01,
            0.92,
            panel_label(i),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=FONTSIZE,
            va="top",
        )


        ax.grid(
            alpha=0.3
        )


        ax.tick_params(
            labelsize=FONTSIZE
        )


    if len(years) > 1:

        axes[0].legend(
            title="Year",
            loc="upper right",
        )


    format_time_axis(
        axes[-1],
        compare,
    )


    fig.suptitle(
        "Heat-budget inputs (processed)",
        fontsize=FONTSIZE + 2,
    )


    save_and_show(
        fig,
        "Figure5_inputs.png",
    )


# =============================================================================
# RUN
# =============================================================================


def main():
    years = resolve_years()

    print("Plotting years:", years)

    figure_1_weather(years)
    figure_2_simba(years)
    figure_3_heat_fluxes(years)
    figure_4_thicknesses(years)
    figure_5_inputs(years)

if __name__ == "__main__":
    main()
