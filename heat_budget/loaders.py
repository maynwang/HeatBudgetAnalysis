from pathlib import Path
from typing import Dict

import pandas as pd
import xarray as xr

from functions import IMS_toolbox as IMS

from .config import SeasonConfig, HeatBudgetParameters

def _rename_time(da):
    """Rename SIMBA time dimension to 'time'."""

    if "time_6h" in da.dims:
        return da.rename({"time_6h": "time"})

    if "time_daily" in da.dims:
        return da.rename({"time_daily": "time"})

    return da

def _daily_mean(da, start: str, end: str):
    return (
        da
        .sel(time=slice(start, end))
        .resample(time="1D")
        .mean("time")
    )


def load_raw_data(cfg: SeasonConfig) -> Dict[str, object]:
    """Load raw files used by the heat-budget and residual calculations."""

    # --------------------------------------------------
    # Weather
    # --------------------------------------------------
    weather = xr.open_dataset(cfg.weather_file)

    # --------------------------------------------------
    # Processed SIMBA interfaces
    # --------------------------------------------------
    interfaces = xr.open_dataset(cfg.simba_interfaces_file)

    raw = {
        "weather": xr.open_dataset(cfg.weather_file),

        "temperature": _rename_time(interfaces["temperature"]),
        "temp_ice": _rename_time(interfaces["temp_ice"]),
        "H_ice": _rename_time(interfaces["H_ice"]),
        "H_bottom": _rename_time(interfaces["H_bottom"]),
        "snow_air": _rename_time(interfaces["snow_air"]),
        "snow_ice": _rename_time(interfaces["snow_ice"]),
        "snow_ice_smoothed": _rename_time(interfaces["snow_ice_smoothed"]),
        "H_snow": _rename_time(interfaces["H_snow"]),
    }


    # --------------------------------------------------
    # Rain
    # --------------------------------------------------
    if cfg.rain_path is not None:
        rain_file = pd.read_csv(
            cfg.rain_path,
            low_memory=False,
        )

        raw["rain"] = IMS.load_Rway_station_data(
            rain_file,
            "rain",
        )

    return raw

def preprocess_daily(
    raw: Dict[str, object],
    cfg: SeasonConfig,
    params: HeatBudgetParameters,
) -> xr.Dataset:
    """Convert all downstream inputs to a common daily time grid."""

    weather = raw["weather"].sel(time=slice(cfg.start, cfg.end))

    T_air = weather[cfg.air_temperature_var].resample(time="1D").mean("time")
    RH = weather[cfg.relative_humidity_var].resample(time="1D").mean("time")
    pressure = weather[cfg.pressure_var].resample(time="1D").mean("time")

    # first average weather data to hourly resolution (for max winds in residual)
    weather_hourly = weather.resample(time="1h").mean("time")

    # Convert hourly wind from measurement height to 10 m
    wind_10m_hourly = (
        weather_hourly[cfg.wind_speed_var]
        * (10.0 / cfg.wind_measurement_height) ** (1.0 / 7.0)
    )

    # Daily mean 10 m wind
    wind_10m = wind_10m_hourly.resample(time="1D").mean("time")

    # Daily maximum of the hourly-mean 10 m wind
    wind_10m_max = wind_10m_hourly.resample(time="1D").max("time")

    F_lw_measured = -weather[cfg.net_longwave_var].resample(time="1D").mean("time")

    SW_in_native = weather[cfg.sw_in_var]
    SW_out_native = weather[cfg.sw_out_var]

    SW_in = (
        SW_in_native.where(SW_in_native > 0, 0)
        .resample(time="1D")
        .mean("time")
    )
    SW_out = (
        SW_out_native.where(SW_out_native > 0, 0)
        .resample(time="1D")
        .mean("time")
    )

    LW_in_native = weather[cfg.lw_in_var]
    LW_out_native = weather[cfg.lw_out_var]

    # Calculate the surface temp using an assumed emissivity and LWR
    T_surface_native = (
        (
            (
                LW_out_native
                - (1.0 - params.epsilon) * LW_in_native
            )
            / (params.epsilon * params.sigma)
        ) ** 0.25
        - 273.15
    )
    T_surface = T_surface_native.resample(time="1D").mean("time")

    ds = xr.Dataset(
        {
            "T_air": T_air,
            "RH": RH,
            "pressure": pressure,
            "wind_10m": wind_10m,
            "wind_10m_max": wind_10m_max,
            "F_lw_measured": F_lw_measured,
            "SW_in": SW_in,
            "SW_out": SW_out,
            "T_surface": T_surface,
            "H_ice": _daily_mean(raw["H_ice"], cfg.start, cfg.end),
            "H_snow": _daily_mean(raw["H_snow"], cfg.start, cfg.end),
            "H_bottom": _daily_mean(raw["H_bottom"], cfg.start, cfg.end),
            "snow_air": _daily_mean(raw["snow_air"], cfg.start, cfg.end),
            "snow_ice": _daily_mean(raw["snow_ice"], cfg.start, cfg.end),
            "temp_ice": _daily_mean(raw["temp_ice"], cfg.start, cfg.end),
            "temperature": _daily_mean(raw["temperature"], cfg.start, cfg.end),
        }
    )


    if "rain" in raw:
        ds["rain_mm"] = (
            raw["rain"]
            .sel(time=slice(cfg.start, cfg.end))
            .resample(time="1D")
            .sum("time")
        )

    ds.attrs.update(
        {
            "year": cfg.year,
            "analysis_start": cfg.start,
            "analysis_end": cfg.end,
            "time_resolution": "daily",
        }
    )
    return ds
