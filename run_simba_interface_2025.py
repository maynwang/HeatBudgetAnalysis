from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from simba_interfaces import (
    SIMBAConfig,
    load_simba_data,
    correct_temperature,
    detect_snow_ice,
    detect_ice_water,
    detect_snow_air,
    node_to_z,
    apply_values,
    apply_ranges,
    smooth_snow_ice,
    smooth_ice_water,
    build_interface_dataset,
    save_interface_dataset,
    smooth_ice_water_clamped,
    correct_interface_outliers,
    detect_snow_air_from_heating
)


ROOT = Path(".")


# ============================================================
# CONFIG
# ============================================================

cfg = SIMBAConfig(

    year=2025,

    start="2025-02-18",
    end="2025-04-12",

    # IMPORTANT: offset (node number at top of ice counted in the field)
    # Or detected by eye from pcolormesh plot
    # 2025 top of ice was node #75, top of water #77
    offset_cm=146,

    calibration_start="2025-03-25",
    calibration_end="2025-04-10",
)



# ============================================================
# LOAD DATA
# ============================================================

simba = load_simba_data(
    ROOT,
    cfg,
)


weekly_ice = xr.open_dataset(
    "data/2025/SiteVisits/SiteVisits_Weekly_IceSnowWater.nc"
)


# ============================================================
# TEMPERATURE
# ============================================================

da_temp = correct_temperature(
    simba["temp"],
    cfg,
)

# Plot temperature

# Convert timestamps to numeric Matplotlib dates
time = mdates.date2num(
    pd.to_datetime(da_temp.time.values).to_pydatetime()
)

fig, ax = plt.subplots(figsize=(12, 6))

im = ax.pcolormesh(
    time,
    da_temp.z.values,
    da_temp.transpose("z", "time").values,
    shading="auto",
    cmap="RdBu_r",
)

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

fig.autofmt_xdate()
ax.set_ylabel("z (m)")
fig.colorbar(im, ax=ax, label="Temperature (°C)")

im.set_clim(-20,2)

# ============================================================
# SNOW-ICE
# ============================================================

snow_ice_raw = detect_snow_ice(
    simba["del1"],
    simba["del4"],
    cfg,
)


snow_ice = snow_ice_raw.copy()


# Node -> metres
snow_ice_m = node_to_z(
    snow_ice,
    cfg,
)

## Plot the temp gradient for manual corrections 

T_30_120 = simba["del1"] / simba["del4"]

z = (
    cfg.offset_cm
    - T_30_120.node.values * cfg.node_spacing_cm
) / 100

fig, ax = plt.subplots(figsize=(12, 6))

im = ax.pcolormesh(
    T_30_120.time.values,
    z,
    T_30_120.transpose("node", "time").values,
    shading="auto",
    cmap="RdBu_r",
)

fig.colorbar(im, ax=ax, label="HT30/HT120")
ax.set_ylabel("z (m)")
ax.set_xlabel("Time")

# Manual 2025 corrections

snow_ice_m = apply_values(
    snow_ice_m,
    {
        "2025-02-27": 0,
        "2025-03-25": 0.02,
    },
)



(
    snow_ice_daily,
    snow_ice_6h,
    snow_ice_smooth,
) = smooth_snow_ice(
    snow_ice_m,
    da_temp.time,
    cfg,
)

# Fill nans at the beginning and end with first and last values
snow_ice_smooth = (
    snow_ice_smooth
    .ffill(dim="time")
    .bfill(dim="time")
)



# ============================================================
# ICE-WATER
# ============================================================

ice_water_raw = detect_ice_water(
    simba["del0"],
    simba["del1"],
    cfg,
)


ice_water = ice_water_raw.copy()

ice_water_m = node_to_z(
    ice_water,
    cfg,
)


# Manual node corrections
ice_water_m = correct_interface_outliers(
    ice_water_m,
    threshold=0.06,
)

# Fix the start by matching the interfce with weekly observations

# Interpolate weekly measurements onto the SIMBA timestamps
weekly_hi_interp = (-weekly_ice.hi).interp(time=ice_water_m.time)

# Replace ice_water_m before February 25
ice_water_m = ice_water_m.where(
    ice_water_m.time >= np.datetime64("2025-02-25"),
    weekly_hi_interp
)

# Replace ice_water_m after April 9
ice_water_m = ice_water_m.where(
    ice_water_m.time <= np.datetime64("2025-04-09"),
    weekly_hi_interp
)

# --------------------------------------------------
# Final clamped ice-water interface
# --------------------------------------------------

# Smooth ice-water interface - 5-day smoothing twice
ice_water_smooth = (
    ice_water_m
    .rolling(time=5, center=True, min_periods=1)
    .mean()
    .rolling(time=5, center=True, min_periods=1)
    .mean()
)

# Interpolate onto the temperature timestamps
ice_water_smooth = ice_water_smooth.interp(time=da_temp.time)


ice_water_smooth = ice_water_smooth.ffill("time").bfill("time")


# ============================================================
# SNOW-AIR
# ============================================================

snow_air_raw = detect_snow_air_from_heating(
    simba["del1"],
    simba["del4"],
    cfg,
    threshold=0.04,
    window=5,
    node_min=50,
    node_max=80,
)

snow_air_m = node_to_z(snow_air_raw, cfg)

# Interpolate weekly measurements onto the SIMBA timestamps
weekly_hs_interp = (weekly_ice.hs).interp(time=snow_air_m.time)

# Replace ice_water_m over certain time periods
snow_air_m = snow_air_m.where(
    snow_air_m.time >= np.datetime64("2025-03-10"),
    weekly_hs_interp
)

# Convert to m
snow_air_m = node_to_z(snow_air_raw, cfg)

# First round of "smoothing"
snow_air_m = correct_interface_outliers(
    snow_air_m,
    threshold=0.06,
)

# Direct manual fixes

snow_air_m = apply_values(
    snow_air_m,
    {
        "2025-02-21": 0.14,
        "2025-02-22": 0.14,
        "2025-03-06": weekly_hs_interp.sel(time="2025-03-06").values,
        "2025-03-07": weekly_hs_interp.sel(time="2025-03-07").values,
        "2025-03-08": weekly_hs_interp.sel(time="2025-03-08").values,
        "2025-03-09": 0.2,
        "2025-03-14": 0.22,
        "2025-03-21": 0.24,
        "2025-03-22": weekly_hs_interp.sel(time="2025-03-22").values,
        "2025-03-23": weekly_hs_interp.sel(time="2025-03-23").values,
        "2025-03-24": weekly_hs_interp.sel(time="2025-03-24").values,
        "2025-03-25": 0.34,
        "2025-03-31": 0.28,
        "2025-04-02": 0.32,
        "2025-04-03": 0.3,
        "2025-04-04": 0.3,
        "2025-04-05": 0.28,
        "2025-04-06": 0.28,
        "2025-04-07": 0.26,

    },
)


snow_air_m = apply_ranges(
    snow_air_m,
    [
        (
            "2025-03-26",
            "2025-03-30",
            0.3,
        ),
    ],
)

snow_air_m = apply_ranges(
    snow_air_m,
    [
        (
            "2025-02-23",
            "2025-02-27",
            snow_air_m.sel(time="2025-03-01").values,
        ),
    ],
)

snow_air_m = apply_ranges(
    snow_air_m,
    [
        (
            "2025-04-09",
            None,
            0.34,
        ),
    ],
)

# Remove time of day stamp 
snow_air_m = snow_air_m.resample(time='1D').mean()

# ============================================================
# FINAL PRODUCTS
# ============================================================

da_temp = da_temp.sel(
    time=slice(
        cfg.start,
        cfg.end,
    )
)


snow_air = snow_air_m.sel(
    time=slice(
        cfg.start,
        cfg.end,
    )
)


snow_ice_daily = (
    snow_ice_daily
    .sel(
        time=slice(
            cfg.start,
            cfg.end,
        )
    )
)


snow_ice_smooth = (
    snow_ice_smooth
    .sel(
        time=slice(
            cfg.start,
            cfg.end,
        )
    )
)


ice_water_smooth = (
    ice_water_smooth
    .sel(
        time=slice(
            cfg.start,
            cfg.end,
        )
    )
)



# Ice temperature

temp_ice = da_temp.where(
    (
        da_temp.z
        < snow_ice_smooth
    )
    &
    (
        da_temp.z
        > ice_water_smooth
    ),
    np.nan,
)



# Snow thickness

H_snow = (
    snow_air_m
    - snow_ice_daily
)


H_snow = H_snow.where(
    H_snow > 0.01,
    0,
)



# Ice thickness

H_ice = (
    ice_water_smooth
    - snow_ice_smooth
)

# ============================================================
# SAVE FINAL SIMBA PRODUCTS
# ============================================================


interfaces = build_interface_dataset(

    temperature=da_temp,

    temp_ice=temp_ice,

    H_ice=H_ice,

    H_bottom=ice_water_smooth,

    snow_air=snow_air_m,

    snow_ice=snow_ice_daily,

    H_snow=H_snow,

    snow_ice_smoothed=snow_ice_smooth,

    year=cfg.year,
)


output_file = (
    Path("data")
    / str(cfg.year)
    / "SIMBA"
    / "processed"
    / f"SIMBA_interfaces_{cfg.year}.nc"
)


save_interface_dataset(
    interfaces,
    output_file,
)