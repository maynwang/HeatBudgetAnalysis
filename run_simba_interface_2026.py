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
    apply_values,
    apply_ranges,
    smooth_snow_ice,
    build_interface_dataset,
    save_interface_dataset,
    correct_interface_outliers,
    detect_snow_air_from_heating,
)


ROOT = Path(".")


# ============================================================
# CONFIG
# ============================================================

cfg = SIMBAConfig(
    year=2026,
    start="2026-02-06",
    end="2026-04-21",

    calibration_start="2026-03-25",
    calibration_end="2026-04-10",

    # Snow-ice detection
    snow_ice_search_min_z_m=0.0,
    snow_ice_search_max_z_m=0.4,

    # Ice-water detection
    ice_water_search_min_z_m= -1.0,
    ice_water_search_max_z_m= 0.0,

)

# ============================================================
# LOAD DATA
# ============================================================

simba = load_simba_data(
    ROOT,
    cfg,
)

weekly_ice = xr.open_dataset(
    "data/2026/SiteVisits/SiteVisits_Weekly_IceSnowWater.nc"
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

im.set_clim(-10, 2)


# ============================================================
# SNOW-ICE
# ============================================================

snow_ice_m = detect_snow_ice(
    simba["del1"],
    simba["del4"],
    cfg,
)


# Plot the heating ratio for manual corrections
T_30_120 = simba["del1"] / simba["del4"]

fig, ax = plt.subplots(figsize=(12, 6))

im = ax.pcolormesh(
    T_30_120.time.values,
    T_30_120.z.values,
    T_30_120.transpose("z", "time").values,
    shading="auto",
    cmap="RdBu_r",
)

fig.colorbar(im, ax=ax, label="HT30/HT120")
ax.set_ylabel("z (m)")
ax.set_xlabel("Time")


# Manual 2026 corrections

# Correct isolated interface spikes
snow_ice_m = correct_interface_outliers(
    snow_ice_m,
    threshold=0.05,
)


snow_ice_m = apply_values(
    snow_ice_m,
    {
        "2026-04-02": snow_ice_m.sel(time="2026-04-01").values,
        "2026-04-03": snow_ice_m.sel(time="2026-04-01").values,
        "2026-02-24": 0,
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

# Fill NaNs at the beginning and end with first and last values
snow_ice_smooth = (
    snow_ice_smooth
    .ffill(dim="time")
    .bfill(dim="time")
)


# ============================================================
# ICE-WATER
# ============================================================

# Plot the heating ratio for manual corrections
T_0_30 = simba["del0"] / simba["del1"]

# detect_ice_water now returns physical z directly in metres.
ice_water_m = detect_ice_water(
    simba["del0"],
    simba["del1"],
    cfg,
)


# Correct isolated interface spikes
ice_water_m = correct_interface_outliers(
    ice_water_m,
    threshold=0.05,
)


ice_water_m = apply_values(
    ice_water_m,
    {
        "2026-03-02": ice_water_m.sel(time="2026-03-01").values,
        "2026-03-15": ice_water_m.sel(time="2026-03-14").values,
        "2026-03-16": ice_water_m.sel(time="2026-03-14").values,
        "2026-03-21": ice_water_m.sel(time="2026-03-20").values,
        "2026-03-22": ice_water_m.sel(time="2026-03-20").values,
        "2026-03-29": ice_water_m.sel(time="2026-03-30").values,
        "2026-04-12": ice_water_m.sel(time="2026-04-11").values,
    },
)


# Interpolate weekly measurements onto the SIMBA timestamps
weekly_hi_interp = (-weekly_ice.hi).interp(time=ice_water_m.time)


# --------------------------------------------------
# Final smoothed ice-water interface
# --------------------------------------------------

# Smooth ice-water interface - 5-point smoothing twice
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

# detect_snow_air_from_heating now returns physical z directly in metres.
# The 2025-specific search range/threshold are defined in cfg above.
snow_air_m = detect_snow_air_from_heating(
    simba["del1"],
    simba["del4"],
    cfg,
)


# Interpolate weekly measurements onto the SIMBA timestamps
weekly_hs_interp = weekly_ice.hs.interp(time=snow_air_m.time)

# Replace snow-air interface before March 10 with weekly observations
snow_air_m = snow_air_m.where(
    snow_air_m.time >= np.datetime64("2025-03-10"),
    weekly_hs_interp,
)


# First round of smoothing / isolated-spike correction
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

# Remove time-of-day stamp
snow_air_m = snow_air_m.resample(time="1D").mean()


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
    snow_air
    - snow_ice_daily
)

H_snow = H_snow.where(
    H_snow > 0.01,
    0,
)


# Ice thickness
# Preserve historical sign convention: z decreases downward, so this is negative.
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
    snow_air=snow_air,
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

# save_interface_dataset(
#     interfaces,
#     output_file,
# )
