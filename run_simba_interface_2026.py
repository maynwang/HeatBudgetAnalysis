from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.ion()

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
    detect_snow_air,
)


ROOT = Path(".")


# ============================================================
# CONFIG
# ============================================================


cfg = SIMBAConfig(
    year=2026,
    start="2026-02-03",
    end="2026-04-21",

    calibration_start="2026-03-25",
    calibration_end="2026-04-10",

    # Snow-ice detection
    snow_ice_search_min_z_m=0.0,
    snow_ice_search_max_z_m=0.4,

    # Ice-water detection
    ice_water_search_min_z_m= -1.0,
    ice_water_search_max_z_m= 0.0,

    snow_air_heating_search_min_z_m=0.1,
    snow_air_heating_search_max_z_m=0.5,
    snow_air_heating_gradient_threshold_per_m=2.0,
    snow_air_heating_smooth_depth_m=0.10,

    snow_air_threshold = 5,
    snow_air_search_max_z_m = 0.4,
    snow_air_search_min_z_m = 0.2

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

# Temp correction introduces 2 strange lines in the temp time series. For not comment it out
# da_temp = correct_temperature(
#     simba["temp"],
#     cfg,
# )

# The minus 0 is a hacky way to drop the temp label of the DataArray
da_temp = simba["temp"].copy() - 0


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

im.set_clim(-20, 2)


# ============================================================
# SNOW-ICE
# ============================================================

snow_ice_m = detect_snow_ice(
    simba["del1"],
    simba["del4"],
    cfg,
)


# Plot the heating ratio for manual corrections
T_30_120 = simba["del0"] / simba["del4"]

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

# Replace snow-air interface before March 12 with 0
snow_ice_m = snow_ice_m.where(
    snow_ice_m.time >= np.datetime64("2026-03-10"),
    0,
)


# Correct isolated interface spikes
snow_ice_m = correct_interface_outliers(
    snow_ice_m,
    threshold=0.04,
)


snow_ice_m = apply_values(
    snow_ice_m,
    {
        "2026-04-02": snow_ice_m.sel(time="2026-04-01").values,
        "2026-04-03": snow_ice_m.sel(time="2026-04-01").values,
        "2026-02-24": 0,
        "2026-04-19": snow_ice_m.sel(time="2026-04-18").values,
        "2026-04-21": 0.24,
    },
)

snow_ice_m = apply_ranges(
    snow_ice_m,
    [
        (
            "2026-03-29",
            "2026-04-01",
            snow_ice_m.sel(time="2026-03-28").values,
        ),
    ],
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

snow_ice_smooth = snow_ice_smooth.interp(
    time=da_temp.time
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

snow_air_m = detect_snow_air(
    da_temp,
    cfg,
)


# Interpolate weekly measurements onto the SIMBA timestamps
weekly_hs_interp = weekly_ice.hs_snowStake_mean.interp(time=snow_air_m.time)

# No manual fixing for 2026 data needed (for now)

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

save_interface_dataset(
    interfaces,
    output_file,
)
