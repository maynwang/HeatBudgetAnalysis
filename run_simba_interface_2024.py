from pathlib import Path

import numpy as np
import xarray as xr

from simba_interfaces import (
    SIMBAConfig,
    load_simba_data,
    correct_temperature,
    detect_snow_ice,
    detect_ice_water,
    detect_snow_air,
    apply_values,
    apply_ranges,
    smooth_snow_ice,
    smooth_ice_water,
    build_interface_dataset,
    save_interface_dataset,
    smooth_ice_water_clamped,
)


ROOT = Path(".")


# ============================================================
# CONFIG
# ============================================================

cfg = SIMBAConfig(
    year=2024,

    # Keep the longer processing window used by the original 2024 workflow.
    # The final H_bottom product is subset to Jan 26-Apr 15 below.
    start="2024-01-23",
    end="2024-04-15",

    calibration_start="2024-03-25",
    calibration_end="2024-04-10",
)


# ============================================================
# LOAD DATA
# ============================================================

simba = load_simba_data(
    ROOT,
    cfg,
)


weekly_ice = xr.open_dataset(
    "data/2024/SiteVisits/SiteVisits_Weekly_IceSnowWater.nc"
)


# ============================================================
# TEMPERATURE
# ============================================================

# Temperature data already use physical z (m) from SIMBA_temp.nc.
da_temp = correct_temperature(
    simba["temp"],
    cfg,
)


# ============================================================
# SNOW-ICE
# ============================================================

# Returns the detected snow-ice interface directly in metres.
snow_ice_m = detect_snow_ice(
    simba["del1"],
    simba["del4"],
    cfg,
)


# Preserve the original 2024 positional corrections.
snow_ice_m = snow_ice_m.copy()

snow_ice_m[:30] = snow_ice_m[0]

snow_ice_m[-9] = snow_ice_m[-10]


# Preserve the original treatment where negative snow-ice elevations
# were set to zero before the late-season manual corrections.
snow_ice_m = snow_ice_m.where(
    snow_ice_m > 0,
    0,
)


# Manual 2024 corrections.
snow_ice_m = apply_ranges(
    snow_ice_m,
    [
        (
            "2024-03-27",
            "2024-04-11",
            0.38,
        ),
    ],
)


snow_ice_m = apply_values(
    snow_ice_m,
    {
        "2024-04-12": 0.36,
        "2024-04-13": 0.34,
        "2024-04-14": 0.32,
        "2024-04-15": 0.28,
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


# Preserve the original late-season edge treatment.
fill_vals = snow_ice_6h.sel(
    time=slice(
        "2024-04-10",
        None,
    )
)


tt = snow_ice_smooth.sel(
    time=slice(
        "2024-04-10",
        None,
    )
).time.values


snow_ice_smooth.data[
    snow_ice_smooth.time.isin(tt)
] = fill_vals.values


# Preserve the original early edge fill.
first_notnan = np.where(
    ~np.isnan(
        snow_ice_smooth
    )
)[0][0]


snow_ice_smooth[:100] = (
    snow_ice_smooth[:100]
    .fillna(
        snow_ice_smooth[
            first_notnan
        ]
    )
)


# ============================================================
# ICE-WATER
# ============================================================

# Returns the detected ice-water interface directly in metres.
ice_water_m = detect_ice_water(
    simba["del0"],
    simba["del1"],
    cfg,
)


# Preserve the original manual node corrections, now expressed directly
# in metres using the historical 2024 conversion:
#     z = (176 cm - node * 2 cm) / 100
# node 116 -> -0.56 m
# node 117 -> -0.58 m
# node 118 -> -0.60 m
ice_water_m = apply_ranges(
    ice_water_m,
    [
        (
            "2024-02-27",
            "2024-02-28",
            -0.58,
        ),
    ],
)


ice_water_m = apply_values(
    ice_water_m,
    {
        "2024-02-22": -0.56,
        "2024-03-11": -0.60,
    },
)


# --------------------------------------------------
# Final clamped ice-water interface used in heat budget
# --------------------------------------------------

H_bottom = smooth_ice_water_clamped(
    ice_water_m,
    da_temp.time,
    clamp_value=-0.62,
    taper_len=20,
    spline_s=0.004,
    apply_2024_patch=True,
)


# Also reproduce the original twice-smoothed interface used for
# temperature masking and H_ice.
ice_water_smooth = smooth_ice_water(
    ice_water_m,
    da_temp.time,
    cfg,
)


# Original edge corrections.
ice_water_smooth[100:] = (
    ice_water_smooth[100:]
    .fillna(-0.62)
)


ice_water_smooth[14] = (
    ice_water_m[3]
    + 0.015
)


ice_water_smooth[14:20] = (
    ice_water_smooth[14:20]
    .interpolate_na(
        dim="time"
    )
)


# ============================================================
# SNOW-AIR
# ============================================================

snow_air_raw = detect_snow_air(
    da_temp,
    cfg,

    drop_times=[
        "2024-02-15T03:00:16.000000000",
        "2024-02-18T03:00:16.000000000",
        "2024-02-19T03:00:16.000000000",
        "2024-03-03T03:00:16.000000000",
    ],

    replacement_times=[
        "2024-02-15T09:00:16.000000000",
        "2024-02-18T09:00:16.000000000",
        "2024-02-19T09:00:16.000000000",
        "2024-03-03T09:00:16.000000000",
    ],
)


snow_air = snow_air_raw.copy()


# Direct manual fixes.
snow_air = apply_values(
    snow_air,
    {
        "2024-02-14": 0.20,

        # Original script had 0.85 and then immediately overwrote
        # it with 0.90, so 0.90 is the effective value.
        "2024-02-20": 0.90,

        "2024-02-21": 0.90,
        "2024-02-22": 0.80,
        "2024-02-26": 0.70,
        "2024-02-27": 0.79,
        "2024-02-28": 0.64,
        "2024-02-29": 0.80,
        "2024-03-10": 0.40,
        "2024-03-11": 0.45,
        "2024-03-12": 0.39,
        "2024-03-23": 0.70,
        "2024-03-29": 0.48,
        "2024-03-30": 0.46,
        "2024-03-31": 0.46,
        "2024-04-01": 0.46,
        "2024-04-03": 0.46,
        "2024-04-04": 0.46,
        "2024-04-05": 0.46,
        "2024-04-06": 0.46,
        "2024-04-07": 0.44,
    },
)


# February 4: use the February 3 interface.
snow_air.loc[dict(time="2024-02-04")] = (
    snow_air.sel(time="2024-02-03").item()
)


# March 13-15: use the weekly observation nearest March 12.
march_snow = weekly_ice["hs_snowStake_mean"].sel(
    time="2024-03-12",
    method="nearest",
).item()

snow_air.loc[
    dict(time=slice("2024-03-13", "2024-03-15"))
] = march_snow


# April 8-15: no dry snow remains, so the snow-air interface
# equals the snow-ice interface.
snow_air.loc[
    dict(time=slice("2024-04-08", "2024-04-15"))
] = snow_ice_m.sel(
    time=slice("2024-04-08", "2024-04-15")
).values


# ============================================================
# FINAL PRODUCTS
# ============================================================

da_temp = da_temp.sel(
    time=slice(
        cfg.start,
        cfg.end,
    )
)


snow_air = snow_air.sel(
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


# Ice temperature.
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


# Snow thickness.
H_snow = (
    snow_air
    - snow_ice_daily
)


H_snow = H_snow.where(
    H_snow > 0.01,
    0,
)


# Ice thickness.
# Preserve the historical sign convention used by the original workflow.
H_ice = (
    ice_water_smooth
    - snow_ice_smooth
)


# ============================================================
# SAVE FINAL SIMBA PRODUCTS
# ============================================================

# The heat-budget H_bottom product starts Jan 26 even though earlier
# observations are retained above for the smoothing procedure.
H_bottom = H_bottom.sel(
    time=slice("2024-01-26", "2024-04-15")
)


interfaces = build_interface_dataset(
    temperature=da_temp,
    temp_ice=temp_ice,
    H_ice=H_ice,
    H_bottom=H_bottom,
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
