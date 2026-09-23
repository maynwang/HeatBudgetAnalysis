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
    node_to_z,
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

    start="2024-01-23",
    end="2024-04-15",

    # IMPORTANT: 2024 offset
    offset_cm=176,

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

da_temp = correct_temperature(
    simba["temp"],
    cfg,
)



# ============================================================
# SNOW-ICE
# ============================================================

snow_ice_raw = detect_snow_ice(
    simba["del1"],
    simba["del4"],
    cfg,
)


snow_ice = snow_ice_raw.copy()


# Existing 2024 manual node corrections
snow_ice[:30] = snow_ice[0]

snow_ice[-9] = snow_ice[-10]


# Node -> metres
snow_ice_m = node_to_z(
    snow_ice,
    cfg,
)


snow_ice_m = snow_ice_m.where(
    snow_ice_m > 0,
    0,
)


# Manual 2024 corrections

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


# Preserve your original end treatment
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


# Preserve original early edge fill
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

ice_water_raw = detect_ice_water(
    simba["del0"],
    simba["del1"],
    cfg,
)


ice_water = ice_water_raw.copy()


# Manual node corrections

ice_water = apply_ranges(
    ice_water,
    [
        (
            "2024-02-27",
            "2024-02-28",
            117,
        ),
    ],
)


ice_water = apply_values(
    ice_water,
    {
        "2024-02-22": 116,
        "2024-03-11": 118,
    },
)


ice_water_m = node_to_z(
    ice_water,
    cfg,
)

# --------------------------------------------------
# Final clamped ice-water interface
# --------------------------------------------------

H_bottom = smooth_ice_water_clamped(
    ice_water_m,
    da_temp.time,
    clamp_value=-0.62,
    taper_len=20,
    spline_s=0.004,
    apply_2024_patch=True
)

ice_water_smooth = smooth_ice_water(
    ice_water_m,
    da_temp.time,
    cfg,
)


# Original edge corrections

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


# Direct manual fixes

snow_air = apply_values(
    snow_air,
    {
        "2024-02-14": 0.20,

        # Original script had 0.85 and then immediately
        # overwrote it with 0.90, so 0.90 is the
        # effective value.
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

# February 4: use the February 3 interface
snow_air.loc[dict(time="2024-02-04")] = (
    snow_air.sel(time="2024-02-03").item()
)

# March 13–15: use the weekly observation nearest March 12
march_snow = weekly_ice["hs"].sel(
    time="2024-03-12",
    method="nearest",
).item()

snow_air.loc[
    dict(time=slice("2024-03-13", "2024-03-15"))
] = march_snow

# April 8–15: no dry snow remains
# Snow-air interface equals snow-ice interface

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

H_ice = (
    ice_water_smooth
    - snow_ice_smooth
)

# ============================================================
# SAVE FINAL SIMBA PRODUCTS
# ============================================================

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


# output_file = (
#     Path("data")
#     / str(cfg.year)
#     / "SIMBA"
#     / "processed"
#     / f"SIMBA_interfaces_{cfg.year}.nc"
# )


# save_interface_dataset(
#     interfaces,
#     output_file,
# )