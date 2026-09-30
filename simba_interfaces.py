from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import UnivariateSpline


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SIMBAConfig:

    year: int

    start: str
    end: str

    # Temperature calibration correction
    calibration_start: str = None
    calibration_end: str = None
    spline_s: float = 1.0

    # Vertical search ranges are expressed in metres below the
    # top thermistor. With 2 cm thermistor spacing, these reproduce
    # the original node-space windows exactly:
    #   snow-ice: 0:100 nodes -> 0.00 <= depth < 2.00 m
    #   ice-water: 80:125 nodes -> 1.60 <= depth < 2.50 m

    # Snow-ice detection
    snow_ice_search_min_z_m: float = 0.0
    snow_ice_search_max_z_m: float = 1
    snow_ice_smooth_window: int = 4

    # Ice-water detection
    ice_water_search_min_z_m: float = -1.0
    ice_water_search_max_z_m: float = 0.0
    ice_water_gradient_threshold_per_m: float = 5.0
    ice_water_smooth_window: int = 5

    # Snow-air detection from heating ratios
    snow_air_heating_search_min_z_m: float = 0.0
    snow_air_heating_search_max_z_m: float = 1.0
    snow_air_heating_gradient_threshold_per_m: float = 2.5
    snow_air_heating_smooth_depth_m: float = 0.10


# ============================================================
# LOAD SIMBA FILES
# ============================================================


def _prepare_simba_vertical_coordinate(da):
    """Return a SIMBA DataArray ordered from the top sensor downward in z."""

    if "z" not in da.dims:
        raise ValueError(
            "SIMBA DataArray must use physical height 'z' as a dimension."
        )

    # Standardize the ordering used by all detection routines:
    # largest z (top thermistor) -> smallest z (bottom thermistor).
    return da.sortby("z", ascending=False)



def load_simba_data(
    root_dir,
    cfg,
):
    """
    Load SIMBA temperature and heating-cycle data from NetCDF files.

    Expected files:
        <root_dir>/<year>/SIMBA/SIMBA_temp.nc
        <root_dir>/<year>/SIMBA/SIMBA_del.nc

    Both files are expected to contain a physical vertical coordinate ``z``
    in metres. No node-to-z conversion or deployment offset is required.

    Returns
    -------
    dict
        {
            "temp": temperature DataArray,
            "del0": DELDATA 0 s DataArray,
            "del1": DELDATA 30 s DataArray,
            "del4": DELDATA 120 s DataArray,
        }
    """

    simba_dir = (
        Path(root_dir)
        / "data"
        / str(cfg.year)
        / "SIMBA"
    )

    temp = xr.open_dataset(
        simba_dir / "SIMBA_temp.nc"
    )["temp"]

    del_ds = xr.open_dataset(
        simba_dir / "SIMBA_del.nc"
    )

    return {
        "temp": _prepare_simba_vertical_coordinate(temp),
        "del0": _prepare_simba_vertical_coordinate(del_ds["del0"]),
        "del1": _prepare_simba_vertical_coordinate(del_ds["del1"]),
        "del4": _prepare_simba_vertical_coordinate(del_ds["del4"]),
    }


# ============================================================
# TEMPERATURE CALIBRATION CORRECTION
# ============================================================


def correct_temperature(
    da_temp,
    cfg,
):
    """
    Correct persistent sensor-to-sensor temperature offsets.

    The calibration spline is fit directly in physical z space. The input and
    output both retain ``z`` in metres; no node-to-z conversion is performed.
    """

    calibration = da_temp.sel(
        time=slice(
            cfg.calibration_start,
            cfg.calibration_end,
        )
    )

    if calibration.sizes.get("time", 0) == 0:
        raise ValueError(
            "No SIMBA temperature data found in calibration period "
            f"{cfg.calibration_start} to {cfg.calibration_end}. "
            f"Available data span {da_temp.time.min().values} to "
            f"{da_temp.time.max().values}."
        )

    # Mean profile during a relatively stable period.
    T_meas = calibration.mean("time")

    valid = np.isfinite(T_meas.values)
    n_valid = int(valid.sum())

    if n_valid < 4:
        raise ValueError(
            "Not enough valid thermistors to fit the cubic temperature "
            f"correction spline: {n_valid} valid z levels."
        )

    # UnivariateSpline requires increasing x. SIMBA z is stored top-down,
    # so sort the valid calibration profile by z before fitting.
    x = T_meas.z.values[valid]
    y = T_meas.values[valid]
    order = np.argsort(x)

    cs = UnivariateSpline(
        x[order],
        y[order],
        s=cfg.spline_s,
    )

    T_spl = xr.DataArray(
        cs(T_meas.z.values),
        dims=["z"],
        coords={"z": T_meas.z},
    )

    # Sensor-specific calibration offset.
    T_err = T_meas - T_spl

    return da_temp - T_err



# ============================================================
# HELPER
# ============================================================

def _select_z_range(
    da,
    min_z_m,
    max_z_m,
):
    """
    Select a vertical interval using the physical z coordinate.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with vertical dimension 'z'.
    min_z_m : float
        Minimum z coordinate in metres.
    max_z_m : float
        Maximum z coordinate in metres.
    """

    if "z" not in da.dims:
        raise ValueError(
            "Expected a physical vertical dimension named 'z'."
        )

    if max_z_m <= min_z_m:
        raise ValueError(
            "max_z_m must be greater than min_z_m."
        )

    return da.where(
        (da.z >= min_z_m)
        & (da.z < max_z_m),
        drop=True,
    )


def _median_vertical_spacing_m(da):
    """Return the median absolute thermistor spacing in metres."""

    dz = np.diff(da.z.values.astype(float))
    dz = np.abs(dz[np.isfinite(dz)])

    if dz.size == 0:
        raise ValueError("Could not determine SIMBA vertical spacing from z.")

    return float(np.median(dz))



def apply_values(
    interface,
    corrections,
):
    """
    Apply manual values:

    corrections = {
        "2024-03-27": 0.38,
        ...
    }
    """

    # Some detected interfaces originate from a coordinate variable
    # (e.g. selecting values from the z coordinate).  Coordinate-backed
    # IndexVariables are immutable in xarray, so explicitly create a
    # writable DataArray before applying manual corrections.
    interface = xr.DataArray(
        interface.values.copy(),
        coords={
            dim: interface[dim].values
            for dim in interface.dims
        },
        dims=interface.dims,
        name=interface.name,
        attrs=interface.attrs,
    )

    for date, value in corrections.items():

        interface.loc[date] = value


    return interface



def apply_ranges(
    interface,
    corrections,
):
    """
    corrections = [
        ("2024-03-27", "2024-04-11", 0.38)
    ]
    """

    # Ensure the interface is writable even if it originated from
    # an xarray coordinate / IndexVariable.
    interface = xr.DataArray(
        interface.values.copy(),
        coords={
            dim: interface[dim].values
            for dim in interface.dims
        },
        dims=interface.dims,
        name=interface.name,
        attrs=interface.attrs,
    )

    for start, end, value in corrections:

        interface.loc[
            start:end
        ] = value


    return interface



# ============================================================
# SNOW-ICE
# ============================================================

def detect_snow_ice(
    da_del1,
    da_del4,
    cfg,
):

    T_30_120 = da_del1 / da_del4

    T_30_120 = _select_z_range(
        T_30_120,
        cfg.snow_ice_search_min_z_m,
        cfg.snow_ice_search_max_z_m,
    )

    gradient = T_30_120.differentiate("z")

    # z decreases downward, so the old positive
    # node-space gradient becomes a negative z-gradient
    interface_index = gradient.argmin("z")

    # Preserve original +1-node (= 2 cm downward) behavior
    interface_index = np.minimum(
        interface_index + 1,
        T_30_120.sizes["z"] - 1,
    )

    snow_ice_z = xr.DataArray(
        T_30_120.z.values[
            interface_index.values
        ],
        coords={
            "time": T_30_120.time.values
        },
        dims=["time"],
        name="snow_ice",
    )

    return snow_ice_z



def smooth_snow_ice(
    snow_ice_m,
    temp_time,
    cfg,
):

    # Daily interface
    snow_ice_daily = (
        snow_ice_m
        .resample(
            time="1D"
        )
        .mean()
    )


    # Back to temperature timestamps
    snow_ice_6h = (
        snow_ice_daily
        .interp(
            time=temp_time,
            method="linear",
        )
    )


    snow_ice_smooth = (
        snow_ice_6h
        .rolling(
            time=cfg.snow_ice_smooth_window,
            center=True,
        )
        .mean()
    )


    return (
        snow_ice_daily,
        snow_ice_6h,
        snow_ice_smooth,
    )



# ============================================================
# ICE-WATER
# ============================================================

def detect_ice_water(
    da_del0,
    da_del1,
    cfg,
):
    """
    Detect the ice-water interface directly in physical z space.

    The original criterion was dR/dnode > 0.1.

    Since:
        1 node = 0.02 m
        z decreases downward,

    the equivalent criterion is:

        dR/dz < -5 m^-1

    The deepest qualifying gradient (smallest z) is returned
    as the ice-water interface elevation.
    """

    # Heating ratio
    T_0_30 = da_del0 / da_del1

    # Restrict search to the specified physical z range
    T_0_30_icewater = _select_z_range(
        T_0_30,
        cfg.ice_water_search_min_z_m,
        cfg.ice_water_search_max_z_m,
    )

    # Vertical gradient in physical coordinates
    T_0_30_grad = (
        T_0_30_icewater
        .differentiate("z")
    )

    # Because dz/dnode = -0.02 m,
    # this becomes:
    #     dR/dz < -5 m^-1
    steep = (
        T_0_30_grad
        < -cfg.ice_water_gradient_threshold_per_m
    )

    # Since z decreases downward, this corresponds
    # to the smallest qualifying z value.
    ice_water_coord = (
        T_0_30_grad.z
        .where(steep)
        .min("z", skipna=True)
    )

    # Make an ordinary writable DataArray
    ice_water_z = xr.DataArray(
        ice_water_coord.values.copy(),
        coords={
            "time": T_0_30_grad.time.values
        },
        dims=["time"],
        name="ice_water",
    )

    return ice_water_z



def smooth_ice_water(
    ice_water_m,
    temp_time,
    cfg,
):

    roll1 = (
        ice_water_m
        .rolling(
            time=cfg.ice_water_smooth_window,
            center=True,
        )
        .mean()
    )


    roll2 = (
        roll1
        .rolling(
            time=cfg.ice_water_smooth_window,
            center=True,
        )
        .mean()
    )


    ice_water_smooth = (
        roll2
        .interp(
            time=temp_time,
            method="linear",
        )
    )


    return ice_water_smooth



# ============================================================
# SNOW-AIR
# ============================================================

def detect_snow_air(
    da_temp,
    cfg,
    drop_times=None,
    replacement_times=None,
    ):
    '''
    Calculate the air temperature mean using the top of the thermistor chain, then find where the temperature deviates from the mean (snow-air interface)

    Use for 2024
    '''

    # Original nighttime hours
    hours = [
        3,
        1,
        11,
        10,
        0,
    ]


    da_temp_night = da_temp.sel(
        time=da_temp.time.dt.hour.isin(
            hours
        )
    )


    if drop_times is not None:

        da_temp_night = (
            da_temp_night
            .drop_sel(
                time=drop_times
            )
        )


    if replacement_times is not None:

        replacements = da_temp.loc[
            dict(
                time=replacement_times
            )
        ]


        da_temp_night = xr.concat(
            [
                da_temp_night,
                replacements,
            ],
            dim="time",
        ).sortby("time")


    T_snowair = da_temp_night.sel(
        z=slice(
            1,
            0,
        )
    )


    Tair_mean = (
        T_snowair
        .sel(
            z=slice(
                1.4,
                1,
            )
        )
        .mean("z")
    )


    difference = (
        T_snowair
        - Tair_mean
    )


    detected = difference.where(
        abs(difference)
        > cfg.snow_air_threshold,
        0,
    )


    index = (
        detected != 0
    ).argmax("z")


    snow_air = (
        T_snowair.z
        .isel(
            z=index
        )
    )


    snow_air = (
        snow_air
        .resample(
            time="1D"
        )
        .mean()
    )

    return snow_air

def detect_snow_air_from_heating(
    da_del1,
    da_del4,
    cfg,
):
    """
    Detect the snow-air interface from HT30/HT120
    directly in physical z space.

    The first qualifying location from the top downward
    is returned directly as z in metres.
    """

    T_30_120 = da_del1 / da_del4

    # Restrict search to absolute physical z range
    T_30_120 = _select_z_range(
        T_30_120,
        cfg.snow_air_heating_search_min_z_m,
        cfg.snow_air_heating_search_max_z_m,
    )

    # Forward difference expressed per metre
    dz = T_30_120.z.diff("z")

    gradient = (
        T_30_120.diff("z")
        / dz
    )

    # Convert smoothing depth in metres to number of sensors
    spacing_m = _median_vertical_spacing_m(
        T_30_120
    )

    window_points = max(
        1,
        int(
            round(
                cfg.snow_air_heating_smooth_depth_m
                / spacing_m
            )
        ),
    )

    gradient_smooth = (
        abs(gradient)
        .rolling(
            z=window_points,
            center=True,
            min_periods=window_points,
        )
        .mean()
    )

    steep = (
        gradient_smooth
        > cfg.snow_air_heating_gradient_threshold_per_m
    )

    # z decreases downward.
    # Therefore the FIRST qualifying point from the top
    # is the largest qualifying z value.
    snow_air_coord = (
        gradient_smooth.z
        .where(steep)
        .max("z", skipna=True)
    )

    # Return an ordinary writable DataArray
    snow_air_z = xr.DataArray(
        snow_air_coord.values.copy(),
        coords={
            "time": gradient_smooth.time.values
        },
        dims=["time"],
        name="snow_air",
    )

    return snow_air_z


def build_interface_dataset(
    temperature,
    temp_ice,
    H_ice,
    H_bottom,
    snow_air,
    snow_ice,
    H_snow,
    snow_ice_smoothed,
    year,
):
    """
    Combine processed SIMBA interface variables into one Dataset.

    Variables retain their native temporal resolution:
        time_6h    : SIMBA-temperature-resolution variables
        time_daily : daily interface variables
    """

    ds = xr.Dataset({

        # 6-hourly variables
        "temperature":
            temperature.rename(
                {"time": "time_6h"}
            ),

        "temp_ice":
            temp_ice.rename(
                {"time": "time_6h"}
            ),

        "H_ice":
            H_ice.rename(
                {"time": "time_6h"}
            ),

        "H_bottom":
            H_bottom.rename(
                {"time": "time_6h"}
            ),

        "snow_ice_smoothed":
            snow_ice_smoothed.rename(
                {"time": "time_6h"}
            ),


        # Daily variables
        "snow_air":
            snow_air.rename(
                {"time": "time_daily"}
            ),

        "snow_ice":
            snow_ice.rename(
                {"time": "time_daily"}
            ),

        "H_snow":
            H_snow.rename(
                {"time": "time_daily"}
            ),
    })


    # Some useful metadata
    ds.attrs["year"] = year

    ds["temperature"].attrs["units"] = "degC"
    ds["temp_ice"].attrs["units"] = "degC"

    ds["H_ice"].attrs["units"] = "m"
    ds["H_bottom"].attrs["units"] = "m"
    ds["snow_air"].attrs["units"] = "m"
    ds["snow_ice"].attrs["units"] = "m"
    ds["snow_ice_smoothed"].attrs["units"] = "m"
    ds["H_snow"].attrs["units"] = "m"


    return ds

def correct_interface_outliers(interface, threshold=0.06):
    """
    Correct isolated spikes in a SIMBA interface time series.

    If a point differs from the previous point by more than
    threshold, but the following point returns close to the
    previous value, replace the spike with the average of
    its two neighbours.

    Parameters
    ----------
    interface : xr.DataArray
        Interface elevation (m), with a time dimension.
    threshold : float
        Maximum permitted jump (m). Default is 0.06 m.

    Returns
    -------
    xr.DataArray
        Corrected interface with original coordinates preserved.
    """

    original = interface.values.copy()
    corrected = original.copy()

    for i in range(1, len(original) - 1):

        previous = original[i - 1]
        current = original[i]
        following = original[i + 1]

        # Skip missing values
        if not np.all(np.isfinite([previous, current, following])):
            continue

        # Detect an isolated spike
        if (
            abs(current - previous) > threshold
            and abs(following - previous) <= threshold
        ):
            corrected[i] = (previous + following) / 2

    return interface.copy(data=corrected)


def smooth_ice_water_clamped(
    ice_water_m,
    temp_time,
    clamp_value,
    taper_len=20,
    spline_s=0.004,
    apply_2024_patch=False,
):
    """
    Smooth the ice-water interface using a clamped spline.

    The localized 2024 correction can be disabled for other years.
    """

    from scipy.interpolate import splrep, splev

    # Convert time to seconds since epoch
    x_dt = ice_water_m.time.values
    x_all = pd.to_datetime(x_dt).astype(np.int64) / 1e9
    y_all = ice_water_m.values

    # Find where the flat region begins
    flat_indices = np.flatnonzero(y_all == clamp_value)

    if len(flat_indices) == 0:
        raise ValueError(
            f"No ice-water values equal {clamp_value} m. "
            "Check whether clamping is appropriate for this year."
        )

    flat_start_idx = flat_indices[0]
    fit_end_idx = flat_start_idx + taper_len

    if fit_end_idx > len(y_all):
        raise ValueError(
            "Not enough observations for the spline taper."
        )

    # Split into fitting and flat regions
    x_fit = x_all[:fit_end_idx]
    y_fit = y_all[:fit_end_idx]

    x_flat = x_all[fit_end_idx:]
    y_flat = np.full_like(x_flat, clamp_value)

    # Fit smoothing spline
    f_smooth = splrep(
        x_fit,
        y_fit,
        s=spline_s,
    )

    y_spline = splev(x_fit, f_smooth)

    # Cosine taper
    taper_weights = 0.5 * (
        1 + np.cos(np.linspace(0, np.pi, taper_len))
    )

    y_blended = y_spline.copy()

    y_blended[-taper_len:] = (
        taper_weights * y_spline[-taper_len:]
        + (1 - taper_weights) * clamp_value
    )

    # Combine spline and flat region
    y_smooth_full = np.concatenate([
        y_blended,
        y_flat,
    ])

    # Convert back to xarray
    clamped = xr.DataArray(
        y_smooth_full,
        coords={"time": x_dt},
        dims=["time"],
        name="H_bottom",
    )

    # Hard clamp
    clamped_ = clamped.where(
        clamped >= clamp_value,
        clamp_value,
    )

    # Apply the original localized correction only for 2024
    if apply_2024_patch:

        if clamped_.sizes["time"] < 70:
            raise ValueError(
                "The 2024 correction requires at least 70 time points."
            )

        clamped_roll = (
            clamped_.isel(time=slice(40, 70))
            .rolling(time=5, center=True)
            .mean()
        )

        # Assign by position rather than mismatched time coordinates
        corrected = clamped_.values.copy()

        corrected[45:65] = (
            clamped_roll
            .isel(time=slice(5, -5))
            .values
        )

        clamped_ = clamped_.copy(data=corrected)

    # Interpolate to temperature timestamps
    clamped_interp = clamped_.interp(
        time=temp_time,
        method="linear",
    )

    return clamped_interp


def save_interface_dataset(
    ds,
    output_file,
):
    """
    Save processed SIMBA interfaces to NetCDF.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ds.to_netcdf(
        output_file
    )

    print(
        f"Saved SIMBA interfaces to:\n{output_file}"
    )

