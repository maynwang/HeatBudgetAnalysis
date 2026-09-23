from dataclasses import dataclass
from pathlib import Path

import glob
import pickle
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

    # SIMBA geometry
    offset_cm: float
    node_spacing_cm: float = 2.0
    max_node: int = 236

    # Calibration correction
    calibration_start: str = None
    calibration_end: str = None
    spline_s: float = 1.0

    # Snow-ice detection
    snow_ice_node_min: int = 0
    snow_ice_node_max: int = 100
    snow_ice_smooth_window: int = 4

    # Ice-water detection
    ice_water_node_min: int = 80
    ice_water_node_max: int = 125
    ice_water_gradient_threshold: float = 0.1
    ice_water_smooth_window: int = 5

    # Snow-air detection
    snow_air_threshold: float = 0.7



# ============================================================
# LOAD SIMBA FILES
# ============================================================

def simba_to_da(path):
    """
    Load all SIMBA files matching path into one DataArray.
    """

    files = sorted(
        glob.glob(str(path))
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No SIMBA files found for:\n{path}"
        )

    dfs = []

    for file in files:

        df = pd.read_fwf(
            file,
            header=None,
        )

        dfs.append(df)


    # replaces deprecated df.append()
    df = pd.concat(
        dfs,
        ignore_index=True,
    )


    time = pd.to_datetime(
        df.loc[:, 1].astype(str)
        + " "
        + df.loc[:, 2].astype(str)
    )


    values = (
        df.loc[:, 8:]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy()
    )


    da = xr.DataArray(
        values,
        dims=[
            "time",
            "node",
        ],
        coords={
            "time": time.values,
            "node": np.arange(
                values.shape[1]
            ),
        },
    )

    return da



def load_simba_data(
    root_dir,
    cfg,
):

    root_dir = Path(root_dir)

    data_dir = (
        root_dir
        / "data"
        / str(cfg.year)
        / "SIMBA"
        
    )


    da_temp = simba_to_da(
        data_dir
        / "TEMPDATA*"
    )

    da_del0 = simba_to_da(
        data_dir
        / "DELDATA0*"
    )

    da_del1 = simba_to_da(
        data_dir
        / "DELDATA1*"
    )

    da_del4 = simba_to_da(
        data_dir
        / "DELDATA4*"
    )


    # Remove bad bottom nodes
    da_temp = da_temp.isel(
        node=slice(
            0,
            cfg.max_node,
        )
    )

    da_del0 = da_del0.isel(
        node=slice(
            0,
            cfg.max_node,
        )
    )

    da_del1 = da_del1.isel(
        node=slice(
            0,
            cfg.max_node,
        )
    )

    da_del4 = da_del4.isel(
        node=slice(
            0,
            cfg.max_node,
        )
    )


    return {
        "temp": da_temp,
        "del0": da_del0,
        "del1": da_del1,
        "del4": da_del4,
    }



# ============================================================
# TEMPERATURE CALIBRATION CORRECTION
# ============================================================


def correct_temperature(
    da_temp,
    cfg,
):

    # Mean profile during a relatively stable period
    T_meas = (
        da_temp
        .sel(
            time=slice(
                cfg.calibration_start,
                cfg.calibration_end,
            )
        )
        .mean("time")
    )


    x = T_meas.node.values
    y = T_meas.values


    valid = np.isfinite(y)


    cs = UnivariateSpline(
        x[valid],
        y[valid],
        s=cfg.spline_s,
    )


    T_spl = xr.DataArray(
        cs(x),
        dims=["node"],
        coords={
            "node": T_meas.node,
        },
    )


    # Sensor-specific calibration offset
    T_err = (
        T_meas
        - T_spl
    )


    da_temp_corrected = (
        da_temp
        - T_err
    )


    # Convert node coordinate to z
    z = (
        -da_temp_corrected.node
        * cfg.node_spacing_cm
        + cfg.offset_cm
    ) / 100


    da_temp_z = (
        da_temp_corrected
        .assign_coords(
            node=z
        )
        .rename({
            "node": "z"
        })
    )


    return da_temp_z



# ============================================================
# HELPER
# ============================================================

def node_to_z(
    node,
    cfg,
):

    return (
        -node
        * cfg.node_spacing_cm
        + cfg.offset_cm
    ) / 100



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

    interface = interface.copy()

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

    interface = interface.copy()

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

    T_30_120 = (
        da_del1
        / da_del4
    )


    T_30_120 = T_30_120.isel(
        node=slice(
            cfg.snow_ice_node_min,
            cfg.snow_ice_node_max,
        )
    )


    gradient = (
        T_30_120
        .differentiate("node")
    )


    snow_ice_node = (
        gradient
        .argmax("node")
        + 1
    )


    return snow_ice_node



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

def detect_ice_water(da_del0, da_del1, cfg):

    T_0_30 = da_del0 / da_del1

    T_0_30_icewater = T_0_30.isel(
        node=slice(
            cfg.ice_water_node_min,
            cfg.ice_water_node_max,
        )
    )

    T_0_30_grad = T_0_30_icewater.differentiate("node")

    def last_nonzero(arr, axis, invalid_val=np.nan):
        mask = arr != 0
        val = (
            arr.shape[axis]
            - np.flip(mask, axis=axis).argmax(axis=axis)
            - 1
        )
        return xr.where(mask.any(axis=axis), val, invalid_val)

    max_grads = T_0_30_grad.where(
        T_0_30_grad > cfg.ice_water_gradient_threshold,
        other=0,
    )

    ice_water = (
        last_nonzero(max_grads, axis=1)
        + cfg.ice_water_node_min
    )

    return ice_water


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
    threshold=0.05,
    window=5,
    node_min=0,
    node_max=100,
):
    """
    Detect the snow-air interface from the first sustained region
    of strong HT30/HT120 gradients, searching top-down.

    Returns the detected node number.
    """

    # Calculate heating ratio
    T_30_120 = da_del1 / da_del4

    # Restrict to the upper part of the chain
    T_30_120 = T_30_120.isel(
        node=slice(node_min, node_max)
    )

    # Calculate gradient along the chain
    gradient = T_30_120.diff("node")

    # Rolling mean of absolute gradient
    gradient_smooth = (
        abs(gradient)
        .rolling(
            node=window,
            center=True,
            min_periods=window,
        )
        .mean()
    )

    # Identify regions exceeding the threshold
    steep = gradient_smooth > threshold

    # Find the FIRST qualifying node from the top downward
    snow_air_node = (
        gradient_smooth.node
        .where(steep)
        .min("node", skipna=True)
    )

    return snow_air_node

from pathlib import Path
import xarray as xr


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

