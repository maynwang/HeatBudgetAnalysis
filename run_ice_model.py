"""
Year-agnostic 1-D Semtner/Kirillov sea-ice model.

The model is intentionally separate from the observational heat-budget
workflow. Model physics and constants are defined locally; the only quantity
read from the heat-budget output is the observationally derived ocean heat flux F_w.

Run, for example:

    python run_semtner_model.py --year 2024
    python run_semtner_model.py --year 2025
    python run_semtner_model.py --year 2026

Required processed inputs for year YYYY
---------------------------------------
- data/YYYY/SIMBA/processed/SIMBA_interfaces_YYYY.nc
- data/YYYY/WeatherStation/WeatherVars.nc
- data/YYYY/HeatBudget/processed/HeatBudget_fluxes_YYYY.nc

Rain is kept as an independent forcing because the model needs the hourly rain
series. Pass a non-standard source with ``--rain-file``. The loader accepts a
CSV supported by ``IMS.load_Rway_station_data(..., 'rain')`` or a NetCDF
containing ``rain_mm`` or ``rain``.

Output
------
- data/YYYY/Model/processed/model_scenarios_YYYY.nc

Scenarios differ only in whether:
1. ocean heat flux F_w is included,
2. Kirillov snow-ice formation is included,
3. rain is included.

Any combination of those three switches can be run at the same time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from functions import IMS_toolbox as IMS
from heat_budget import HeatBudgetParameters

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ModelConfig:
    root: Path = Path(".")
    year: int = 2024

    # These can be left as None for automatic resolution from the processed
    # observations. Explicit per-year overrides are applied below when needed.
    start: str | None = None
    end: str | None = None
    initial_ice_thickness_m: float | None = None
    initial_snow_ice_thickness_m: float | None = None
    snow_ice_onset: str | None = None

    dt_seconds: int = 3600

    # Ocean temperature is sampled 4 cm below the detected ice bottom,
    # preserving the original two-node offset used for 2024.
    ocean_temperature_offset_m: float = 0.04

    # Optional explicit rain path. If None, standard candidate paths are tried.
    rain_file_override: Path | None = None

    @property
    def year_dir(self) -> Path:
        return self.root / "data" / str(self.year)

    @property
    def simba_file(self) -> Path:
        return (
            self.year_dir
            / "SIMBA"
            / "processed"
            / f"SIMBA_interfaces_{self.year}.nc"
        )

    @property
    def weather_file(self) -> Path:
        return self.year_dir / "WeatherStation" / "WeatherVars.nc"

    @property
    def heat_budget_fluxes_file(self) -> Path:
        return (
            self.year_dir
            / "HeatBudget"
            / "processed"
            / f"HeatBudget_fluxes_{self.year}.nc"
        )

    @property
    def output_file(self) -> Path:
        return (
            self.year_dir
            / "Model"
            / "processed"
            / f"model_scenarios_{self.year}.nc"
        )

    @property
    def rain_candidates(self) -> list[Path]:
        weather_dir = self.root / "data" / str(self.year) / "PostvilleWeather" 

        candidates = [
            weather_dir / f"postville_weather_2019-2026.csv",
        ]
        return candidates


# Explicit settings are used only where the historical analysis needs them.
# Other years are resolved automatically from their processed observations.
YEAR_OVERRIDES = {
    2024: {
        "start": "2024-02-03",
        "end": "2024-04-15",
        "initial_ice_thickness_m": 0.48,
        "initial_snow_ice_thickness_m": 0.0,
        "snow_ice_onset": "2024-03-01",
    },
}


@dataclass(frozen=True)
class ModelConstants:
    """Constants used by the model that are not defined in HeatBudgetParameters."""

    rho_w: float = 1027.0
    rho_fw: float = 1000.0
    c_fw: float = 4186.0

@dataclass(frozen=True)
class ModelScenario:
    """One process-toggle experiment."""

    name: str
    use_f_w: bool = True
    use_snow_ice: bool = True
    use_rain: bool = True


# Edit only this list to choose which model experiments to run.
# Any combination of the three switches is valid.
SCENARIOS = [
    ModelScenario(
        name="full",
        use_f_w=True,
        use_snow_ice=True,
        use_rain=True,
    ),
    ModelScenario(
        name="no_Fw",
        use_f_w=False,
        use_snow_ice=True,
        use_rain=True,
    ),
    ModelScenario(
        name="no_snow_ice",
        use_f_w=True,
        use_snow_ice=False,
        use_rain=True,
    ),
    # Reproduces the no-Fw / no-flooding comparison used in Figure 8.
    # With snow-ice flooding disabled but rain enabled, H_si contains only
    # the surface melt/rain contribution.
    ModelScenario(
        name="no_Fw_no_snow_ice",
        use_f_w=False,
        use_snow_ice=False,
        use_rain=True,
    ),
    ModelScenario(
        name="no_rain",
        use_f_w=True,
        use_snow_ice=True,
        use_rain=False,
    ),
    ModelScenario(
        name="no_rain_no_snow_ice",
        use_f_w=True,
        use_snow_ice=False,
        use_rain=False,
    ),
]


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required model input not found: {path}")
    return path


def _rename_native_time(da: xr.DataArray) -> xr.DataArray:
    """Normalize SIMBA native time dimensions to ``time``."""
    if "time_6h" in da.dims:
        return da.rename({"time_6h": "time"})
    if "time_daily" in da.dims:
        return da.rename({"time_daily": "time"})
    return da


def _interp_to_time(
    da: xr.DataArray,
    target_time: xr.DataArray,
) -> xr.DataArray:
    """Linearly interpolate a 1-D timeseries onto model timestamps."""
    da = da.sortby("time")

    out = da.interp(
        time=target_time,
        method="linear",
    )

    values = out.values.copy()
    valid = np.flatnonzero(np.isfinite(values))

    if len(valid) == 0:
        raise ValueError(f"No finite values available for {da.name!r}.")

    # Only fill extrapolation gaps at the edges. Internal gaps remain visible.
    values[: valid[0]] = values[valid[0]]
    values[valid[-1] + 1 :] = values[valid[-1]]

    return out.copy(data=values)


def _first_valid_time(da: xr.DataArray) -> np.datetime64:
    da = da.dropna("time")
    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"No valid timestamps found for {da.name!r}.")
    return da.time.values[0]


def _last_valid_time(da: xr.DataArray) -> np.datetime64:
    da = da.dropna("time")
    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"No valid timestamps found for {da.name!r}.")
    return da.time.values[-1]


def apply_year_overrides(cfg: ModelConfig) -> ModelConfig:
    """Apply explicit settings needed to reproduce a particular season."""
    overrides = YEAR_OVERRIDES.get(cfg.year, {})

    updates = {}
    for field, value in overrides.items():
        if getattr(cfg, field) is None:
            updates[field] = value

    return replace(cfg, **updates)


# =============================================================================
# RESOLVE YEAR-SPECIFIC SETTINGS FROM DATA
# =============================================================================


def resolve_model_config(cfg: ModelConfig) -> ModelConfig:
    """Resolve dates and initial conditions for the selected year.

    Explicit values in ``cfg`` or ``YEAR_OVERRIDES`` take priority. Missing
    values are inferred from the processed SIMBA and heat-budget products.
    This keeps the model code year-agnostic while preserving the historical
    2024 choices exactly.
    """

    cfg = apply_year_overrides(cfg)

    simba = xr.open_dataset(_require_file(cfg.simba_file))
    weather = xr.open_dataset(_require_file(cfg.weather_file))
    hb = xr.open_dataset(_require_file(cfg.heat_budget_fluxes_file))
    temperature = _rename_native_time(simba["temperature"])
    H_bottom = _rename_native_time(simba["H_bottom"])
    snow_air = _rename_native_time(simba["snow_air"])
    snow_ice = _rename_native_time(simba["snow_ice"])


    if "F_w" not in hb:
        raise KeyError(
            f"Expected 'F_w' in {cfg.heat_budget_fluxes_file}. "
            f"Found: {list(hb.data_vars)}"
        )

    if cfg.start is None:
        start_candidates = [
            _first_valid_time(temperature),
            _first_valid_time(H_bottom),
            _first_valid_time(snow_air),
            _first_valid_time(hb["F_w"]),
            weather.time.values[0],
        ]
        start = max(pd.Timestamp(t) for t in start_candidates)
        cfg = replace(cfg, start=start.strftime("%Y-%m-%d"))

    if cfg.end is None:
        end_candidates = [
            _last_valid_time(temperature),
            _last_valid_time(H_bottom),
            _last_valid_time(snow_air),
            _last_valid_time(hb["F_w"]),
            weather.time.values[-1],
        ]
        end = min(pd.Timestamp(t) for t in end_candidates)
        cfg = replace(cfg, end=end.strftime("%Y-%m-%d"))

    start_time = np.datetime64(cfg.start)

    if cfg.initial_ice_thickness_m is None:
        bottom0 = H_bottom.interp(time=start_time).item()
        if not np.isfinite(bottom0):
            bottom0 = H_bottom.sel(time=start_time, method="nearest").item()

        # H_bottom is negative below z=0. This reproduces the original 2024
        # interpretation of H_i when snow ice is initially zero.
        cfg = replace(
            cfg,
            initial_ice_thickness_m=float(abs(bottom0)),
        )

    if cfg.initial_snow_ice_thickness_m is None:
        snow_ice0 = snow_ice.interp(time=start_time).item()
        if not np.isfinite(snow_ice0):
            snow_ice0 = snow_ice.sel(time=start_time, method="nearest").item()

        cfg = replace(
            cfg,
            initial_snow_ice_thickness_m=float(max(snow_ice0, 0.0)),
        )

    if cfg.snow_ice_onset is None:
        positive = snow_ice.where(snow_ice > 0, drop=True)

        if positive.sizes.get("time", 0) > 0:
            onset_time = pd.Timestamp(positive.time.values[0])
            cfg = replace(
                cfg,
                snow_ice_onset=onset_time.strftime("%Y-%m-%d"),
            )
        elif any(s.use_snow_ice for s in SCENARIOS):
            raise ValueError(
                f"No positive observed snow-ice interface was found for {cfg.year}. "
                "Set ModelConfig.snow_ice_onset explicitly, or disable snow ice "
                "for the scenarios you want to run."
            )

    if pd.Timestamp(cfg.end) <= pd.Timestamp(cfg.start):
        raise ValueError(
            f"Resolved model period is invalid: {cfg.start} to {cfg.end}."
        )

    return cfg


# =============================================================================
# RAIN
# =============================================================================


def resolve_rain_file(cfg: ModelConfig) -> Path:
    if cfg.rain_file_override is not None:
        return _require_file(Path(cfg.rain_file_override))

    for candidate in cfg.rain_candidates:
        if candidate.exists():
            return candidate

    candidates = "\n".join(f"  {p}" for p in cfg.rain_candidates)
    raise FileNotFoundError(
        "No hourly rain source was found. Tried:\n"
        f"{candidates}\n"
        "Supply the correct file with --rain-file."
    )


def load_rain_hourly(
    cfg: ModelConfig,
    model_time: xr.DataArray,
) -> xr.DataArray:
    """Load hourly rainfall in mm h-1 on the model time grid."""

    # If every selected scenario has rain disabled, rain is not required.
    if not any(s.use_rain for s in SCENARIOS):
        return xr.zeros_like(model_time, dtype=float).rename("rain_mm")

    path = resolve_rain_file(cfg)

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, low_memory=False)
        rain = IMS.load_Rway_station_data(frame, "rain")
    elif path.suffix.lower() in {".nc", ".netcdf"}:
        ds = xr.open_dataset(path)
        if "rain_mm" in ds:
            rain = ds["rain_mm"]
        elif "rain" in ds:
            rain = ds["rain"]
        else:
            raise KeyError(
                f"Rain NetCDF must contain 'rain_mm' or 'rain': {path}"
            )
    else:
        raise ValueError(f"Unsupported rain file type: {path.suffix}")

    rain_hourly = (
        rain
        .resample(time="1h")
        .mean()
        .sel(time=slice(cfg.start, cfg.end))
        .fillna(0)
    )

    # Rain events are not linearly interpolated. Missing model hours are dry.
    return rain_hourly.reindex(
        time=model_time,
        fill_value=0,
    )


# =============================================================================
# LOAD AND PREPARE MODEL FORCING
# =============================================================================


def load_model_forcing(
    cfg: ModelConfig,
    hb_params: HeatBudgetParameters,
) -> xr.Dataset:
    """Load all observational forcing needed by the 1-D model."""

    simba = xr.open_dataset(_require_file(cfg.simba_file))

    temperature = _rename_native_time(simba["temperature"])
    H_bottom = _rename_native_time(simba["H_bottom"])
    snow_air = _rename_native_time(simba["snow_air"])

    weather = (
        xr.open_dataset(_require_file(cfg.weather_file))
        .resample(time="1h")
        .mean()
        .sel(time=slice(cfg.start, cfg.end))
    )

    if weather.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No weather data found between {cfg.start} and {cfg.end}."
        )

    model_time = weather.time

    T_air_c = weather["AirT_C_Avg"]
    RH = weather["RH"]
    pressure_mbar = weather["BP_mbar_Avg"]
    wind_2m = weather["WS_ms_Avg"]

    F_sw_net = weather["RsNet_Avg"]
    F_lw_net = weather["RlNet_Avg"]
    LW_in = weather["LWUpperCo_Avg"]
    LW_out = weather["LWLowerCo_Avg"]

    # Radiatively derived surface temperature, as in the original model.
    T_surface_c = (
        ((LW_out - (1.0 - hb_params.epsilon) * LW_in)
            / (hb_params.epsilon* hb_params.sigma))** 0.25- 273.15
    )

    wind_10m = wind_2m * (10.0 / 2.0) ** (1.0 / 7.0)

    # Under-ice ocean temperature from SIMBA, preserving the original approach.
    H_bottom_daily = (
        H_bottom
        .sel(time=slice(cfg.start, cfg.end))
        .resample(time="1D")
        .mean()
        .dropna("time")
    )

    temperature_daily = (
        temperature
        .sel(time=slice(cfg.start, cfg.end))
        .resample(time="1D")
        .mean()
    )

    common_time = np.intersect1d(
        temperature_daily.time.values,
        H_bottom_daily.time.values,
    )

    temperature_daily = temperature_daily.sel(time=common_time)
    H_bottom_daily = H_bottom_daily.sel(time=common_time)

    ocean_temperature_c = temperature_daily.sel(
        z=H_bottom_daily - cfg.ocean_temperature_offset_m,
        method="nearest",
    )

    ocean_temperature_c = _interp_to_time(
        ocean_temperature_c,
        model_time,
    )

    snow_air_hourly = _interp_to_time(
        snow_air.sel(time=slice(cfg.start, cfg.end)),
        model_time,
    )

    # Use the seasonal-mean ocean heat flux calculated by the observational
    # heat-budget analysis over the model period.
    hb = xr.open_dataset(
        _require_file(cfg.heat_budget_fluxes_file)
    )

    if "F_w" not in hb:
        raise KeyError(
            f"Expected 'F_w' in {cfg.heat_budget_fluxes_file}. "
            f"Found: {list(hb.data_vars)}"
        )

    F_w_value = float(
        hb["F_w"]
        .sel(time=slice(cfg.start, cfg.end))
        .mean(skipna=True)
    )

    if not np.isfinite(F_w_value):
        raise ValueError(
            f"No finite F_w values found in {cfg.heat_budget_fluxes_file} "
            f"between {cfg.start} and {cfg.end}."
        )

    print(
        f"Using F_w from heat budget: "
        f"{F_w_value:.3f} W m-2"
    )

    F_ocean = xr.full_like(
        model_time,
        F_w_value,
        dtype=float,
    )


    rain_mm_hourly = load_rain_hourly(
        cfg,
        model_time,
    )

    rain_m_hourly = rain_mm_hourly / 1000.0
    rain_rate_m_s = rain_m_hourly / cfg.dt_seconds

    forcing = xr.Dataset(
        coords={"time": model_time.values},
        data_vars={
            "T_air_c": ("time", T_air_c.values),
            "T_surface_obs_c": ("time", T_surface_c.values),
            "RH": ("time", RH.values),
            "pressure_mbar": ("time", pressure_mbar.values),
            "wind_10m": ("time", wind_10m.values),
            "F_sw_net": ("time", F_sw_net.values),
            "F_lw_net": ("time", F_lw_net.values),
            "T_bottom_c": ("time", ocean_temperature_c.values),
            "snow_air": ("time", snow_air_hourly.values),
            "F_ocean": ("time", F_ocean.values),
            "rain_m": ("time", rain_m_hourly.values),
            "rain_rate_m_s": ("time", rain_rate_m_s.values),
        },
    )

    forcing["T_air_c"].attrs["units"] = "degC"
    forcing["T_surface_obs_c"].attrs["units"] = "degC"
    forcing["T_bottom_c"].attrs["units"] = "degC"
    forcing["wind_10m"].attrs["units"] = "m s-1"
    forcing["F_sw_net"].attrs["units"] = "W m-2"
    forcing["F_lw_net"].attrs["units"] = "W m-2"
    forcing["F_ocean"].attrs["units"] = "W m-2"
    forcing["F_ocean"].attrs["description"] = "Scalar F_w calculated from the saved heat-budget product"
    forcing["snow_air"].attrs["units"] = "m"
    forcing["rain_m"].attrs["units"] = "m h-1"
    forcing["rain_rate_m_s"].attrs["units"] = "m s-1"

    return forcing


# =============================================================================
# MODEL
# =============================================================================


def run_model_scenario(
    forcing: xr.Dataset,
    cfg: ModelConfig,
    hb_params: HeatBudgetParameters,
    model_constants: ModelConstants,
    scenario: ModelScenario,
) -> xr.Dataset:
    """Run one Semtner/Kirillov model scenario."""

    nt = forcing.sizes["time"]
    dt = float(cfg.dt_seconds)
    time = forcing.time.values

    # Shared heat-budget parameters
    rho_i = hb_params.rho_i
    rho_s = hb_params.rho_s
    k_i = hb_params.k_i
    k_s = hb_params.k_s
    L_ice = hb_params.L_ice

    # Model-specific constants
    rho_w = model_constants.rho_w
    rho_fw = model_constants.rho_fw
    c_fw = model_constants.c_fw

    T_air_k = forcing["T_air_c"].values + 273.15
    T_surface_obs_k = forcing["T_surface_obs_c"].values + 273.15
    T_bottom_k = forcing["T_bottom_c"].values + 273.15

    F_sw = forcing["F_sw_net"].values
    F_lw = forcing["F_lw_net"].values
    U = forcing["wind_10m"].values
    RH = forcing["RH"].values
    P = forcing["pressure_mbar"].values
    F_ocean = forcing["F_ocean"].values
    rain_m = forcing["rain_m"].values
    rain_rate = forcing["rain_rate_m_s"].values

    H_s = forcing["snow_air"].values

    H_i = float(cfg.initial_ice_thickness_m)
    H_si = float(cfg.initial_snow_ice_thickness_m)
    h_i = H_i + H_si
    h_s = float(H_s[0] - H_si)

    H_i_out = np.zeros(nt)
    H_si_out = np.zeros(nt)
    H_total_out = np.zeros(nt)
    H_snow_out = np.zeros(nt)
    freeboard_out = np.zeros(nt)
    T_surface_out = np.zeros(nt)
    T_interface_out = np.zeros(nt)
    F_cond_ice_out = np.zeros(nt)
    dH_i_dt_out = np.zeros(nt)
    F_surface_out = np.zeros(nt)
    F_cond_surface_out = np.zeros(nt)
    rain_heat_out = np.zeros(nt)
    melt_out = np.zeros(nt)

    onset = np.datetime64(cfg.snow_ice_onset)

    for n in tqdm.tqdm(
        range(nt),
        desc=f"{cfg.year} {scenario.name}",
    ):
        denom = k_s * h_i + k_i * h_s

        if np.isclose(denom, 0.0):
            raise ZeroDivisionError(
                f"Conductive denominator reached zero at {time[n]} "
                f"for scenario {scenario.name!r}."
            )

        conductive_factor = k_s * k_i / denom

        freeboard = (
            h_i
            - (rho_i * h_i + rho_s * h_s) / rho_w
        )

        if (
            scenario.use_snow_ice
            and freeboard < 0
            and time[n] >= onset
        ):
            H_si += (
                h_i * (rho_i - rho_w)
                + h_s * rho_s
            ) / (
                rho_w - (1.0 / 3.0) * rho_i
            )

        T_surface = T_surface_obs_k[n]
        rain_heat = 0.0

        # Surface energy balance from the original model code. Shortwave,
        # longwave, sensible, latent, and melt are always enabled; rain is
        # controlled by the scenario toggle.
        F_surface = F_sw[n] + F_lw[n]
        F_surface += IMS.sensible_heat_flux(
            U[n],
            T_surface,
            T_air_k[n],
        )
        F_surface += IMS.latent_heat_flux(
            U[n],
            RH[n],
            P[n],
            T_surface,
            T_air_k[n],
        )

        if scenario.use_rain:
            T_rain = T_air_k[n]

            if T_surface >= 273.15:
                rain_heat = (
                    rho_fw
                    * c_fw
                    * rain_rate[n]
                    * (T_rain - 273.15)
                )
            else:
                rain_heat = (
                    rho_fw
                    * c_fw
                    * rain_rate[n]
                    * (T_rain - 273.15)
                    + rho_fw * L_ice * rain_rate[n]
                )

            F_surface += rain_heat

        F_cond_surface = conductive_factor * (
            T_bottom_k[n] - T_surface
        )
        F_surface -= F_cond_surface

        melt = 0.0

        # Melt calculations
        if T_surface > 273.15:
            T_surface = 273.15

            if H_s[n] > H_si:
                H_m = F_surface / (rho_fw * L_ice) * dt
                melt = min(H_m, H_s[n] - H_si)
                # H_si += melt

            if H_s[n] <= H_si:
                H_m = F_surface / (rho_i * L_ice) * dt
                H_si -= H_m
                melt = H_m

        # Rain-driven melt below the melting point
        if scenario.use_rain:
            if T_surface < 273.15 and rain_heat > 0:
                if H_s[n] > H_si:
                    H_m = rain_heat / (rho_s * L_ice) * dt
                    melt = min(H_m, H_s[n] - H_si)

                if H_s[n] <= H_si:
                    H_m = rain_heat / (rho_s * L_ice) * dt
                    H_si -= H_m
                    melt = H_m

            if T_surface < 273.15 and rain_heat < 0:
                melt = 0.0

            # Direct rainfall contribution to the modeled meltwater/SWE level.
            # H_si += rain_m[n]

        T_interface = (
            k_s * h_i * T_surface
            + k_i * h_s * T_bottom_k[n]
        ) / (
            k_s * h_i
            + k_i * h_s
        )

        F_cond_ice = (
            k_i
            * (T_interface - T_bottom_k[n])
            / h_i
        )

        F_bottom = -F_cond_ice

        if scenario.use_f_w:
            F_bottom -= F_ocean[n]

        dH_i_dt = F_bottom / (rho_i * L_ice)
        H_i += dH_i_dt * dt

        h_i = H_i + H_si
        h_s = H_s[n] - H_si

        H_i_out[n] = H_i
        H_si_out[n] = H_si
        H_total_out[n] = h_i
        H_snow_out[n] = h_s
        freeboard_out[n] = freeboard
        T_surface_out[n] = T_surface
        T_interface_out[n] = T_interface
        F_cond_ice_out[n] = F_cond_ice
        dH_i_dt_out[n] = dH_i_dt
        F_surface_out[n] = F_surface
        F_cond_surface_out[n] = F_cond_surface
        rain_heat_out[n] = rain_heat
        melt_out[n] = melt

    # Save outputs
    out = xr.Dataset(
        coords={"time": forcing.time.values},
        data_vars={
            "H_i": ("time", H_i_out),
            "H_si": ("time", H_si_out),
            "H_total": ("time", H_total_out),
            "H_snow": ("time", H_snow_out),
            "freeboard": ("time", freeboard_out),
            "T_surface": ("time", T_surface_out),
            "T_snow_ice": ("time", T_interface_out),
            "F_cond_ice": ("time", F_cond_ice_out),
            "dH_i_dt": ("time", dH_i_dt_out),
            "F_surface": ("time", F_surface_out),
            "F_cond_surface": ("time", F_cond_surface_out),
            "F_rain": ("time", rain_heat_out),
            "H_melt": ("time", melt_out),
        },
    )

    for name in [
        "H_i",
        "H_si",
        "H_total",
        "H_snow",
        "freeboard",
        "H_melt",
    ]:
        out[name].attrs["units"] = "m"

    out["T_surface"].attrs["units"] = "K"
    out["T_snow_ice"].attrs["units"] = "K"
    out["F_cond_ice"].attrs["units"] = "W m-2"
    out["F_surface"].attrs["units"] = "W m-2"
    out["F_cond_surface"].attrs["units"] = "W m-2"
    out["F_rain"].attrs["units"] = "W m-2"
    out["dH_i_dt"].attrs["units"] = "m s-1"

    return out


# =============================================================================
# SCENARIO LOOP
# =============================================================================


def run_scenarios(
    forcing: xr.Dataset,
    cfg: ModelConfig,
    hb_params: HeatBudgetParameters,
    model_constants: ModelConstants,
    scenarios: list[ModelScenario],
) -> xr.Dataset:
    """Run multiple process-toggle scenarios into one output Dataset."""

    results = xr.Dataset(
        coords={"time": forcing.time.values}
    )

    for scenario in scenarios:
        print(
            f"\nRunning {cfg.year} / {scenario.name}: "
            f"F_w={scenario.use_f_w}, "
            f"snow_ice={scenario.use_snow_ice}, "
            f"rain={scenario.use_rain}"
        )

        run = run_model_scenario(
            forcing=forcing,
            cfg=cfg,
            hb_params=hb_params,
            model_constants=model_constants,
            scenario=scenario,
        )

        common_attrs = {
            "scenario": scenario.name,
            "use_F_w": int(scenario.use_f_w),
            "use_snow_ice": int(scenario.use_snow_ice),
            "use_rain": int(scenario.use_rain),
        }

        results = IMS.add_to_dataset(
            results,
            f"H_i_{scenario.name}",
            run["H_i"].values,
            attrs={
                **common_attrs,
                "long_name": "Congelation ice thickness",
                "units": "m",
            },
        )

        results = IMS.add_to_dataset(
            results,
            f"H_si_{scenario.name}",
            run["H_si"].values,
            attrs={
                **common_attrs,
                "long_name": "Snow-ice thickness",
                "units": "m",
            },
        )

    results.attrs.update(
        {
            "model": (
                "Semtner (1976) two-layer thermodynamic model with optional "
                "Kirillov et al. (2015) snow-ice formation"
            ),
            "year": cfg.year,
            "start": cfg.start,
            "end": cfg.end,
            "dt_seconds": cfg.dt_seconds,
            "snow_ice_onset": cfg.snow_ice_onset,
            "initial_ice_thickness_m": cfg.initial_ice_thickness_m,
            "initial_snow_ice_thickness_m": cfg.initial_snow_ice_thickness_m,
            "parameter_source": "heat_budget.HeatBudgetParameters",
            "rho_i": hb_params.rho_i,
            "rho_s": hb_params.rho_s,
            "k_i": hb_params.k_i,
            "k_s": hb_params.k_s,
            "L_ice": hb_params.L_ice,
            "epsilon": hb_params.epsilon,
            "sigma": hb_params.sigma,
            "F_w_source": str(cfg.heat_budget_fluxes_file),
            "F_w_used_W_m2": float(forcing["F_ocean"].isel(time=0)),
        }
    )

    return results


# =============================================================================
# COMMAND LINE / RUN
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Semtner/Kirillov model scenarios for one IMS season."
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="IMS season year (e.g. 2024, 2025, 2026).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root containing data/.",
    )
    parser.add_argument(
        "--rain-file",
        type=Path,
        default=None,
        help="Optional explicit hourly rain CSV/NetCDF path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ModelConfig(
        root=args.root,
        year=args.year,
        rain_file_override=args.rain_file,
    )

    cfg = resolve_model_config(cfg)

    print(
        "\nResolved model configuration:\n"
        f"  year: {cfg.year}\n"
        f"  start: {cfg.start}\n"
        f"  end: {cfg.end}\n"
        f"  initial H_i: {cfg.initial_ice_thickness_m:.3f} m\n"
        f"  initial H_si: {cfg.initial_snow_ice_thickness_m:.3f} m\n"
        f"  snow-ice onset: {cfg.snow_ice_onset}"
    )

    hb_params = HeatBudgetParameters()
    model_constants = ModelConstants()

    forcing = load_model_forcing(
        cfg,
        hb_params,
    )

    results = run_scenarios(
        forcing=forcing,
        cfg=cfg,
        hb_params=hb_params,
        model_constants=model_constants,
        scenarios=SCENARIOS,
    )

    cfg.output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_netcdf(cfg.output_file)

    print(f"\nSaved model scenarios to:\n{cfg.output_file}")


if __name__ == "__main__":
    main()
