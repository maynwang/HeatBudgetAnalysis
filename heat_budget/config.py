from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SeasonConfig:
    """Files and dates that can change from one field season to another."""

    year: int
    start: str
    end: str
    base_dir: Path = Path(".")
    simba_dir: Optional[Path] = None

    # Optional season-specific files / dates
    rain_file: Optional[Path] = None
    basal_plot_start: Optional[str] = None

    # Instrument setup
    wind_measurement_height: float = 2.0

    # Weather-station variable names
    air_temperature_var: str = "AirT_C_Avg"
    net_longwave_var: str = "RlNet_Avg"
    net_shortwave_var: str = "RsNet_Avg"
    wind_speed_var: str = "WS_ms_Avg"
    relative_humidity_var: str = "Humidity"
    pressure_var: str = "BP_mbar_Avg"
    sw_in_var: str = "SWUpper_Avg"
    sw_out_var: str = "SWLower_Avg"
    lw_in_var: str = "LWUpperCo_Avg"
    lw_out_var: str = "LWLowerCo_Avg"

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        if self.simba_dir is not None:
            self.simba_dir = Path(self.simba_dir)
        if self.rain_file is not None:
            self.rain_file = Path(self.rain_file)

    @property
    def weather_file(self) -> Path:
        return (
            self.base_dir
            / str(self.year)
            / "WeatherStation"
            / "WeatherVars.nc"
        )

    @property
    def simba_path(self) -> Path:
        if self.simba_dir is None:
            return self.base_dir / str(self.year) / "SIMBA"
        if self.simba_dir.is_absolute():
            return self.simba_dir
        return self.base_dir / self.simba_dir

    @property
    def rain_path(self) -> Optional[Path]:
        if self.rain_file is None:
            return None
        if self.rain_file.is_absolute():
            return self.rain_file
        return self.base_dir / self.rain_file


@dataclass(frozen=True)
class HeatBudgetParameters:
    """Physical constants and analysis choices used by the heat budget."""

    # SIMBA / layer geometry
    dz: float = 0.02
    ref_layer_bottom: float = 0.08
    ref_layer_top: float = 0.20
    surface_layer_depth: float = 0.10
    snow_cover_threshold: float = 0.05

    # Ice / snow
    c_i: float = 2100.0
    rho_i: float = 910.0
    rho_s: float = 330.0
    k_i: float = 2.3
    k_s: float = 0.3

    ice_salinity: float = 5.0
    latent_heat_reference_temperature: float = -0.78

    # Radiation
    epsilon: float = 0.985
    sigma: float = 5.67e-8

    # Fraction of shortwave penetrating bare ice
    I0: float = 0.18

    # Residual / melt analysis
    wind_transport_threshold: float = 7.7
    snow_melt_fraction_during_transport: float = 0.12
    rho_snow_ice: float = 600.0
    residual_rolling_days: int = 5

    # Uncertainties
    sig_Ta: float = 0.3
    relative_sig_RH: float = 0.02
    relative_sig_wind: float = 0.05
    relative_sig_pressure: float = 0.04
    relative_sig_radiation: float = 0.10
    sig_emissivity: float = 0.005
    sig_k: float = 0.10
    sig_temperature_sensor: float = 0.0625
    n_surface_temperature_sensors: int = 10
    sig_dz: float = 0.0
    uncertainty_rolling_days: int = 5

    @property
    def L_ice(self) -> float:
        """Latent heat of sea ice from Ono (1968), J kg-1."""
        T = self.latent_heat_reference_temperature
        S = self.ice_salinity
        return 333394 - 2113 * T - 114.2 * S + 18040 * (S / T)
