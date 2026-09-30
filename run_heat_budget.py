# Automatically reload scripts after saving edits
%load_ext autoreload
%autoreload 2

from pathlib import Path

from heat_budget import (
    HeatBudgetParameters,
    SeasonConfig,
    run_heat_budget,
    summarize_rain_residual,
)


season = SeasonConfig(
    year=2024,
    start="2024-01-26",
    end="2024-04-15",
    base_dir=Path("."),
    rain_file=Path(
        "data/2024/PostvilleWeather/postville_weather_2020-2024.csv"
    ),
    basal_plot_start="2024-02-02",
)

# season = SeasonConfig(
#     year=2025,
#     start="2025-02-19",
#     end="2025-04-22",
#     base_dir=Path("."),
#     rain_file=Path(
#         "data/2024/PostvilleWeather/postville_weather_2019-2026.csv"
#     ),
#     basal_plot_start=None,
# )

# season = SeasonConfig(
#     year=2026,
#     start="2026-02-06",
#     end="2026-04-21",
#     base_dir=Path("."),
#     rain_file=Path(
#         "data/2024/PostvilleWeather/postville_weather_2019-2026.csv"
#     ),
#     basal_plot_start=None,
# )


params = HeatBudgetParameters()

daily, fluxes = run_heat_budget(
    season,
    params,
    include_residual=True,
)


print("\nMean daily heat fluxes:")
for name in [
    "F_lw",
    "F_sw",
    "F_sens",
    "F_lat",
    "F_cond_surface",
    "F_net_surface",
    "F_w",
]:
    value = float(fluxes[name].mean(skipna=True))
    print(f"{name:22s}: {value:8.3f} W m-2")


if "rain_mm" in daily:
    rain_summary = summarize_rain_residual(
        daily,
        fluxes,
    )

    print("\nResidual / rain summary:")
    for key, value in rain_summary.items():
        print(f"{key}: {value}")


from pathlib import Path

output_dir = (
    Path("data")
    / str(season.year)
    / "HeatBudget"
    / "processed"
)

output_dir.mkdir(parents=True, exist_ok=True)

daily.to_netcdf(
    output_dir / f"HeatBudget_inputs_{season.year}.nc"
)

fluxes.to_netcdf(
    output_dir / f"HeatBudget_fluxes_{season.year}.nc"
)

print(f"\nSaved outputs to {output_dir}")