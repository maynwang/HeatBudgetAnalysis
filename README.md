# Kaipokok Bay Sea Ice Heat Budget and Model Code

This repository contains the analysis code used to process observations, calculate the sea-ice heat budget, run the 1-D thermodynamic sea-ice model, and generate figures for paper "Thermodynamics of Sub-Arctic Landfast Ice: A Heat Budget from Kaipokok Bay, Nunatsiavut (Labrador)"

The main workflow is:

1. Process the SIMBA temperature-chain observations and identify the snow-air, snow-ice, and ice-ocean interfaces.
2. Process meteorological and oceanographic observations.
3. Run the observational heat-budget analysis.
4. Run the 1-D sea-ice model and sensitivity experiments.
5. Generate the manuscript figures from the processed outputs.

The heat-budget and model calculations share the relevant physical parameters defined in the `heat_budget` package, including the latent heat of sea ice calculated using the Ono (1968) formulation.

Processed files are generally stored under:

```text
data/YYYY/<data_source>/processed/
```

For example, heat-budget outputs for 2024 are saved to:

```text
data/2024/HeatBudget/processed/
```

and model outputs are saved to:

```text
data/2024/Model/processed/
```

The main scripts should be run from the repository root so that relative file paths resolve correctly.

This code was developed for the Kaipokok Bay Ice Monitoring Site analysis and reflects the processing choices and parameterizations used in the associated study.
