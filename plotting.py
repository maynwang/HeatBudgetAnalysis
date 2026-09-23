from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.dates as mdates

from heat_budget.config import HeatBudgetParameters

# ============================================================
# GENERAL PLOT SETTINGS
# ============================================================

# # 2024
# t1 = np.datetime64("2024-01-26")
# t2 = np.datetime64("2025-04-15")

# 2025
t1 = np.datetime64("2025-02-19")
t2 = np.datetime64("2025-04-10")

# Date to start the basal heat fluxes
# basal_start = "2024-02-02" 
basal_start = None


fontsize = 12

savefig = 'yes'

# ============================================================
# LOAD NEW HEAT-BUDGET OUTPUT
# ============================================================

processed_dir = Path("outputs/")

daily = xr.open_dataset(
    processed_dir / "HeatBudget_inputs_2025.nc"
)

fluxes = xr.open_dataset(
    processed_dir / "HeatBudget_fluxes_2025.nc"
)

params = HeatBudgetParameters()


# ============================================================
# Define variables
# ============================================================

# ---- Surface heat budget ----

F_sw_net = fluxes["F_sw"]

F_lw_net_clean = fluxes["F_lw"]

# The old plotting code also refers to F_lw_net separately
F_lw_net = fluxes["F_lw"]

F_sens = fluxes["F_sens"]

F_lat = fluxes["F_lat"]

F_c_surf = fluxes["F_cond_surface"]

F_net_surf = fluxes["F_net_surface"]


# ---- 2D conductive heat flux ----

F_c_2d_daily = fluxes["F_cond_ice_2d"]

# Used only for its time coordinate in the old plotting code
F_c_2d = F_c_2d_daily

F_c_snow = fluxes["F_cond_snow_2d"]


# ---- Interfaces ----

snow_ice = daily["snow_ice"]

snow_air = daily["snow_air"]

H_bottom = daily["H_bottom"]


# Reference layer
# Original definitions:
# H_bottom + 0.08 m to H_bottom + 0.20 m

ref_layer_lower = (
    daily["H_bottom"]
    + params.ref_layer_bottom
)

ref_layer_upper = (
    daily["H_bottom"]
    + params.ref_layer_top
)


# Surface layer
# Original definition = upper 0.10 m

surf_layer_lower = (
    daily["snow_air"]
    - params.surface_layer_depth
)

surf_layer_upper = daily["snow_air"]


# ---- Basal heat budget ----

F_l = fluxes["F_phase_change_basal"]

F_c = fluxes["F_cond_ref"]

F_w = fluxes["F_w"]


# ---- Residual analysis ----

F_snowmelt = fluxes["F_snow_melt"]

F_icemelt = fluxes["F_ice_melt"]

residual = fluxes["residual"]

residual_roll = fluxes["residual_roll"]

sig_F_surf_roll = fluxes["sig_F_surface_roll"]


# ---- Rain ----

rain_non0 = daily["rain_mm"].where(
    daily["rain_mm"] > 0,
    drop=True,
)


# ============================================================
# FIG. 6: HEAT FLUXES AT SURFACE AND BOTTOM
#
# From here onward this is essentially your original code.
# ============================================================

F_l_daily = F_l.resample(time="1D").mean()

F_w_daily = F_w.resample(time="1D").mean()


fig, axx = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(10,8),
    sharex=True,
    facecolor="w",
    layout="constrained",
)


t = F_net_surf.time


# ------------------------------------------------------------
# (a) SURFACE HEAT FLUXES
# ------------------------------------------------------------

an = 0

axx[an].text(
    0.01,
    0.96,
    "(a)",
    fontsize=fontsize+4,
    fontweight="bold",
    transform=axx[an].transAxes,
    verticalalignment="top",
)

axx[an].axhline(
    0,
    c="dimgrey",
    alpha=0.8,
)

axx[an].plot(
    t,
    F_sw_net,
    color="peru",
    linestyle="-",
    label=r"$F_\mathrm{netSWR}$",
)

axx[an].plot(
    t,
    F_lw_net_clean,
    c="firebrick",
    label=r"$F_\mathrm{netLWR}$",
)

axx[an].plot(
    t,
    F_sens,
    c="forestgreen",
    label=r"$F_\mathrm{SH}$",
)

axx[an].plot(
    t,
    F_c_surf,
    label=r"$F_\mathrm{c,s}$",
    c="blue",
    linestyle="-",
)

axx[an].plot(
    t,
    F_lat,
    label=r"$F_\mathrm{LH}$",
    c="violet",
    linestyle="-",
)

axx[an].plot(
    t,
    F_net_surf,
    label=r"$F_\mathrm{net,s}$",
    c="black",
    linestyle="-",
    linewidth=1.5,
)

axx[an].grid()

axx[an].set_ylim(
    [-100,100]
)

axx[an].set_ylabel(
    r"W m$^{-2}$",
    fontsize=fontsize,
)

l1 = axx[an].legend(
    bbox_to_anchor=(1.001,1.001),
    fontsize=fontsize,
)

l1.set_in_layout(False)

axx[an].tick_params(
    axis="y",
    labelsize=fontsize,
)



# ------------------------------------------------------------
# (b) 2D CONDUCTIVE HEAT FLUX
# ------------------------------------------------------------

an = 1

axx[an].text(
    0.01,
    0.96,
    "(b)",
    fontsize=fontsize+4,
    fontweight="bold",
    transform=axx[an].transAxes,
    verticalalignment="top",
)


im = axx[an].pcolormesh(
    t.values,
    F_c_2d_daily.z + 0.024,
    F_c_2d_daily.T,
    cmap="RdBu_r",
    vmin=-40,
    vmax=40,
)


axx[an].pcolormesh(
    t.values,
    F_c_snow.z,
    F_c_snow.T,
    cmap="RdBu_r",
    vmin=-40,
    vmax=40,
)


axx[an].plot(
    t,
    snow_ice,
    c="k",
    linewidth=1.5,
)


# Add reference layer

axx[an].plot(
    F_c_2d.time,
    ref_layer_lower,
    linestyle="--",
    c="grey",
)

axx[an].plot(
    F_c_2d.time,
    ref_layer_upper,
    linestyle="--",
    c="grey",
)


# Add surface layer

axx[an].plot(
    F_c_2d_daily.time,
    surf_layer_lower,
    linestyle="--",
    c="grey",
)

axx[an].plot(
    F_c_2d_daily.time,
    surf_layer_upper,
    linestyle="--",
    c="grey",
)


H_bottom_daily = (
    H_bottom
    .resample(time="1D")
    .mean()
)


axx[an].set_ylabel(
    r"$z$ (m)",
    fontdict={"fontsize":fontsize},
)

axx[an].set_ylim(
    [-1,1]
)


cbar = fig.colorbar(
    im,
    ax=axx[an],
    pad=0.01,
)

cbar.set_label(
    label=r"$F_\mathrm{c}$ (W m$^{-2})$",
    size=fontsize,
)

cbar.ax.tick_params(
    labelsize=fontsize,
)

axx[an].tick_params(
    axis="y",
    labelsize=fontsize,
)



# ------------------------------------------------------------
# (c) BOTTOM HEAT FLUXES
# ------------------------------------------------------------

an = 2


axx[an].text(
    0.01,
    0.96,
    "(c)",
    fontsize=fontsize+4,
    fontweight="bold",
    transform=axx[an].transAxes,
    verticalalignment="top",
)


F_l_daily_sub = F_l_daily.sel(
    time=slice(basal_start,None)
)

F_c_daily_sub = F_c.sel(
    time=slice(basal_start,None)
)

F_w_daily_sub = F_w_daily.sel(
    time=slice(basal_start,None)
)


t = F_l_daily_sub.time


axx[an].plot(
    t,
    F_l_daily_sub,
    label=r"$F_\mathrm{g}$",
    c="brown",
)

axx[an].plot(
    t,
    F_c_daily_sub,
    label=r"$F_\mathrm{c,b}$",
    c="blue",
)

axx[an].plot(
    t,
    F_w_daily_sub,
    label=r"$F_\mathrm{w}$",
    c="forestgreen",
)


# Line separating cut-off for acceptable basal estimate
#
# IMPORTANT:
# np.datetime64 is used here instead of the original string
# solely to avoid the matplotlib timezone error.
#
# It does NOT change the appearance of the plot.

# axx[an].axvline(
#     np.datetime64(basal_start),
#     c="grey",
# )


axx[an].set_ylabel(
    r"W m$^{-2}$",
    fontsize=fontsize,
)

axx[an].set_ylim(
    [-40,40]
)


l2 = axx[an].legend(
    bbox_to_anchor=(1.001,1.001),
    loc="upper left",
    fontsize=fontsize,
)

l2.set_in_layout(False)


# trigger a draw so constrained layout is executed once
fig.canvas.draw()

# include legend in bbox_inches='tight'
l1.set_in_layout(True)

# prevent layout from changing afterward
fig.set_layout_engine("none")


axx[an].grid()


axx[an].xaxis.set_major_formatter(
    mdates.DateFormatter("%m-%d")
)


axx[an].set_xlim(
    [t1,t2]
)


axx[an].tick_params(
    axis="y",
    labelsize=fontsize,
)

axx[an].tick_params(
    axis="x",
    labelsize=fontsize,
)


plt.show()


if savefig=='yes':
    fig.savefig(
        "figures/HeatFluxes_2025.png",
        dpi=300,
        bbox_inches="tight",
    )



# ============================================================
# FIG. 7: NET SURFACE HEAT FLUX COMPONENTS
# ============================================================

fig, axx = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(10,8),
    sharex=True,
    facecolor="w",
    layout="constrained",
)


t = F_net_surf.time



# ------------------------------------------------------------
# (a) COMPLETE SURFACE HEAT BALANCE AS STACKED PLOT
# ------------------------------------------------------------

an = 0


F_c_surf_pos = F_c_surf.where(
    F_c_surf > 0,
    0,
)

F_c_surf_neg = F_c_surf.where(
    F_c_surf < 0,
    0,
)

F_sens_pos = F_sens.where(
    F_sens > 0,
    0,
)

F_sens_neg = F_sens.where(
    F_sens < 0,
    0,
)

F_lat_pos = F_lat.where(
    F_lat > 0,
    0,
)

F_lat_neg = F_lat.where(
    F_lat < 0,
    0,
)


axx[an].text(
    0.01,
    0.96,
    "(a) ",
    fontsize=fontsize+4,
    fontweight="bold",
    transform=axx[an].transAxes,
    verticalalignment="top",
)


bar_width = np.timedelta64(
    22,
    "h",
)


axx[an].bar(
    t,
    F_lw_net_clean,
    width=bar_width,
    color="firebrick",
    label=r"$F_{\mathrm{netLWR}}$",
)


axx[an].bar(
    t,
    F_sw_net,
    width=bar_width,
    color="peru",
    label=r"$F_\mathrm{netSWR}$",
)


axx[an].bar(
    t,
    F_c_surf_neg,
    width=bar_width,
    bottom=F_sw_net.where(
        F_c_surf < 0,
        0,
    ),
    color="steelblue",
    label=r"$F_\mathrm{c,s}$",
)


axx[an].bar(
    t,
    F_c_surf_pos,
    width=bar_width,
    bottom=F_lw_net.where(
        F_c_surf > 0,
        0,
    ),
    color="steelblue",
)


axx[an].bar(
    t,
    F_sens_neg,
    width=bar_width,
    bottom=(
        F_c_surf_neg
        + F_sw_net.where(
            F_sens < 0,
            0,
        )
    ),
    color="forestgreen",
    label=r"$F_\mathrm{SH}$",
)


axx[an].bar(
    t,
    F_sens_pos,
    width=bar_width,
    bottom=(
        F_c_surf_pos
        + F_lw_net_clean.where(
            F_sens > 0,
            0,
        )
    ),
    color="forestgreen",
)


axx[an].bar(
    t,
    F_lat_neg,
    width=bar_width,
    bottom=(
        F_c_surf_neg
        + F_sw_net.where(
            F_lat < 0,
            0,
        )
        + F_sens_neg
    ),
    color="violet",
    label=r"$F_\mathrm{LH}$",
)


axx[an].bar(
    t,
    F_lat_pos,
    width=bar_width,
    bottom=(
        F_c_surf_pos
        + F_lw_net_clean.where(
            F_lat > 0,
            0,
        )
        + F_sens_pos
    ),
    color="violet",
)


axx[an].bar(
    t,
    F_snowmelt,
    width=bar_width,
    bottom=(
        F_c_surf_neg
        + F_sw_net.where(
            F_snowmelt < 0,
            0,
        )
        + F_sens_neg
        + F_lat_neg
    ),
    color="lawngreen",
    label=r"$F_{\mathrm{m,sn}}$",
)


axx[an].bar(
    t,
    F_icemelt,
    width=bar_width,
    bottom=(
        F_c_surf_neg
        + F_sw_net.where(
            F_icemelt < 0,
            0,
        )
        + F_sens_neg
        + F_lat_neg
        + F_snowmelt
    ),
    color="gold",
    label=r"$F_{\mathrm{m,ic}}$",
)


axx[an].plot(
    t,
    residual_roll,
    "k--",
    label="Residual \n (smoothed)",
)


axx[an].set_ylabel(
    r"W m$^{-2}$",
    fontsize=fontsize,
)


l2 = axx[an].legend(
    bbox_to_anchor=(1,1.001),
    loc="upper left",
    fontsize=fontsize,
)


axx[an].grid()

axx[an].tick_params(
    axis="y",
    labelsize=fontsize,
)



# ------------------------------------------------------------
# (b) NET SURFACE FLUX, MELT, RESIDUAL
# ------------------------------------------------------------

an = 1


axx[an].text(
    0.01,
    0.96,
    "(b) ",
    fontsize=fontsize+4,
    fontweight="bold",
    transform=axx[an].transAxes,
    verticalalignment="top",
)


axx[an].plot(
    t,
    F_net_surf,
    "k-",
    label=r"$F_{\mathrm{net,s}}$",
)


axx[an].plot(
    t,
    F_icemelt + F_snowmelt,
    c="red",
    label=r"$F_{\mathrm{melt}}$",
)


axx[an].plot(
    t,
    residual,
    c="darkgrey",
    linestyle="--",
    label="Residual \n (unsmoothed)",
)


axx[an].plot(
    t,
    residual_roll,
    "k--",
    label="Residual \n (smoothed)",
)


l2 = axx[an].legend(
    bbox_to_anchor=(1,1.001),
    loc="upper left",
    fontsize=fontsize,
)


axx[an].set_ylabel(
    r"W m$^{-2}$",
    fontsize=fontsize,
)


axx[an].grid()

axx[an].tick_params(
    axis="y",
    labelsize=fontsize,
)



# ------------------------------------------------------------
# (c) RESIDUAL AGAINST UNCERTAINTY
# ------------------------------------------------------------

an = 2


axx[an].text(
    0.01,
    0.96,
    "(c)",
    fontsize=fontsize+4,
    fontweight="bold",
    transform=axx[an].transAxes,
    verticalalignment="top",
)


axx[an].plot(
    t,
    residual_roll,
    "k--",
    label="Residual \n (smoothed)",
)


axx[an].fill_between(
    t.values,
    -sig_F_surf_roll.values,
    sig_F_surf_roll.values,
    alpha=0.3,
    facecolor="black",
    label="Uncertainty",
)


# Add rain

rain_days = (
    rain_non0["time"]
    .dt.floor("D")
    .values
)


rain_days = np.unique(
    rain_days
)


for day in rain_days[:-3]:

    axx[an].axvspan(
        day,
        day + np.timedelta64(1,"D"),
        color="lightblue",
        alpha=0.3,
        lw=0,
    )


axx[an].set_xlim(
    [t[0].values,t[-1].values]
)


axx[an].set_ylabel(
    r"W m$^{-2}$",
    fontsize=fontsize,
)


axx[an].tick_params(
    axis="both",
    labelsize=fontsize,
)


axx[an].xaxis.set_major_formatter(
    mdates.DateFormatter("%m-%d")
)


l3 = axx[an].legend(
    bbox_to_anchor=(1,1.001),
    loc="upper left",
    fontsize=fontsize,
)




# Optional save:
if savefig=="yes":
    fig.savefig(
        "figures/Residual_2025.png",
        dpi=300,
        bbox_inches="tight",
    )