from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from inference.combined_plot.constants import INSET_PAD_HOURS_BEFORE, INSET_PAD_HOURS_AFTER, DST_THRESHOLD


def plot_inset(
    ax_dst_zoom: plt.Axes,
    dst_df: pd.DataFrame,
    ts_col_dst: str,
    dst_col: str,
    anchor_ts: pd.Timestamp,
    anchor_dst: float,
    forecast_times: list[pd.Timestamp],
    q10_list: list[float],
    q50_list: list[float],
    q90_list: list[float],
    hist_segments: list[dict],
    color: str,
    anchor_ts_naive: pd.Timestamp,
) -> None:
    if not forecast_times:
        return

    inset = inset_axes(
        ax_dst_zoom,
        width="35%",
        height="30%",
        loc="lower right",
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax_dst_zoom.transAxes,
        borderpad=0,
    )
    inset_start = forecast_times[0] - pd.Timedelta(hours=INSET_PAD_HOURS_BEFORE)
    inset_end = forecast_times[-1] + pd.Timedelta(hours=INSET_PAD_HOURS_AFTER)

    y_vals_inset: list[float] = []
    dst_inset = dst_df[(dst_df[ts_col_dst] >= inset_start) & (dst_df[ts_col_dst] <= inset_end)]
    inset.plot(dst_inset[ts_col_dst], dst_inset[dst_col], color="black", linewidth=1.0, label=None)
    if not dst_inset.empty:
        y_vals_inset.extend(dst_inset[dst_col].tolist())
    inset.axhline(0, color="dimgray", linewidth=0.8, linestyle="-", alpha=0.8)
    inset.axhline(DST_THRESHOLD, color="dimgray", linewidth=0.8, linestyle="--", alpha=0.8)

    # Future forecasts in inset
    ft_inset_mask = [inset_start <= pd.to_datetime(t) <= inset_end for t in forecast_times]
    if any(ft_inset_mask):
        ft_inset = [t for t, m in zip(forecast_times, ft_inset_mask) if m]
        q10_inset = [v for v, m in zip(q10_list, ft_inset_mask) if m]
        q50_inset = [v for v, m in zip(q50_list, ft_inset_mask) if m]
        q90_inset = [v for v, m in zip(q90_list, ft_inset_mask) if m]
        inset.fill_between(ft_inset, q10_inset, q90_inset, color=color, alpha=0.3, linewidth=0)
        inset.plot(ft_inset, q50_inset, color=color, linestyle="--", linewidth=1.0)
        if q10_inset and q90_inset:
            y_vals_inset.extend(q10_inset)
            y_vals_inset.extend(q90_inset)
    else:
        ft_inset = []
        q10_inset = []
        q50_inset = []
        q90_inset = []

    # Past forecasts inside inset
    last_past = None
    for seg in sorted(hist_segments, key=lambda s: pd.to_datetime(s["start"])):
        ts_seg = [t for t in seg["ts"] if inset_start <= pd.to_datetime(t) <= inset_end]
        if not ts_seg:
            continue
        idxs = [i for i, t in enumerate(seg["ts"]) if inset_start <= pd.to_datetime(t) <= inset_end]
        q10_seg = np.array(seg["q10"])[idxs]
        q90_seg = np.array(seg["q90"])[idxs]
        inset.fill_between(ts_seg, q10_seg, q90_seg, color=seg["color"], alpha=0.3, linewidth=0, label=None)
        y_vals_inset.extend(q10_seg.tolist())
        y_vals_inset.extend(q90_seg.tolist())
        q50_last = None
        if seg.get("q50") is not None:
            q50_vals_full = np.asarray(seg["q50"], dtype=float)
            if q50_vals_full.shape[0] >= len(seg["ts"]):
                q50_seg = q50_vals_full[idxs]
                q50_last = float(q50_seg.ravel()[-1])
        if ft_inset and pd.to_datetime(ts_seg[-1]) < pd.to_datetime(ft_inset[0]):
            last_past = (
                ts_seg[-1],
                float(q10_seg[-1]),
                q50_last,
                float(q90_seg[-1]),
            )

    # Bridge past -> future
    if ft_inset and q10_inset and last_past is not None and pd.to_datetime(last_past[0]) < pd.to_datetime(ft_inset[0]):
        lp_ts, lp_q10, lp_q50, lp_q90 = last_past
        ft_ts = pd.to_datetime(ft_inset[0])
        inset_bridge_start = pd.to_datetime(lp_ts)
        inset_bridge_end = ft_ts
        if inset_bridge_end <= inset_bridge_start:
            inset_bridge_start = inset_bridge_end - pd.Timedelta(hours=1)
        ts_bridge_inset = pd.to_datetime(
            np.linspace(inset_bridge_start.value, inset_bridge_end.value, 3)
        )
        q10_bridge_inset = np.linspace(lp_q10, q10_inset[0], len(ts_bridge_inset))
        q90_bridge_inset = np.linspace(lp_q90, q90_inset[0], len(ts_bridge_inset))
        inset.fill_between(
            ts_bridge_inset,
            q10_bridge_inset,
            q90_bridge_inset,
            color=color,
            alpha=0.2,
            linewidth=0,
            edgecolor="none",
        )
        y_vals_inset.extend(q10_bridge_inset.tolist())
        y_vals_inset.extend(q90_bridge_inset.tolist())
        start_q50 = lp_q50 if lp_q50 is not None else (lp_q10 + lp_q90) / 2
        end_q50 = q50_inset[0] if q50_inset else (q10_inset[0] + q90_inset[0]) / 2
        q50_bridge_inset = np.linspace(start_q50, end_q50, len(ts_bridge_inset))
        inset.plot(
            ts_bridge_inset,
            q50_bridge_inset,
            color=color,
            linewidth=1.0,
            linestyle="--",
        )

    # Anchor line to first forecast for context
    if ft_inset:
        inset.plot(
            [anchor_ts, pd.to_datetime(ft_inset[0])],
            [anchor_dst, q50_inset[0] if q50_inset else anchor_dst],
            color=color,
            linewidth=1.0,
            linestyle="--",
        )

    inset.set_xlim(inset_start, inset_end)
    if y_vals_inset:
        y_min = min(y_vals_inset) - 5
        y_max = max(y_vals_inset) + 5
        inset.set_ylim(y_min, y_max)
    inset.grid(True, linestyle="--", alpha=0.3)
    inset.tick_params(labelsize=8)

    def _hours_from_anchor(x, pos):
        dt = mdates.num2date(x).replace(tzinfo=None)
        return f"{int(round((dt - anchor_ts_naive).total_seconds() / 3600))}"

    inset.xaxis.set_major_formatter(FuncFormatter(_hours_from_anchor))
    ticks = inset.get_xticks()
    tick_labels = [_hours_from_anchor(t, None) for t in ticks]
    if tick_labels:
        tick_labels[-1] = ""  # keep last tick mark but blank its label
        inset.set_xticks(ticks)
        inset.set_xticklabels(tick_labels)
    inset.set_xlabel("Hourly offset", fontsize=8)
    inset.xaxis.tick_top()
    inset.xaxis.set_label_position("top")
    mark_inset(
        ax_dst_zoom,
        inset,
        loc1=2,
        loc2=4,
        fc="none",
        ec="dimgray",
        lw=1.0,
        alpha=0.7,
    )
