from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from inference.combined_plot.constants import DST_THRESHOLD, STORM_COLOR, CALM_COLOR


def plot_dst_zoom(
    ax: plt.Axes,
    dst_df: pd.DataFrame,
    ts_col_dst: str,
    dst_col: str,
    zoom_start: pd.Timestamp,
    hist_segments: list[dict],
    forecast_times: list[pd.Timestamp],
    q10_list: list[float],
    q50_list: list[float],
    q90_list: list[float],
    anchor_ts: pd.Timestamp,
    anchor_dst: float,
    regime: str,
) -> None:
    color = STORM_COLOR if regime == "storm" else CALM_COLOR
    dst_zoom = dst_df[dst_df[ts_col_dst] >= zoom_start]
    ax.plot(dst_zoom[ts_col_dst], dst_zoom[dst_col], label="DST (actual)", color="black", linewidth=1.2)
    ax.axhline(0, color="dimgray", linewidth=1.0, linestyle="-", alpha=0.8)
    ax.axhline(DST_THRESHOLD, color="dimgray", linewidth=1.0, linestyle="--", alpha=0.8)

    forecast_mask = [_ts >= zoom_start for _ts in forecast_times]
    if forecast_times and any(forecast_mask):
        ft_zoom = [t for t, m in zip(forecast_times, forecast_mask) if m]
        q10_zoom = [v for v, m in zip(q10_list, forecast_mask) if m]
        q50_zoom = [v for v, m in zip(q50_list, forecast_mask) if m]
        q90_zoom = [v for v, m in zip(q90_list, forecast_mask) if m]
        ax.fill_between(
            ft_zoom,
            q10_zoom,
            q90_zoom,
            color=color,
            alpha=0.3,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=None,
        )
        ax.plot(
            ft_zoom,
            q50_zoom,
            color=color,
            linestyle="--",
            label=f"DST forecast $q_{{0.5}}$ ({regime})",
        )
        if ft_zoom:
            ax.plot(
                [anchor_ts, ft_zoom[0]],
                [anchor_dst, q50_zoom[0]],
                color=color,
                linewidth=plt.rcParams["lines.linewidth"],
                linestyle="--",
                label=None,
            )

    label_used_zoom = {}
    for seg in sorted(hist_segments, key=lambda s: pd.to_datetime(s["start"])):
        if pd.to_datetime(seg["end"]) < zoom_start:
            continue
        ts_seg = [t for t in seg["ts"] if pd.to_datetime(t) >= zoom_start]
        if not ts_seg:
            continue
        idxs = [i for i, t in enumerate(seg["ts"]) if pd.to_datetime(t) >= zoom_start]
        q10_seg = np.array(seg["q10"])[idxs]
        q90_seg = np.array(seg["q90"])[idxs]
        key = seg["regime"]
        lbl = None if label_used_zoom.get(key) else f"Issued $q_{{0.1}}$–$q_{{0.9}}$ ({key})"
        ax.fill_between(
            ts_seg,
            q10_seg,
            q90_seg,
            color=seg["color"],
            alpha=0.3,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=lbl,
        )
        label_used_zoom[key] = True

    ax.set_title(f"DST Forecast (Last {int((anchor_ts - zoom_start).days)} Days)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
