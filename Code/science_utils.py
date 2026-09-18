"""Shared, result-preserving utilities for the ASKAP/TESS research scripts.

The file is deliberately split into labeled sections rather than a collection
of unrelated helpers.  Only computations reused by several scripts belong
here; science-specific workflow, configuration, and plotting remain in their
original modules.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import sys
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# .ds / HDF5 compatibility utilities（保持 DStools 读数、切片与重采样语义）
# ============================================================================

def rebin(old_length: int, new_length: int, axis: int) -> np.ndarray:
    """Return the DStools-compatible one-dimensional compression matrix."""
    if old_length < 1 or new_length < 1:
        raise ValueError("Rebin dimensions must be positive")
    if new_length > old_length:
        raise ValueError("New length cannot exceed old length")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    compressor = np.zeros((new_length, old_length), dtype=float)
    compression_ratio = new_length / old_length
    row = column = 0
    budget = 1.0
    overflow = 0.0
    while row < new_length and column < old_length:
        if overflow > 0.0:
            value = overflow
            overflow = 0.0
            budget -= value
            row_shift, column_shift = 0, 1
        elif budget < compression_ratio:
            value = budget
            overflow = compression_ratio - budget
            budget = 1.0
            row_shift, column_shift = 1, 0
        else:
            value = compression_ratio
            budget -= value
            row_shift, column_shift = 0, 1
        compressor[row, column] = value
        row += row_shift
        column += column_shift
    return compressor if axis == 0 else compressor.T


def rebin2d(array: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """Apply DStools-compatible weighted compression to a two-dimensional array."""
    if isinstance(array, np.ma.MaskedArray):
        array = array.copy()
        array[array.mask] = np.nan
        array = array.data
    array = np.array(array, copy=True)
    if new_shape == array.shape:
        array[array == 0.0 + 0.0j] = np.nan
        return array
    if new_shape[0] < 1 or new_shape[1] < 1:
        raise ValueError("New shape dimensions must be positive")
    if new_shape[0] > array.shape[0] or new_shape[1] > array.shape[1]:
        raise ValueError("New shape cannot exceed old shape")
    time_compressor = rebin(array.shape[0], new_shape[0], axis=0)
    frequency_compressor = rebin(array.shape[1], new_shape[1], axis=1)
    array[np.isnan(array)] = 0.0 + 0.0j
    rebinned = time_compressor @ array @ frequency_compressor
    rebinned[rebinned == 0.0 + 0.0j] = np.nan
    return rebinned


def slice_open_end(
    array: np.ndarray,
    axis1_min: int,
    axis1_max: int,
    axis2_min: Optional[int] = None,
    axis2_max: Optional[int] = None,
) -> np.ndarray:
    """Slice with DStools' convention that an upper bound of zero is open-ended."""
    axis1_stop = None if axis1_max == 0 else axis1_max
    if axis2_min is None and axis2_max is None:
        return array[axis1_min:axis1_stop]
    axis2_stop = None if axis2_max == 0 else axis2_max
    return array[axis1_min:axis1_stop, axis2_min:axis2_stop]


def as_text(value: object) -> str:
    """Convert an HDF5 byte attribute or ordinary value to text."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


# Backward-compatible aliases for scripts that imported the historical names.
dstools_rebin_matrix = rebin
dstools_rebin2d = rebin2d
attribute_text = as_text


# ============================================================================
# Phase-folding uncertainty and display binning（相位误差传播与固定展示分箱）
# ============================================================================

def propagated_phase_uncertainty(
    delta_time_days: float, period_days: float, period_error_days: float
) -> float:
    """Propagate period uncertainty into a phase uncertainty in cycles."""
    if period_days <= 0 or period_error_days < 0:
        raise ValueError("period_days must be positive and period_error_days non-negative")
    return abs(float(delta_time_days)) * float(period_error_days) / float(period_days) ** 2


def phase_bin_for_display(
    phase: np.ndarray,
    values: np.ndarray,
    nbins: int,
    errors: np.ndarray | None = None,
    min_count: int = 3,
    phase_max: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin phase-folded values across ``[0, phase_max)`` for display."""
    phase = np.asarray(phase, dtype=float)
    values = np.asarray(values, dtype=float)
    input_errors = None if errors is None else np.asarray(errors, dtype=float)
    if input_errors is not None and input_errors.shape != values.shape:
        raise ValueError("display-bin errors must have the same shape as values")
    if phase_max <= 0.0:
        raise ValueError("phase_max must be positive")
    finite = np.isfinite(phase) & np.isfinite(values)
    phase, values = phase[finite], values[finite]
    if input_errors is not None:
        input_errors = input_errors[finite]
    edges = np.linspace(0.0, phase_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(nbins, np.nan)
    binned_errors = np.full(nbins, np.nan)
    for index in range(nbins):
        in_bin = (phase >= edges[index]) & (phase < edges[index + 1])
        count = int(np.sum(in_bin))
        if count >= min_count:
            binned[index] = np.nanmean(values[in_bin])
            if count == 1 and input_errors is not None and np.isfinite(input_errors[in_bin][0]):
                binned_errors[index] = input_errors[in_bin][0]
            else:
                binned_errors[index] = np.nanstd(values[in_bin]) / np.sqrt(count)
    valid = np.isfinite(binned) & np.isfinite(binned_errors)
    return centers[valid], binned[valid], binned_errors[valid]


def list_fits_files(directory: str | os.PathLike[str]) -> set[str]:
    """Recursively list FITS files without changing the caller's file contract."""
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        return set()
    return set(glob.glob(os.path.join(directory, "**", "*.fits"), recursive=True))


def retry_with_exponential_backoff(search_fn, label: str, max_retries: int = 3):
    """Retry a network/search callable with the downloader's historical delays."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return search_fn()
        except Exception as error:
            last_error = error
            print(
                f"  [WARN] {label} 查询失败（第 {attempt}/{max_retries} 次）: "
                f"{type(error).__name__}: {error}"
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise last_error


# ============================================================================
# Server-side DStools command helpers（仅供服务器端管线调用）
# ============================================================================

def extract_sbid_and_beam(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract normalized SBID and beam identifiers from a product filename."""
    sb_match = re.search(r"SB(\d+)", filename, re.IGNORECASE)
    beam_match = re.search(r"beam(\d+)", filename, re.IGNORECASE)
    sbid = str(int(sb_match.group(1))) if sb_match else None
    beam = str(int(beam_match.group(1))) if beam_match else None
    return sbid, beam


def run_cmd(cmd_str: str, cwd: str) -> None:
    """Run a server reduction command with the historical conda library path."""
    conda_bin_dir = os.path.dirname(sys.executable)
    conda_lib_dir = os.path.join(os.path.dirname(conda_bin_dir), "lib")
    custom_env = os.environ.copy()
    custom_env["PATH"] = conda_bin_dir + os.pathsep + custom_env.get("PATH", "")
    custom_env["LD_LIBRARY_PATH"] = conda_lib_dir + os.pathsep + custom_env.get("LD_LIBRARY_PATH", "")
    cmd_str = f"env LD_LIBRARY_PATH={custom_env['LD_LIBRARY_PATH']} {cmd_str}"

    try:
        subprocess.run(
            cmd_str,
            shell=True,
            check=True,
            cwd=cwd,
            executable="/bin/bash",
            env=custom_env,
        )
    except subprocess.CalledProcessError as error:
        logger.error("命令执行失败: %s", cmd_str)
        raise error


# ============================================================================
# 通用绘图标签：白字黑描边，保证叠加在散点、曲线或动态谱上都可读
# ============================================================================

def add_panel_label(axis, text: str, *, fontsize: float = 11, x: float = 0.02,
                    y: float = 0.95) -> None:
    """在面板左上角添加简洁的白色描边标签，不改变数据轴或图例布局。"""
    import matplotlib.patheffects as path_effects

    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        ha="left",
        va="top",
        zorder=20,
        path_effects=[path_effects.withStroke(linewidth=2.4, foreground="black")],
    )
