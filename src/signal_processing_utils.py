"""Util functions for signal processing for evaluating the relationship between two signals.

This module contains utility functions for signal processing, focusing on the evaluation of the relationship between two signals.
It includes functions for calculation cross-correlation, peak detection, Pearson correlation, and plotting.
It also includes functions to process this data, save it into a CSV file, and visualize the results.

"""

import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

import constants

def rmse(y_true, y_pred):
    """Calculate the Root Mean Square Error (RMSE) between two signals."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def normalized_rmse(y_true, y_pred, norm_type="max"):
    avg_rmse = rmse(y_true, y_pred)
    if avg_rmse == 0 or np.all(y_true < 0.009): # Avoid division by zero or negligible values
        return 0.0
    
    # Normalize the RMSE based on the specified norm_type
    if norm_type == "range":
        norm = np.max(y_true) - np.min(y_true)
    elif norm_type == "max":
        norm = np.max(np.abs(y_true))
    elif norm_type == "mean":
        norm = np.mean(np.abs(y_true))
    elif norm_type == "std":
        norm = np.std(y_true)
    else:
        raise ValueError("Unsupported norm_type.")
    
    return avg_rmse / norm

def compute_cross_correlation_lag(gt_axis, pred_axis):
    # normalize the signals
    gt_axis_norm = gt_axis/ np.linalg.norm(gt_axis)
    pred_axis_norm = pred_axis / np.linalg.norm(pred_axis)
    # compute the cross-correlation
    corr = correlate(pred_axis_norm, gt_axis_norm, mode='full')
    lag = corr.argmax() - (len(gt_axis) - 1)
    corr_max = np.max(corr)
    return corr_max, lag

def energy(signal):
    """Calculate the energy of a signal."""
    return np.sum(signal ** 2)

def fwhm_width(signal, peak_idx):
    peak_val = signal[peak_idx]
    half_max = peak_val / 2.0

    # Find left index where signal drops below half_max
    left = peak_idx
    while left > 0 and signal[left] > half_max:
        left -= 1

    if left != peak_idx and signal[left] != signal[left + 1]:
        try:
            interp_left = interp1d([signal[left], signal[left + 1]], [left, left + 1], fill_value="extrapolate")
            left_exact = float(interp_left(half_max))
        except (ZeroDivisionError, ValueError, RuntimeWarning):
            left_exact = float(left)
    else:
        left_exact = float(left)

    # Find right index where signal drops below half_max
    right = peak_idx
    while right < len(signal) - 1 and signal[right] > half_max:
        right += 1

    if right != peak_idx and signal[right - 1] != signal[right]:
        try:
            interp_right = interp1d([signal[right - 1], signal[right]], [right - 1, right], fill_value="extrapolate")
            right_exact = float(interp_right(half_max))
        except (ZeroDivisionError, ValueError, RuntimeWarning):
            right_exact = float(right)
    else:
        right_exact = float(right)

    return right_exact - left_exact, left_exact, right_exact

def compute_fwhm_error(gt_axis, pred_axis):
    gt_peaks, _ = find_peaks(gt_axis, height=0)
    pred_peaks, _ = find_peaks(pred_axis, height=0)

    matched = []
    for g_idx in gt_peaks:
        nearest = min(pred_peaks, key=lambda p: abs(p - g_idx)) if len(pred_peaks) > 0 else None
        if nearest is not None and abs(nearest - g_idx) < 10:
            matched.append((g_idx, nearest))
    
    errors = []
    errors_norm = []
    for g_idx, p_idx in matched:
        gt_width, _, _ = fwhm_width(gt_axis, g_idx)
        pred_width, _, _ = fwhm_width(pred_axis, p_idx)
        errors.append(abs(gt_width - pred_width))
        errors_norm.append(abs(gt_width - pred_width) / gt_width if gt_width != 0 else 0)
    
    fwhm_error = np.mean(errors) if errors else None
    normalized_fwhm_error = np.mean(errors_norm) if errors_norm else None

    return fwhm_error, normalized_fwhm_error, matched


def plotly_peaks_and_fwhm(gt_axis, pred_axis, matched, axis_name="X", sample_rate=1.0):
    gt_peaks, _ = find_peaks(gt_axis, height=0)
    pred_peaks, _ = find_peaks(pred_axis, height=0)

    t = np.arange(len(gt_axis)) / sample_rate
    fig = go.Figure()

    # Plot signals
    fig.add_trace(go.Scatter(x=t, y=gt_axis, mode='lines', name='Ground Truth', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, y=pred_axis, mode='lines', name='Prediction', line=dict(color='orange')))

    # Plot peak markers
    fig.add_trace(go.Scatter(x=t[gt_peaks], y=gt_axis[gt_peaks], mode='markers', name='GT Peaks',
                             marker=dict(color='blue', symbol='circle', size=6)))
    fig.add_trace(go.Scatter(x=t[pred_peaks], y=pred_axis[pred_peaks], mode='markers', name='Pred Peaks',
                             marker=dict(color='orange', symbol='x', size=6)))

    # FWHM visualization
    for g_idx, p_idx in matched:
        _, gl, gr = fwhm_width(gt_axis, g_idx)
        _, pl, pr = fwhm_width(pred_axis, p_idx)

        fig.add_trace(go.Scatter(x=[gl / sample_rate, gr / sample_rate],
                                 y=[gt_axis[g_idx] / 2] * 2,
                                 mode='lines',
                                 name='GT FWHM',
                                 line=dict(dash='dot', color='blue')))

        fig.add_trace(go.Scatter(x=[pl / sample_rate, pr / sample_rate],
                                 y=[pred_axis[p_idx] / 2] * 2,
                                 mode='lines',
                                 name='Pred FWHM',
                                 line=dict(dash='dot', color='orange')))

    fig.update_layout(title=f"Axis {axis_name}: Signal Comparison with Peaks and FWHM",
                      xaxis_title="Time (s)",
                      yaxis_title="Force",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      height=500)
    fig.show()

def compare_highest_peaks(gt_signal, pred_signal):
    # Find the strongest (most extreme) peak in terms of absolute value
    gt_peak_val = np.max(np.abs(gt_signal))
    pred_peak_val = np.max(np.abs(pred_signal))

    # Compare predicted vs. ground truth peak
    diff = pred_peak_val - gt_peak_val

    # Normalized overshoot based on GT peak magnitude
    normalized_diff = diff / gt_peak_val if gt_peak_val != 0 else np.nan

    return diff, normalized_diff

def peak_wise_analysis(gt, pred, run=1):
    """Perform peak-wise analysis on the given ground truth and predicted signals.
    Args:
        gt (np.ndarray): Ground truth signal data.
        pred (np.ndarray): Predicted signal data.
        run (int): Run number for analysis.
    Returns:
        list: A list of dictionaries containing analysis results for each axis.
    """
    results_all = []
    peak_times = constants.INDIVIDUAL_PEAK_TIMES["force_policy"][run]
    assert len(peak_times) >= 1, f"{peak_times=}" 

    for start_idx, end_idx in peak_times:
        results = {}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            # Flatten the arrays to 1D for cross-correlation
            gt_axis = gt[start_idx:end_idx, i]
            pred_axis = pred[start_idx:end_idx, i]
            assert gt_axis.shape == pred_axis.shape, f"{gt_axis.shape=}, {pred_axis.shape=}"

            corr, lag = compute_cross_correlation_lag(gt_axis, pred_axis)
            fwhm_err, norm_fwhm_error, matched = compute_fwhm_error(gt_axis, pred_axis)
            pcc = pearsonr(gt_axis, pred_axis)[0]
            energy_diff = energy(pred_axis) - energy(gt_axis)
            energy_diff_norm = energy_diff / energy(gt_axis) if energy(gt_axis) != 0 else 0

            peak_height_diff, peak_height_diff_norm = compare_highest_peaks(gt_axis, pred_axis)

            results[axis] = { 
                'CrossCorr Lag': lag,
                'Corr': corr,
                'FWHM Width Error': fwhm_err,
                'Normalized FWHM Width Error': norm_fwhm_error,
                'Pearson Correlation': pcc,
                'Energy Difference': energy_diff,
                'Energy Difference Norm': energy_diff_norm,
                'Max Magnitude Difference': peak_height_diff,
                'Max Magnitude Difference Norm': peak_height_diff_norm
            }
        results_all.append(results)

        # plotly_peaks_and_fwhm(gt_axis, pred_axis, matched, axis_name=axis, sample_rate=sample_rate)

    return results_all

def run_wise_analysis(gt, pred, run=1):
    """Perform run-wise analysis on the given ground truth and predicted signals.
    Args:
        gt (np.ndarray): Ground truth signal data.
        pred (np.ndarray): Predicted signal data.
        run (int): Run number for analysis.
    Returns:
        dict: A dictionary containing analysis results for each axis.
    """
    results = {}

    # Ensure the data is in the correct shape
    assert gt.shape[1] == 3, f"Expected 3 columns for GT, got {gt.shape[1]}"
    assert pred.shape[1] == 3, f"Expected 3 columns for Pred, got {pred.shape[1]}"

    # Calculate the rmse and normalized rmse for the signal
    results['RMSE'] = rmse(gt, pred)
    results['NRMSE'] = normalized_rmse(gt, pred)

    # Calculate the Pearson correlation coefficient
    pcc = pearsonr(gt.flatten(), pred.flatten())[0]
    results['Pearson Correlation'] = pcc

    # calculate metrics for each axis
    for i, axis in enumerate(['X', 'Y', 'Z']):
        gt_axis = gt[:, i]
        pred_axis = pred[:, i]
        assert gt_axis.shape == pred_axis.shape, f"{gt_axis.shape=}, {pred_axis.shape=}"

        rmse_axis = rmse(gt_axis, pred_axis)
        nrmse_axis = normalized_rmse(gt_axis, pred_axis)
        pcc_axis = pearsonr(gt_axis, pred_axis)[0]

        results[f'RMSE_{axis}'] = rmse_axis
        results[f'NRMSE_{axis}'] = nrmse_axis
        results[f'Pearson Correlation_{axis}'] = pcc_axis
    
    return results

def save_results_to_csv(results, run_num=1, save_dir="quantified_results", peak_wise=True):
    # Flatten into a list of rows
    rows = []
    if peak_wise:
        for peak_num, peak_result in enumerate(results, start=1):
            for axis in ['X', 'Y', 'Z']:
                metric = peak_result[axis]
                row = {
                    'Run': run_num,
                    'Peak': peak_num,
                    'Axis': axis,
                    'CrossCorr Lag': metric['CrossCorr Lag'],
                    'Corr': metric['Corr'],
                    'FWHM Width Error': metric['FWHM Width Error'],
                    'Normalized FWHM Width Error': metric['Normalized FWHM Width Error'],
                    'Energy Difference': metric['Energy Difference'],
                    'Energy Difference Norm': metric['Energy Difference Norm'],
                    'Max Magnitude Difference': metric['Max Magnitude Difference'],
                    'Max Magnitude Difference Norm': metric['Max Magnitude Difference Norm']
                }
                rows.append(row)
        # Convert to DataFrame
        df = pd.DataFrame(rows)
    else:
        results["Run"] = run_num
        df = pd.DataFrame.from_dict([results])

    # Ensure the directory exists or create it
    os.makedirs(save_dir, exist_ok=True)
    if peak_wise:
        file_path = os.path.join(save_dir, f"quantified_results_peakwise.csv")
    else:
        file_path = os.path.join(save_dir, f"quantified_results_runwise.csv")
        
    # Save to CSV
    df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))

def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Computes the moving average for each column of data separately.

    Parameters:
    data (np.ndarray): A 2D array where each column represents a series of data points.
    window_size (int): The number of data points in each moving average window.

    Returns:
    np.ndarray: A 2D array with the same shape as data, containing the moving averages.
    """
    if window_size > data.shape[0]:
        raise ValueError(
            "window_size is larger than the number of rows in data.")

    zero_pad = np.zeros((window_size-1, data.shape[1]))
    data_padded = np.insert(data, 0, zero_pad, axis=0)
    cumsum_vec = np.cumsum(data_padded, axis=0)
    moving_avg = (cumsum_vec[window_size:] -
                  cumsum_vec[:-window_size]) / window_size

    assert moving_avg.shape[1] == data.shape[
        1], f"The output shape {moving_avg.shape} does not match the input shape {data.shape}"
    return moving_avg

def plot_forces_plotly(forces_pred: np.ndarray,
                        forces_smooth: np.ndarray,
                        forces_gt: np.ndarray,
                        avg_rmse: float,
                        avg_nrmse: float,
                        run: int,
                        save_dir: str,
                        crop_intervals=None,
                        loss_criterion: str = "mse") -> None:
    assert forces_pred.shape == forces_gt.shape
    assert forces_pred.shape[1] == 3

    time_axis = np.arange(forces_pred.shape[0])
    axes = ["X", "Y", "Z"]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[f"Force in {axis} Direction" for axis in axes])

    for i, axis_label in enumerate(axes):
        # Add smoothed prediction
        fig.add_trace(go.Scatter(
            x=time_axis[:-1],
            y=forces_smooth[:, i],
            mode='lines',
            name='Smoothed Prediction',
            hovertemplate='Time: %{x}<br>Force: %{y:.2f} N<extra></extra>'
        ), row=i+1, col=1)

        # Add ground truth
        fig.add_trace(go.Scatter(
            x=time_axis[:-1],
            y=forces_gt[:-1, i],
            mode='lines',
            name='Ground Truth',
            hovertemplate='Time: %{x}<br>Force: %{y:.2f} N<extra></extra>'
        ), row=i+1, col=1)

        # Add crop intervals
        if crop_intervals:
            for start, end in crop_intervals:
                end = end if end != -1 else forces_gt.shape[0]
                fig.add_vline(x=start, line=dict(color='red', dash='dash'), row=i+1, col=1)
                fig.add_vline(x=end, line=dict(color='green', dash='dash'), row=i+1, col=1)

        ax_rmse = rmse(forces_gt[:-1, i], forces_smooth[:, i])
        ax_nrmse = rmse(forces_gt[:-1, i], forces_smooth[:, i])
        fig.update_yaxes(title_text=f"{axis_label}-Force [N]<br>RMSE: {ax_rmse:.4f}, NRMSE: {ax_nrmse:.4f}", row=i+1, col=1)

    fig.update_layout(
        height=1000,
        width=900,
        title_text=f"Force Predictions vs Ground Truth (Run {run}) - Avg RMSE: {avg_rmse:.4f}, Avg NRMSE: {avg_nrmse:.4f}",
        showlegend=True
    )

    save_path = os.path.join(save_dir, f"pred_run_{run}_forces_{loss_criterion}_combined.html")
    fig.write_html(save_path)

def plot_forces_matplotlib(forces_pred: np.ndarray,
                forces_smooth: np.ndarray,
                forces_gt: np.ndarray,
                avg_rmse: float,
                avg_nrmse: float,
                run: int,
                save_dir: str = 'plots/transformer',
                crop_intervals = None,
                loss_criterion: str = "mse",
                pdf: bool = False) -> None:
    """
    Plots forces for x, y, z axes as subplots and saves the figures to the specified directory.

    Args:
        forces_pred: Predicted forces (Nx3 array).
        forces_smooth: Smoothed predicted forces (Nx3 array).
        forces_gt: Ground truth forces (Nx3 array).
        avg_rmse: Average RMSE value for the run.
        run: Run number.
        pdf: Whether to save the plots as PDFs (if False, saves as PNG).
        save_dir: Directory to save the plots (default is 'plots/').
        crop_intervals: List of tuples indicating the start and end of crop intervals (default is None).
        loss_criterion: Loss criterion used for training (default is 'mse').
    """
    assert forces_pred.shape == forces_gt.shape
    assert forces_pred.shape[1] == 3

    os.makedirs(save_dir, exist_ok=True)

    time_axis = np.arange(forces_pred.shape[0])
    axes = ["X", "Y", "Z"]

    # Create a figure with three subplots for raw predictions
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f"Force Predictions vs Ground Truth, Run {run}, Avg RMSE: {avg_rmse:.4f}, Avg NRMSE: {avg_nrmse:.4f}")

    for i, ax_label in enumerate(axes):
        axs[i].plot(time_axis, forces_pred[:, i], label='Predicted', linestyle='-', marker='')
        axs[i].plot(time_axis, forces_gt[:, i], label='Ground Truth', linestyle='-', marker='')
        ax_rmse = rmse(forces_gt[:, i], forces_pred[:, i])
        ax_nrmse = rmse(forces_gt[:-1, i], forces_pred[:, i])
        axs[i].set_title(f"Force in {ax_label} Direction, Avg RMSE: {ax_rmse:.4f}, NRMSE: {ax_nrmse:.4f}")
        axs[i].set_ylabel('Force [N]')
        axs[i].grid()
        #axs[i].set_ylim(-6, 2)
        axs[i].legend()

        # Add vertical lines for crop timestamps
        if crop_intervals:
            for j, (start, end) in enumerate(crop_intervals):
                end = end if end != -1 else forces_gt.shape[0]
                axs[i].axvline(x=start, color='r', linestyle='--', label="Crop Start" if j == 0 else "")
                axs[i].axvline(x=end, color='g', linestyle='--', label="Crop End" if j == 0 else "")

    axs[-1].set_xlabel('Time (Seconds)')
    
    save_path = os.path.join(save_dir, f"pred_run_{run}_forces.{'pdf' if pdf else 'png'}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the suptitle
    plt.savefig(save_path)
    plt.close()

    # Create a second figure with three subplots for smoothed predictions
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f"Smoothed Force Predictions vs Ground Truth, Run {run}, Avg RMSE: {avg_rmse:.4f}")

    for i, ax_label in enumerate(axes):
        axs[i].plot(time_axis[:-1], forces_smooth[:, i], label='Smoothed Predictions', linestyle='-', marker='')
        axs[i].plot(time_axis[:-1], forces_gt[:-1, i], label='Ground Truth', linestyle='-', marker='')
        ax_rmse = rmse(forces_gt[:, i], forces_pred[:, i])
        ax_nrmse = rmse(forces_gt[:-1, i], forces_pred[:, i])
        axs[i].set_title(f"Force in {ax_label} Direction, Avg RMSE: {ax_rmse:.4f}, NRMSE: {ax_nrmse:.4f}")
        axs[i].set_ylabel('Force [N]')
        axs[i].grid()
        # axs[i].set_ylim(-1, 1)
        axs[i].legend()

        # Add vertical lines for crop timestamps
        if crop_intervals:
            for j, (start, end) in enumerate(crop_intervals):
                end = end if end != -1 else forces_gt.shape[0]
                axs[i].axvline(x=start, color='r', linestyle='--', label="Crop Start" if j == 0 else "")
                axs[i].axvline(x=end, color='g', linestyle='--', label="Crop End" if j == 0 else "")

    axs[-1].set_xlabel('Time')

    save_path = os.path.join(save_dir, f"pred_smooth_run_{run}_forces.{'pdf' if pdf else 'png'}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()