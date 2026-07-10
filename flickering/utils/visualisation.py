import numpy as np
import scipy
from scipy.interpolate import interp2d, interp1d, RegularGridInterpolator
from scipy.signal import correlate, correlation_lags
import itertools
import cv2
from time import time
from flickering.utils.movie_reader import get_movie_reader

# from contour_analysis.contour_fitter import  # legacy, removed *
from flickering.analysis.fitter import (
    ContourFitter,
    theoretical_tau,
    theoretical_tau_yoon,
)
from glob import glob
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import logging
from threading import RLock
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
from flickering.utils.debug import *

# from autoimager.correlation_contour import  # legacy module removed *
import gc
from os.path import exists
import os

from flickering.tracking.contour_io import ContourIO


class _LazyCCT:
    @property
    def CorrelationContourTracker(self):
        from flickering.tracking.correlation_tracker import CorrelationContourTracker

        return CorrelationContourTracker


CCT = _LazyCCT()
from tqdm.auto import tqdm

logger = logging.getLogger("VIS")


def visualise_contours(contours, valid_indices, output_file):
    """
    Currently hard coded for 360 rays. Won't work with variable number of samples and assumes the xy points
    in increasing azimuth order.
    Produces a visualisation showing deviation from mean contour over time over azimuth
    meant as a quick check if there wasn't a catastrophic failure
    """
    filtered_contours = []
    if not len(valid_indices) == np.sum(valid_indices):
        for i, contour in enumerate(contours):
            if valid_indices[i] and len(contour) == 360:
                filtered_contours.append(contour)
            else:
                filtered_contours.append(np.zeros((360, 2)))
        contours = np.array(filtered_contours)
    else:
        contours = np.array(contours)

    centers = (contours).mean(axis=1, keepdims=True)
    contours[:, :, :] -= centers
    contours_rads = np.linalg.norm(contours, axis=2)
    mean_shape = contours_rads[valid_indices].mean(axis=0)
    result = contours_rads - mean_shape
    max_value = np.ceil(result[np.array(valid_indices)].max())
    result[np.array(valid_indices) == False] = np.repeat(max_value, 360)
    fig, ax = plt.subplots()
    try:
        axes_image = ax.imshow(result, interpolation="nearest", aspect="auto")
        fig.colorbar(axes_image)
        ax.set_xlabel("Azimuth / degrees")
        ax.set_ylabel("Frame number")
        ax.set_title(f"Deviation from average shape (invalid={max_value})")
        fig.savefig(output_file, format="png", dpi=900)
    finally:
        plt.close(fig)
    gc.collect()  # Trying to survive some memory leaks


def generate_contour_video_sequence(
    frames,
    contours,
    output_file="/tmp/frames.mp4",
    fps=30,
    extra_contours=[],
    scale=1.0,
    valid_indices=None,
):
    frame = next(frames)
    contour = contours[0]
    video_shape = np.array(frame.shape) * scale
    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(video_shape[1]), int(video_shape[0])),
    )
    i = 1  # TODO: check the offset
    invalid_color = [0, 0, 255]
    colors = [[255, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 255]]
    for frame in tqdm(frames):
        img = CCT.CorrelationContourTracker.normalise_image_values(frame)
        disp = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if contours[i] is not None:
            if valid_indices is not None and not valid_indices[i]:
                c = invalid_color
            else:
                c = colors[0]
            disp = draw_xy_contour(disp, contours[i], c, scale=scale)
        for colorid, ec in enumerate(extra_contours):
            # print(ec[i].shape)
            # print(ec[i])
            disp = draw_xy_contour(disp, np.array(ec[i]) * scale, colors[colorid + 1])
        # return disp
        video.write(disp)
        i += 1

    video.release()


def generate_contour_video(
    movie_file=None,
    contour_file=None,
    output_file="contour_output.mp4",
    normalise_each=True,
    speed_correction=1,  # skip frames to get correct speed instead of all frames
    fps=30,
):
    """
    Produces a video with the contour highlighted for more detailed inspection
    """
    if movie_file is None or contour_file is None:
        raise ValueError("Both movie_file and contour_file must be specified.")

    if contour_file is not None:
        cio = ContourIO()
        cio.load(contour_file)

    movie = get_movie_reader(movie_file)
    #    img_array = []
    video_shape = movie.get_frame(0).shape
    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_shape[1], video_shape[0]),
    )
    invalid_color = [0, 0, 255]
    valid_color = [255, 255, 0]
    min_norm = 0
    max_norm = 65500
    scale_norm = 1
    for i in tqdm(range(movie.n_frames), total=movie.n_frames):
        if i % speed_correction != 0:
            continue
        if normalise_each:
            img = CCT.CorrelationContourTracker.normalise_image_values(
                movie.get_frame(i)
            )
        else:
            if i == 0:
                min_norm = movie.get_frame(i).min()
                max_norm = movie.get_frame(i).max()
                scale_norm = max_norm - min_norm
                scale_norm = 1.5 * scale_norm
                min_norm = min_norm * 0.95
                max_norm = max_norm * 1.05
            img = movie.get_frame(i)

            img = (img - min_norm) / scale_norm
        disp = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if contour_file is not None:
            xy_contour = cio.contours[i]

            if cio.valid_indices is not None and not cio.valid_indices[i]:
                c = invalid_color
            else:
                c = valid_color
            if cio.mode == "R":
                xy_contour = CCT.CorrelationContourTracker.convert_contour_xy(
                    cio.contours[i], cio.centers[i]
                )

            draw_xy_contour(
                disp, xy_contour, img_value=c
            )  # TODO: remove center argument
        # img_array.append(disp)
        # plt.imshow(disp)
        # TODO: fix?
        video.write(disp)
    video.release()


# TODO: check these, they were copied from correlation_contour.py
def draw_contour(image, rads, center, img_value=(255, 0, 0), display=False):
    rads = scipy.ndimage.gaussian_filter1d(rads, 4, mode="wrap")  # TODO remove

    if len(image.shape) == 2:
        disp = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        disp = image
    if len(rads) > 0:
        xy_contour = CCT.CorrelationContourTracker.convert_contour_xy(rads, center)
        coordinates = xy_contour.swapaxes(0, 1).astype(int)
        coordinates[0] = np.clip(coordinates[0], 0, disp.shape[1] - 1)
        coordinates[1] = np.clip(coordinates[1], 0, disp.shape[0] - 1)

        disp[coordinates[1], coordinates[0]] = img_value  # TODO: check if rotated

    if display:
        debug_display("CONTOUR", disp)

    return disp
    # cv2.destroyAllWindows()


def draw_xy_contour(image, contour, img_value=[255, 0, 0], scale=1.0, display=False):
    if len(contour) > 0:
        i = 0
        if len(image.shape) == 2:
            disp = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            disp = image
        if scale != 1:
            disp = cv2.resize(
                disp, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            contour = scale * np.array(contour)  # TODO?
        for coordinates in contour:
            if np.any(np.isnan(coordinates)):
                continue
            pixel_x = int(coordinates[0])
            pixel_y = int(coordinates[1])
            if pixel_x >= disp.shape[0] or pixel_y >= disp.shape[1]:
                logger.warning("Pixel outside of image")
                continue
            disp[pixel_x, pixel_y] = np.array(img_value).astype(
                np.uint8
            )  # ones(3) * 255.0  # np.array(img_value)

        if display:
            debug_display("contour_detection", disp)
    else:
        return None
    # cv2.destroyAllWindows()
    return disp


def create_summary_plot(
    cf,
    fit_results,
    contours,
    valid_indices,
    preview_file,
    centers=None,
    movie_file=None,
    output_file=None,
    autocorrelation_modes_1=[5, 8, 12],
    autocorrelation_modes_2=[16, 20, 24],
    autocorrelation_xlim_1=50,
    autocorrelation_xlim_2=30,
    title=None,
):
    """
    Creates a summary plot with 6 panels in a 2x3 grid:
    Row 0: Preview, Autocorrelation 1, Spectrum
    Row 1: Contour, Autocorrelation 2, Viscosity
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.2, 1], wspace=0.1)

    try:
        if title is None:
            if output_file:
                title = os.path.basename(output_file)
            else:
                title = "Summary Plot"
        fig.suptitle(title, fontsize=16)

        # 1. Preview (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        img = None
        if preview_file and exists(preview_file):
            try:
                img = cv2.imread(preview_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax1.imshow(img)
                ax1.set_title("Preview")
                ax1.axis("off")
            except Exception as e:
                logger.error(f"Failed to load preview {preview_file}: {e}")
                ax1.text(0.5, 0.5, "Preview Load Error", ha="center")
        else:
            ax1.text(0.5, 0.5, "No Preview", ha="center")
            ax1.axis("off")

        # 2. Contour (Bottom Left)
        ax2 = fig.add_subplot(gs[1, 0])
        img_contour = None
        if movie_file and exists(movie_file):
            try:
                movie = get_movie_reader(movie_file)
                try:
                    frame = CCT.CorrelationContourTracker.normalise_image_values(
                        movie.get_frame(0)
                    )
                    if frame is not None:
                        if frame.dtype != np.uint8:
                            frame = cv2.normalize(
                                frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                            )
                        img_contour = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                finally:
                    movie.destroy()
            except Exception as e:
                logger.error(f"Failed to load movie frame from {movie_file}: {e}")

        if img_contour is None and img is not None:
            img_contour = img.copy()

        if img_contour is not None:
            if centers is not None:
                contour_xy = CCT.CorrelationContourTracker.convert_contour_xy(
                    contours[0], centers[0]
                )
            else:  # XY mode?
                contour_xy = contours[0]
            draw_xy_contour(img_contour, contour_xy, img_value=[255, 0, 0])
            ax2.imshow(img_contour)
            ax2.set_title("Contour")
            ax2.axis("off")
        else:
            ax2.text(0.5, 0.5, "No Preview/Movie", ha="center")
            ax2.axis("off")

        # Helper for autocorrelation plotting
        def plot_autocorrelation(ax, modes, title, xlim):
            ax.set_box_aspect(1)
            if (
                "autocorrelation_function" in fit_results
                and fit_results["autocorrelation_function"] is not None
            ):
                acf = np.array(fit_results["autocorrelation_function"])
                acf_fits = fit_results.get("autocorrelation_fits")

                all_y_values = []
                end_for_yval = int(xlim / cf.delay_between_frames_ms)
                for i, mode in enumerate(modes):
                    if mode < len(acf):
                        data = acf[mode]
                        x = np.arange(len(data)) * cf.delay_between_frames_ms

                        # Plot real part
                        ax.errorbar(
                            x,
                            data[:, 0],
                            yerr=data[:, 1],
                            fmt="o",
                            label=f"Mode {mode}",
                            alpha=0.5,
                            markersize=2,
                        )
                        all_y_values.extend(data[:end_for_yval, 0])

                        # Fit
                        if acf_fits and mode < len(acf_fits):
                            fit_params = acf_fits[mode]
                            if "real" in fit_params:
                                params = fit_params["real"]["params"]
                                x_fit_frames = np.linspace(
                                    0, xlim / cf.delay_between_frames_ms, 500
                                )
                                x_fit_ms = x_fit_frames * cf.delay_between_frames_ms

                                if len(params) == 3:  # Exponential: C_0, tau, B
                                    y_fit = (
                                        params[0] * np.exp(-x_fit_frames / params[1])
                                        + params[2]
                                    )
                                elif (
                                    len(params) == 4
                                ):  # Linear Exponential: alpha, C_th, slope_lin, D
                                    # alpha = 1/tau
                                    y_fit = (
                                        params[1] * np.exp(-params[0] * x_fit_frames)
                                        + params[2] * x_fit_frames
                                        + (1 - params[1] + params[3])
                                    )
                                else:
                                    y_fit = np.zeros_like(x_fit_frames)

                                ax.plot(
                                    x_fit_ms,
                                    y_fit,
                                    linestyle="--",
                                    color=ax.lines[-1].get_color(),
                                )
                                all_y_values.extend(y_fit[:end_for_yval])

                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Autocorrelation")
                ax.set_title(title)
                ax.legend()
                ax.set_xlim(0, xlim)

                # Smart Y-limits
                if all_y_values:
                    y_min, y_max = min(all_y_values), max(all_y_values)
                    data_range = y_max - y_min
                    fixed_range = 1.3  # 1.1 - (-0.2)

                    if data_range < fixed_range:
                        padding = data_range * 0.1
                        ax.set_ylim(y_min - padding, y_max + padding)
                    else:
                        ax.set_ylim(-0.2, 1.1)
                else:
                    ax.set_ylim(-0.2, 1.1)

            else:
                ax.text(0.5, 0.5, "No Autocorrelation Data", ha="center")

        # 3. Autocorrelation 1 (Top Middle)
        ax3 = fig.add_subplot(gs[0, 1])
        plot_autocorrelation(
            ax3,
            autocorrelation_modes_1,
            "Autocorrelations (Low Modes)",
            autocorrelation_xlim_1,
        )

        # 4. Autocorrelation 2 (Bottom Middle)
        ax4_mid = fig.add_subplot(gs[1, 1])
        plot_autocorrelation(
            ax4_mid,
            autocorrelation_modes_2,
            "Autocorrelations (High Modes)",
            autocorrelation_xlim_2,
        )

        # 5. Spectrum (Top Right)
        ax4 = fig.add_subplot(gs[0, 2])
        plt.sca(ax4)
        ax4.set_box_aspect(1)

        if "mps" in fit_results:
            mps = np.array(fit_results["mps"])
            mps_err = np.array(fit_results["mps_err"])
            taus = (
                np.array(fit_results["decay_times"]["value"])
                if fit_results.get("decay_times")
                and fit_results["decay_times"].get("value") is not None
                else None
            )
            radius_nm = fit_results["radius"]["value"]

            cf.plot_spectrum(
                mps,
                mps_err,
                fit_results,
                fit_results.get("alpha_beta"),
                taus,
                cf.delta(radius_nm),
                filename=None,
            )
            ax4.set_title("Power Spectrum")
        else:
            ax4.text(0.5, 0.5, "No Spectrum Data", ha="center")

        # 6. Decay Times (Bottom Right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_box_aspect(1)
        if (
            "viscosity_fit" in fit_results
            and fit_results.get("decay_times")
            and fit_results["decay_times"].get("value") is not None
        ):
            if "original_decay_times" in fit_results:
                taus = np.array(fit_results["original_decay_times"]["value"])
            else:
                taus = np.array(fit_results["decay_times"]["value"])

            modes = np.arange(len(taus))
            tau_values = taus[:, 0]
            tau_errors = taus[:, 1]

            fit_modes = None
            if "fit_modes" in fit_results["viscosity_fit"]:
                fit_modes = np.array(fit_results["viscosity_fit"]["fit_modes"])

            ax5.errorbar(
                modes,
                tau_values,
                yerr=tau_errors,
                fmt="o",
                alpha=0.5,
            )

            plot_max_mode = cf.max_mode
            if plot_max_mode == "auto":
                if (
                    fit_results
                    and "detected_max_mode" in fit_results
                    and fit_results["detected_max_mode"] is not None
                ):
                    plot_max_mode = fit_results["detected_max_mode"]
                else:
                    plot_max_mode = getattr(cf, "auto_max_mode_range", [5, 20])[1]

            extended_min = max(4, cf.min_mode - 2)
            extended_max = plot_max_mode + 5

            if "eta_in" in fit_results["viscosity_fit"]:
                eta_in = fit_results["viscosity_fit"]["eta_in"]["value"]
                eta_in_err = fit_results["viscosity_fit"]["eta_in"]["error"]
                eta_out = fit_results["viscosity_fit"]["eta_out"]["value"]

                if cf.fit_viscosity_lock_sigma_kappa:
                    sigma = fit_results["sigma"]["value"] * 1e-7
                    kappa = fit_results["kappa"]["value"] * 1e-21
                else:
                    if "sigma" in fit_results["viscosity_fit"]:
                        sigma = fit_results["viscosity_fit"]["sigma"]["value"] * 1e-7
                        kappa = fit_results["viscosity_fit"]["kappa"]["value"] * 1e-21
                    else:
                        sigma = fit_results["sigma"]["value"] * 1e-7
                        kappa = fit_results["kappa"]["value"] * 1e-21

                x_plot = np.linspace(extended_min, extended_max, 100)

                if cf.fit_viscosity_method == "rautu":
                    y_plot = theoretical_tau(
                        x_plot, eta_in, eta_out, radius_nm * 1e-9, kappa, sigma
                    )
                elif cf.fit_viscosity_method == "yoon":
                    y_plot = theoretical_tau_yoon(
                        x_plot, eta_in, eta_out, radius_nm * 1e-9, kappa, sigma
                    )
                else:
                    y_plot = np.zeros_like(x_plot)

                if fit_modes is not None:
                    ax5.axvspan(
                        min(fit_modes),
                        max(fit_modes),
                        color="green",
                        alpha=0.1,
                        label="Fit Range",
                    )

                    ax5.plot(
                        x_plot,
                        y_plot * 1000,
                        label="Fit (Extended)",
                        color="r",
                        linestyle="--",
                    )

                    if fit_modes is not None:
                        x_fit_range = np.linspace(min(fit_modes), max(fit_modes), 50)
                        if cf.fit_viscosity_method == "rautu":
                            y_fit_range = theoretical_tau(
                                x_fit_range,
                                eta_in,
                                eta_out,
                                radius_nm * 1e-9,
                                kappa,
                                sigma,
                            )
                        elif cf.fit_viscosity_method == "yoon":
                            y_fit_range = theoretical_tau_yoon(
                                x_fit_range,
                                eta_in,
                                eta_out,
                                radius_nm * 1e-9,
                                kappa,
                                sigma,
                            )
                        else:
                            y_fit_range = np.zeros_like(x_fit_range)

                        ax5.plot(
                            x_fit_range, y_fit_range * 1000, color="r", linewidth=2
                        )

            ax5.set_xlim(extended_min, extended_max)
            ax5.set_xlabel("Mode")
            ax5.set_ylabel("Decay Time (ms)")
            ax5.legend()
            ax5.set_yscale("log")
            ax5.set_xscale("log")
            if "eta_in" in fit_results["viscosity_fit"]:
                ax5.set_title(
                    rf"Viscosity Fit $\eta_{{in}}$"
                    rf"=({eta_in*1000:.2f} $\pm$ {eta_in_err*1000:.2f}) mPa.s)"
                )
            else:
                ax5.set_title("Viscosity Fit failed")
            ax5.set_ylim((0.5, 100))
        else:
            ax5.text(0.5, 0.5, "No Viscosity Fit", ha="center")

        fig.tight_layout()
        if output_file is not None:
            fig.savefig(output_file)
        else:
            plt.show()
    finally:
        plt.close(fig)


def process_and_plot(
    contour_file, cf, output_file=None, title=None, preview_file=None, movie_file=None
):
    """
    Loads a contour, processes it with the given ContourFitter, and saves a summary plot.
    """
    no_preview = False
    if isinstance(contour_file, str):
        cio = ContourIO(contour_file)
        # Try to find preview file
        # Assuming standard naming convention: contour_file has _contour.npz or similar
        # and preview has -preview.jpeg
        # Example: run1..._contour.npz -> run1...-preview.jpeg
        # But contour file might be ...-linear_contour.npz
        # Let's try to deduce it.
        base_name = contour_file.split("_contour")[0]
        # Sometimes it's just .npz
        if base_name == contour_file:
            base_name = contour_file.replace(".npz", "")

        if preview_file is None:
            # Try common patterns
            base = base_name.split("/")[-1].split(".")[0]
            path = "/".join(base_name.split("/")[:-1])
            candidates = [
                path + "/" + base + "-preview.jpeg",
                path + "/" + base + "_preview.jpeg",
            ]
            # path + "/" + base + "-preview.jpeg")

            preview_file = None
            for c in candidates:
                if exists(c):
                    preview_file = c
                    break
        if movie_file is None:  # Try to find movie file
            movie_file = contour_file.replace("_contour.npz", ".movie")
            if not exists(movie_file):
                movie_file = None
    elif isinstance(contour_file, ContourIO):
        cio = contour_file
    else:
        raise ValueError("contour_file must be a string or ContourIO object")

    # Process
    results = cf.process_contours(cio.contours, cio.valid_indices, plot=False)

    # Plot
    create_summary_plot(
        cf,
        results,
        [cio.contours[0]],
        cio.valid_indices,
        centers=cio.centers if cio.mode == "R" else None,
        preview_file=preview_file,
        movie_file=movie_file,
        output_file=output_file,
        title=title,
    )

    return results
