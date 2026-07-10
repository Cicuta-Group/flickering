import numpy as np
import scipy
from scipy.interpolate import interp2d, interp1d, RegularGridInterpolator
from scipy.signal import correlate, correlation_lags, windows
import itertools
import cv2
import matplotlib.pyplot as plt
from time import time
from glob import glob
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import multiprocessing
import re
import gc
import json
import matplotlib.pyplot as plt
import logging
from threading import RLock
import importlib
from copy import deepcopy
from flickering.utils.visualisation import *
from flickering.analysis.fitter import ContourFitter as CF

# from autoimager.correlation_contour import  # legacy module removed *
from matplotlib.ticker import FormatStrFormatter
from flickering.tracking.correlation_tracker import CorrelationContourTracker as CCT
from flickering.tracking.contour_io import ContourIO
from tqdm.auto import tqdm
from time import time
from copy import deepcopy
from functools import partial
import pickle
from os.path import isfile
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
import warnings
from PIL import Image
from flickering.tracking.contour_io import ContourIO
from flickering.utils.standard_configs import *
from flickering.utils.postprocessing import *
import statsmodels.api as sm
import statsmodels.formula.api as smf

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp, t, norm
from IPython.display import display  # Added for Jupyter Notebook compatibility

warnings.filterwarnings("ignore")


def remap_labels(wells, well_labels):
    labels = []
    for well in wells:
        labels.append(well_labels[well])
    return labels


def get_repeat_from_global_id(global_id):
    """
    Extract repeat number (or cell ID for manual) from global_id.
    Standard: Well-CellId-Repeat -> Return Repeat
    Manual: Well-CellId-Repeat_Group -> Return Repeat
    """
    try:
        last_part = global_id.split("-")[-1]
        if "_" in last_part:
            return int(last_part.split("_")[0])
        return int(last_part)
    except Exception:
        return -1


def get_group_from_global_id(global_id):
    """
    Extract group ID from global_id suffix.
    Standard: Well-CellId-Repeat -> Return 0
    Manual: Well-CellId-Repeat_Group -> Return Group
    """
    try:
        last_part = global_id.split("-")[-1]
        if "_" in last_part:
            group_val = last_part.split("_", 1)[-1]
            try:
                return int(group_val)
            except ValueError:
                return group_val
        return 0  # Default group
    except Exception:
        return 0


def get_group_results(collected_data):  # TODO
    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)

    fns = ["mean", "std", "min", "max", "median", q25, q75, "count"]
    agg = {
        "sample": "first",
        "sigma_value": fns,
        "sigma_corrected": fns,
        "kappa_corrected": fns,
        "fit_status": "first",
        "kappa_value": fns,
        "radius_value": fns,
        "valid_p": fns,
        "time": fns,
        "r2": fns,
        "pfs_offset": fns,
        "cell_id": "count",
    }

    if "contour_variance" in collected_data.columns:
        agg["contour_variance"] = fns
        agg["contour_variance_butter"] = fns

    return collected_data.groupby("well").agg(agg).sort_values(("sample", "first"))


def plot_collected_data_time(
    collected_data, column, label, wells, err_column=None, labels=None, unit="unit"
):
    if labels is None:
        labels = wells
    plt.figure(dpi=200)
    start_time = np.min(collected_data["time"])
    fit_results = {}

    # Iterate over wells and splitting by group
    for l, well in zip(
        list(map(lambda x: x.replace(":", "\n"), labels)), wells
    ):  # TODO different formats
        well_data_full = collected_data[collected_data["well"] == well]

        # Determine groups present in this well
        if "group" in well_data_full.columns:
            groups = sorted(
                well_data_full["group"].unique(),
                key=lambda x: (0, float(x)) if isinstance(x, (int, float, np.number)) else (1, str(x))
            )
        else:
            groups = [0]

        for g in groups:
            # Filter by group if we have multiple groups or specific group logic
            if "group" in well_data_full.columns:
                well_data = well_data_full[well_data_full["group"] == g]
            else:
                well_data = well_data_full

            if np.sum(well_data["sigma_value"] > 0) < 3:
                logging.warning(
                    f"Not enough data for {well} Group {g} to plot {column}"
                )
                continue

            well_data = well_data[
                ~np.isnan(well_data["sigma_value"])
            ]  # remove failed fits and rejected by filters

            try:
                coeffs, cov = np.polyfit(
                    well_data["time"] - start_time,
                    well_data[column],
                    1,
                    full=False,
                    cov=True,
                )
            except Exception:
                continue

            fit_key = f"{well}" if len(groups) <= 1 else f"{well}_G{g}"
            fit_results[fit_key] = (coeffs, cov)

            # Label suffix
            l_suffix = "" if len(groups) <= 1 else f" G{g}"

            plt.errorbar(
                (well_data["time"] - start_time).to_numpy(),
                well_data[column].to_numpy(),
                yerr=None if err_column is None else well_data[err_column],
                fmt="o",
                capsize=3,
                label=f"{l}{l_suffix}" if len(groups) > 1 else None,
            )
            plt.plot(
                (well_data["time"] - start_time).to_numpy(),
                (coeffs[0] * (well_data["time"] - start_time) + coeffs[1]).to_numpy(),
                label=f"{l}{l_suffix}: ({coeffs[0]*3600:.3f} +- {np.sqrt(cov[0,0])*3600:.3f}) {unit}/h",
            )
    plt.legend()
    plt.xlabel("Time/s")
    plt.ylabel(label)
    try:
        plt.ylim(
            (0, np.max(collected_data[collected_data[column] < 100000][column]) * 1.1)
        )
    except:
        pass
    return fit_results


def plot_collected_data_comparison(
    collected_data, column, label, wells, err_column=None, labels=None
):
    well_data_for_plot = []
    final_labels = []

    if labels is None:
        labels = wells

    for well, l in zip(wells, labels):  # TODO different formats
        well_data_full = collected_data[collected_data["well"] == well]

        # Determine groups
        if "group" in well_data_full.columns:
            groups = sorted(
                well_data_full["group"].unique(),
                key=lambda x: (0, float(x)) if isinstance(x, (int, float, np.number)) else (1, str(x))
            )
        else:
            groups = [0]

        for g in groups:
            if "group" in well_data_full.columns:
                well_data = well_data_full[well_data_full["group"] == g]
            else:
                well_data = well_data_full

            well_data = well_data[~np.isnan(well_data[column])]
            # well_data = well_data[~np.isnan(well_data["sigma_value"])] # keep inconsistent logic consistent? original had this
            if "sigma_value" in well_data.columns:
                well_data = well_data[~np.isnan(well_data["sigma_value"])]

            well_data_for_plot.append(well_data[column].values)

            l_str = l.replace(":", "\n")
            if len(groups) > 1:
                final_labels.append(f"{l_str}\nG{g}")
            else:
                final_labels.append(l_str)

    plt.figure(dpi=200)
    plt.boxplot(well_data_for_plot)
    plt.xticks(
        np.arange(len(final_labels)) + 1,
        final_labels,
        rotation=45,
    )
    plt.ylabel(label)
    plt.grid()

    # plt.errorbar(well_data["time"]-start_time, well_data[column], yerr=None if err_column is None else well_data[err_column], label=well, fmt="o", capsize=3)


def process_entry(
    cell_log_row, metadata, cf=None, contour_preprocessor=None, plot_folder=None
):
    if cf is None:
        cf = default_cf(metadata)

    return process_using_cf(
        cf,
        cell_log_row,
        metadata["folder"],
        metadata["valid_contours_threshold"],
        metadata["r2_threshold"],
        contour_preprocessor=contour_preprocessor,
        plot_folder=plot_folder,
    )


def process_using_cf(
    cf: ContourFitter,
    cell_log_row,
    folder,
    valid_contours_threshold,
    r2_threshold,
    contour_preprocessor=None,
    plot_folder=None,
):
    # if info["id"] not in process_cell_ids:
    #    continue
    # if not info["result"]:
    #    continue

    if callable(cf):
        cf = cf(cell_log_row)

    k = cell_log_row["global_id"]
    if "contour_file" in cell_log_row and cell_log_row["contour_file"] is None:
        # preprocessing failed to find contour
        return cell_log_row

    if "preview_file" not in cell_log_row:
        preview_f = glob(f"{folder}{k}-preview.jpeg")
        if len(preview_f) == 0:
            # this is normal for failed autofocus
            # logging.warning(f"No preview found for {k}")
            preview_f = None
        else:
            preview_f = preview_f[0]

        cell_log_row["preview_file"] = preview_f
    else:
        preview_f = cell_log_row["preview_file"]

    if "movie_f" not in cell_log_row:
        movie_f = glob(f"{folder}{k}.*.movie")
        if len(movie_f) == 0:
            # this is normal for failed autofocus
            # logging.warning(f"No preview found for {k}")
            movie_f = None
        else:
            movie_f = movie_f[0]
        cell_log_row["movie_f"] = movie_f
    else:
        movie_f = cell_log_row["movie_f"]

    cell_log_row["repeat"] = get_repeat_from_global_id(cell_log_row["global_id"])
    cell_log_row["group"] = get_group_from_global_id(cell_log_row["global_id"])
    cell_log_row["fit_start"] = cf.min_mode
    cell_log_row["fit_end"] = cf.max_mode

    if "contour_file" not in cell_log_row:
        contour_f = glob(f"{folder}{k}.*_contour.npz")
        if len(contour_f) == 0:
            # this is normal for failed autofocus
            # logging.warning(f"No contour found for {k}")
            return cell_log_row

        if len(contour_f) == 1:
            cell_log_row["contour_file"] = contour_f[0]
            contour_f = contour_f[0]
    else:
        contour_f = cell_log_row["contour_file"]

    def process_contour_f(contour_f, cell_log_row):
        try:
            cio = ContourIO(contour_f)
            contour_id = contour_f.split("/")[-1].split(".")[0]
            if contour_preprocessor is not None:
                cio = contour_preprocessor(cio, cell_log_row)
            valid_p = np.sum(cio.valid_indices) / len(cio.valid_indices)
            cell_log_row["valid_contour_rate"] = valid_p
            cell_log_row["frames"] = len(cio.valid_indices)

        except Exception as e:
            logging.error("Error reading contour", exc_info=e)
            cell_log_row["fit_status"] = "error"
            cell_log_row["message"] = "corrupted contour"
            return cell_log_row
        # data_line = [info["id"], info["repeat"], info['time'], info['pfs_offset'], info['radius'], valid_p]
        # no_fit_data_line = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        if valid_p >= valid_contours_threshold:
            try:
                results = cf.process_contours(
                    cio.contours, cio.valid_indices, plot=False
                )
                if plot_folder is not None:
                    create_summary_plot(
                        cf,
                        results,
                        cio.contours,
                        cio.valid_indices,
                        centers=cio.centers if cio.mode == "R" else None,
                        preview_file=cell_log_row.get("preview_file"),
                        movie_file=cell_log_row.get("movie_f"),
                        output_file=f"{plot_folder}/{contour_id}-fit.png",
                    )

                if "autocorrelation_function" in results and results["autocorrelation_function"] is not None:
                    results["autocorrelation_function"] = [
                        mode_acf[:cf.save_acf_frames] for mode_acf in results["autocorrelation_function"]
                    ]

                # plt.figure(dpi=200)
                cell_log_row["fit_results"] = results
                # fit_results = [results["sigma"]["value"], results["sigma"]["error"], results["kappa"]["value"], results["kappa"]["error"], results["radius"]["value"], results["r2"]]
                if (
                    results["r2"] < r2_threshold
                ):  # TODO: find a better goodness of fit indicator
                    logging.debug(f"{k}: low r2 {results['r2']:.2f}")
                    cell_log_row["fit_status"] = "r2-fail"
                    # data_line += no_fit_data_line
                else:
                    cell_log_row["fit_status"] = "success"
                    # data_line += fit_results
            except Exception as e:
                logging.error(f"Failed to process {k}", exc_info=e)
                # data_line += no_fit_data_line
                cell_log_row["fit_status"] = "error"
        else:
            logging.debug(f"{k}: low valid % {valid_p:.2f}")
            cell_log_row["fit_status"] = "contour-invalid"

            # data_line += [contour_f, preview_f, movie_f]
            # data.append(data_line)
        return cell_log_row

    if isinstance(cell_log_row["contour_file"], list):
        logging.debug("Multi-contour processing")
        cell_log_row["type"] = "multientry"
        cell_log_row["results"] = []
        for cfname in cell_log_row["contour_file"]:
            fname = cfname.split("/")[-1]
            idents = fname.split(".")[0].split("-")[
                3:
            ]  # we skip the first few identifiers
            entry_res = process_contour_f(cfname, {})
            entry_res["filename"] = fname
            entry_res["idents"] = idents
            cell_log_row["results"].append(entry_res)
    else:
        cell_log_row = process_contour_f(cell_log_row["contour_file"], cell_log_row)

    gc.collect()
    return cell_log_row


def plot_time_gradients(time_gradients, wells=None, labels=None):
    if wells is None:
        wells = time_gradients["well"].to_numpy()
    if labels is None:
        labels = wells

    tg = time_gradients.set_index("well")

    # Filter for wells that exist in the dataframe
    existing_wells_mask = [well in tg.index for well in wells]
    wells = [well for i, well in enumerate(wells) if existing_wells_mask[i]]
    labels = [label for i, label in enumerate(labels) if existing_wells_mask[i]]

    if not wells:
        logging.warning("No wells with data to plot in plot_time_gradients.")
        return

    # Ensure we have a unique set of wells for plotting to avoid size mismatch
    unique_wells = []
    unique_labels = []
    seen_wells = set()
    for well, label in zip(wells, labels):
        if well not in seen_wells:
            unique_wells.append(well)
            unique_labels.append(label)
            seen_wells.add(well)

    # Group by index and take the first entry to handle duplicate indices in tg
    tg = tg.groupby(tg.index).first()
    tg = tg.loc[unique_wells]

    plt.figure(dpi=200)
    plt.errorbar(
        np.arange(len(unique_wells)),
        3600 * tg["sigma_slope"].to_numpy(),
        yerr=3600 * tg["sigma_slope_err"].to_numpy(),
        fmt="o",
        capsize=3,
    )
    plt.xticks(
        np.arange(len(unique_wells)),
        list(map(lambda x: x.replace(":", "\n"), unique_labels)),
        rotation=45,
    )
    plt.ylabel("Tension gradient 1e-7 N/m/h")
    plt.grid()

    plt.figure(dpi=200)
    plt.errorbar(
        np.arange(len(unique_wells)),
        3600 * tg["kappa_slope"].to_numpy(),
        yerr=3600 * tg["kappa_slope_err"].to_numpy(),
        fmt="o",
        capsize=3,
    )
    plt.xticks(
        np.arange(len(unique_wells)),
        list(map(lambda x: x.replace(":", "\n"), unique_labels)),
        rotation=45,
    )
    plt.ylabel("Bending gradient 1e-21 J/h")
    plt.grid()


def plot_cell_numbers(grouped_data, wells, labels, collected_data, r2_threshold):
    plt.figure(dpi=200)
    total_cells = collected_data.groupby("well")["cell_id"].nunique()

    # Calculate the number of cells with r2 < threshold
    rejected_cells = (
        collected_data[collected_data["r2"] < r2_threshold]
        .groupby("well")["cell_id"]
        .nunique()
    )

    # Align rejected_cells with total_cells, filling missing with 0
    rejected_cells = rejected_cells.reindex(total_cells.index, fill_value=0)

    # Calculate the proportion of rejected cells
    proportion_rejected = (rejected_cells / total_cells * 100).fillna(0)

    # Plotting
    ax1 = plt.gca()
    ax1.scatter(
        grouped_data["sigma_value"].index,
        grouped_data["sigma_value"]["count"].values,
        label="Accepted for analysis",
        color="blue",
    )
    ax1.set_xticks(wells)
    ax1.set_xticklabels(list(map(lambda x: x.replace(":", "\n"), labels)), rotation=45)
    ax1.set_ylabel("Number of cells", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.bar(
        proportion_rejected.index,
        proportion_rejected.values,
        color="red",
        alpha=0.5,
        label=f"Rejected (r2 < {r2_threshold})",
    )
    ax2.set_ylabel("Proportion Rejected (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 100)

    plt.title("Cell Counts and Rejection Rates")
    # To ensure legends from both axes are shown
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.tight_layout()
    plt.show()


def trim_cell_log_acf(processed_cell_log, cf):
    if not processed_cell_log or cf is None:
        return
    
    # Calculate modes and frames to keep
    if getattr(cf, "max_mode", None) == "auto":
        max_m = 0
    else:
        max_m = getattr(cf, "max_mode", 0)
        if not isinstance(max_m, int):
            max_m = 0
            
    auto_max_range = getattr(cf, "auto_max_mode_range", None)
    if auto_max_range is not None:
        auto_max = auto_max_range[1]
    else:
        auto_max = 0
        
    keep_modes = max(max_m, auto_max)
    save_acf_frames = getattr(cf, "save_acf_frames", 100)
    
    for row in processed_cell_log:
        # Check single entry
        if "fit_results" in row and row["fit_results"] is not None:
            res = row["fit_results"]
            if "autocorrelation_function" in res and res["autocorrelation_function"] is not None:
                res["autocorrelation_function"] = [
                    mode_acf[:save_acf_frames] for mode_acf in res["autocorrelation_function"][:keep_modes + 1]
                ]
            if "autocorrelation_fits" in res and res["autocorrelation_fits"] is not None:
                res["autocorrelation_fits"] = res["autocorrelation_fits"][:keep_modes + 1]
                
        # Check multientry
        if "results" in row and isinstance(row["results"], list):
            for entry_res in row["results"]:
                if "fit_results" in entry_res and entry_res["fit_results"] is not None:
                    res = entry_res["fit_results"]
                    if "autocorrelation_function" in res and res["autocorrelation_function"] is not None:
                        res["autocorrelation_function"] = [
                            mode_acf[:save_acf_frames] for mode_acf in res["autocorrelation_function"][:keep_modes + 1]
                        ]
                    if "autocorrelation_fits" in res and res["autocorrelation_fits"] is not None:
                        res["autocorrelation_fits"] = res["autocorrelation_fits"][:keep_modes + 1]


# TODO: clean up the fit_start, end etc overrides
def process_folder_caching(
    folder,
    fit_start=8,
    fit_end=21,
    temperature_c=25,
    method="rautu",
    reprocess=False,
    cf=None,
    cluster_client=None,
    contour_preprocessor=None,
    load_last_cache=False,
    pool=None,
):
    caching_processor_version = 2
    cell_log = load_cell_log(folder)
    metadata = cell_log["metadata"]
    metadata["folder"] = folder
    metadata["fit_start"] = fit_start
    metadata["fit_end"] = fit_end
    metadata["r2_threshold"] = -1  # both of these should be handled by filters
    metadata["valid_contours_threshold"] = 0
    metadata["temperature_c"] = temperature_c
    metadata["method"] = method

    if cf is None:
        cf = default_cf(metadata)

    if callable(cf) or contour_preprocessor is not None:
        logging.warning(
            "Caching is not fully supported for callable contour fitters or contour preprocessors"
        )
        cf_hash = "callable" + str(int(time()))
    else:
        cf_hash = cf.get_config_hash()
    current_data_file = (
        f"{folder}/processed_data_{cf_hash}_{caching_processor_version}.pkl"
    )
    plot_folder = f"{folder}/{cf_hash}_{caching_processor_version}/"
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    if isfile(current_data_file) and not reprocess:
        with open(current_data_file, "rb") as f:
            processed_data = pickle.load(f)
        print(f"Loaded data from {current_data_file}")

        trim_cell_log_acf(processed_data["processed_cell_log"], cf)
        return processed_data["processed_cell_log"], processed_data["metadata"]

    if load_last_cache and not reprocess:
        cache_files = glob(f"{folder}/processed_data_*.pkl")
        if cache_files:
            latest_file = max(cache_files, key=os.path.getmtime)
            with open(latest_file, "rb") as f:
                processed_data = pickle.load(f)
            print(f"Loaded most recent cache file: {latest_file}")
            trim_cell_log_acf(processed_data["processed_cell_log"], cf)
            return processed_data["processed_cell_log"], processed_data["metadata"]

    cell_log = cell_log["cells"]
    # glob is slow on network drives, prepare the relevant entries all at once
    all_files = glob(f"{folder}/*")

    for r in cell_log:
        r["contour_file"] = None

    for r in cell_log:
        k = r["global_id"]
        # not the fastest but should be easily fast enough
        for f in all_files:
            basename = os.path.basename(f)
            if basename.startswith(k):
                # Ensure it's not a substring of a larger identifier (e.g. repeat 1 matching 10)
                # or a group prefix collision (e.g. _inf matching _infected)
                suffix = basename[len(k):]
                if not suffix or not suffix[0].isalnum():
                    if "-preview.jpeg" in f:
                        r["preview_file"] = f
                    elif "contour" in f:
                        if isinstance(r["contour_file"], list):
                            r["contour_file"].append(f)
                        elif r["contour_file"] is not None:  # single file so far
                            r["contour_file"] = [r["contour_file"], f]
                        else:
                            r["contour_file"] = f
                    elif ".movie" in f:
                        r["movie_f"] = f

    # run fitting
    if cluster_client:
        # we have a dask cluster, use it instead of regular multiprocessing
        tasks = cluster_client.map(
            partial(
                process_entry,
                metadata=metadata,
                cf=cf,
                contour_preprocessor=contour_preprocessor,
                plot_folder=plot_folder,
            ),
            cell_log,
        )
        processed_cell_log = cluster_client.gather(tasks)
    else:
        # randomising processing order (probably remove, only useful for groups of different length videos)
        shuffle = np.random.permutation(len(cell_log))
        undo_shuffle = np.argsort(shuffle)

        # Use provided pool or create a local one
        if pool is None:
            # Local pool if none provided (legacy behavior)
            p = Pool(12, maxtasksperchild=10)
            must_close_pool = True
        else:
            p = pool
            must_close_pool = False

        try:
            processed_cell_log = []
            for r in tqdm(
                p.imap(
                    partial(
                        process_entry,
                        metadata=metadata,
                        cf=cf,
                        contour_preprocessor=contour_preprocessor,
                        plot_folder=plot_folder,
                    ),
                    [cell_log[j] for j in shuffle],
                ),
                total=len(cell_log),
            ):
                processed_cell_log.append(r)
        finally:
            if must_close_pool:
                p.close()
                p.join()

        processed_cell_log = [processed_cell_log[j] for j in undo_shuffle]

    processed_data = {}
    processed_data["fitting_config"] = {
        "fit_start": fit_start,
        "fit_end": fit_end,
        "temperature_c": temperature_c,
        "method": method,
        "fitter_config": "callable" if callable(cf) else cf.get_config_dict(),
    }

    # processed_data["collected_data"] = collected_data
    # processed_data["wells"] = wells
    processed_data["processed_cell_log"] = processed_cell_log
    processed_data["metadata"] = metadata

    with open(current_data_file, "wb") as f:
        pickle.dump(processed_data, f)

    return processed_cell_log, metadata


def get_data_from_folder(
    folder,
    fit_start=8,
    fit_end=21,
    temperature_c=25,
    filters=None,
    method="rautu",
    reprocess=False,
    fitter=None,
    cluster_client=None,
    contour_preprocessor=None,
    load_last_cache=False,
    pool=None,
):
    processed_cell_log, metadata = process_folder_caching(
        folder,
        fit_start,
        fit_end,
        temperature_c,
        method,
        reprocess,
        cf=fitter,
        cluster_client=cluster_client,
        contour_preprocessor=contour_preprocessor,
        load_last_cache=load_last_cache,
        pool=pool,
    )

    # collect data from runs
    collected_data = []
    columns = [
        "well",
        "cell_id",
        "global_id",
        "time",
        "pfs_offset",
        "radius",
        "fit_status",
        "valid_p",
    ]
    columns += [
        "sigma_value",
        "sigma_error",
        "kappa_value",
        "kappa_error",
        "r2",
        "radius_value",
    ]
    columns += ["fit_p", "fit_cross_rate", "group"]
    have_contour_variance = False
    have_viscosity = False
    for row in processed_cell_log:
        if "fit_results" in row:
            if "sigma" in row["fit_results"]:
                if (
                    "contour_variance" in row["fit_results"]
                    and not have_contour_variance
                ):
                    columns += ["contour_variance", "contour_variance_butter"]
                    have_contour_variance = True

            if "viscosity_fit" in row["fit_results"] and not have_viscosity:
                columns += ["viscosity_value", "viscosity_error", "viscosity_r2"]
                have_viscosity = True

            # Check both if possible, but break if we found both or have seen enough keys
            if have_contour_variance and have_viscosity:
                break

    wells = set()

    for row in processed_cell_log:
        if filters is not None:
            if not filters(row):
                row["fit_status"] = "filter_reject"
        well = row["well"]
        wells.add(well)
        # repeat = int(row["global_id"].split("-")[-1])
        rec_data = [
            well,
            row["cell_id"],
            row["global_id"],
            row["start_time"],
            row["pfs_offset"],
            row["radius"],
            row.get("fit_status", "unknown"),
        ]  #

        if "fit_status" in row and row["fit_status"] == "success":
            rec_data += [
                row["valid_contour_rate"],
                row["fit_results"]["sigma"]["value"],
                row["fit_results"]["sigma"]["error"],
                row["fit_results"]["kappa"]["value"],
                row["fit_results"]["kappa"]["error"],
                row["fit_results"]["r2"],
                row["fit_results"]["radius"]["value"],
                row["fit_results"]["fit_p"],
                row["fit_results"]["fit_cross_rate"],
                get_group_from_global_id(row["global_id"]),
            ]
            if have_contour_variance:
                rec_data += [
                    row["fit_results"].get("contour_variance", np.nan),
                    row["fit_results"].get("contour_variance_butter", np.nan),
                ]
            if have_viscosity:
                if (
                    "viscosity_fit" in row["fit_results"]
                    and "eta_in" in row["fit_results"]["viscosity_fit"]
                ):
                    rec_data += [
                        row["fit_results"]["viscosity_fit"]["eta_in"]["value"],
                        row["fit_results"]["viscosity_fit"]["eta_in"]["error"],
                        row["fit_results"]["viscosity_fit"]["r2"],
                    ]
                else:
                    rec_data += [np.nan, np.nan, np.nan]
        else:
            rec_data += [
                row["valid_contour_rate"] if "valid_contour_rate" in row else np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                row["fit_results"]["r2"] if "fit_results" in row else np.nan,
                (
                    row["fit_results"]["radius"]["value"]
                    if "fit_results" in row
                    else np.nan
                ),
                row["fit_results"]["fit_p"] if "fit_results" in row else np.nan,
                (
                    row["fit_results"]["fit_cross_rate"]
                    if "fit_results" in row
                    else np.nan
                ),
                get_group_from_global_id(row["global_id"]),
            ]
            if have_contour_variance:
                rec_data += [
                    (
                        row["fit_results"]["contour_variance"]
                        if "fit_results" in row
                        and "contour_variance" in row["fit_results"]
                        else np.nan
                    ),
                    (
                        row["fit_results"]["contour_variance_butter"]
                        if "fit_results" in row
                        and "contour_variance_butter" in row["fit_results"]
                        else np.nan
                    ),
                ]
            if have_viscosity:
                rec_data += [np.nan, np.nan, np.nan]
        collected_data.append(rec_data)

    wells = list(wells)
    collected_data = pd.DataFrame(data=collected_data, columns=columns)
    # collected_data["time"] -= min(collected_data["time"])

    return collected_data, wells, processed_cell_log, metadata


def process_folders(
    folders,
    folder_labels,
    well_labelss,
    fit_starts,
    fit_ends,
    temperatures_c,
    filters=None,
    method="rautu",
    keep_order=True,
    correction_per_well=True,
    reprocess=False,
    fitter=None,
    cluster_client=None,
    contour_preprocessor=None,
    r2_plot=0.92,
    plot=True,
    load_last_cache=False,
):
    all_data = pd.DataFrame([])
    all_time_gradients = pd.DataFrame([])

    full_processed_log = []
    all_metadata = {}
    all_wells = []
    all_labels = []

    filters_list = filters is not None and isinstance(filters, list)

    # Use a forkserver pool to avoid memory bloat from the main process
    # Workers started from 'forkserver' are clean and don't inherit the parent's memory
    ctx = multiprocessing.get_context("forkserver")
    with ctx.Pool(12, maxtasksperchild=10) as pool:
        for i in tqdm(range(len(folders))):
            collected_data, wells, processed_cell_log, metadata = get_data_from_folder(
                folders[i],
                fit_starts[i],
                fit_ends[i],
                temperatures_c[i],
                filters if not filters_list else filters[i],
                method,
                reprocess,
                fitter=fitter,
                cluster_client=cluster_client,
                contour_preprocessor=contour_preprocessor,
                load_last_cache=load_last_cache,
                pool=pool,
            )

            time_gradients = get_time_gradients(collected_data)
            collected_data = apply_time_correction(
                collected_data, time_gradients, correction_per_well
            )

            if well_labelss[i] is not None:
                if keep_order:  # TODO: this relies on order in dict
                    wells = list(well_labelss[i].keys())
                    labels = list(well_labelss[i].values())
                else:
                    labels = remap_labels(
                        wells, well_labelss[i]
                    )  # this is now a list in the same order as wells
            else:
                labels = wells

            labels = list(map(lambda x: folder_labels[i] + x, labels))
            all_labels += labels
            wells_col = list(
                map(lambda x: folder_labels[i] + x, collected_data["well"])
            )
            well_entry_label = list(
                map(
                    lambda x: (
                        x
                        if well_labelss[i] is None or x not in well_labelss[i]
                        else well_labelss[i][x]
                    ),
                    collected_data["well"],
                )
            )
            well_run_label = list(
                map(
                    lambda x: folder_labels[i]
                    + (
                        x
                        if well_labelss[i] is None or x not in well_labelss[i]
                        else well_labelss[i][x]
                    ),
                    collected_data["well"],
                )
            )
            collected_data["well"] = wells_col
            collected_data.insert(1, "sample", well_entry_label)
            collected_data.insert(3, "sample_run", well_run_label)
            collected_data.insert(2, "run", folder_labels[i])

            wells_grads = list(
                map(lambda x: folder_labels[i] + x, time_gradients["well"])
            )
            time_gradients["well"] = wells_grads

            wells = list(map(lambda x: folder_labels[i] + x, wells))
            all_wells += wells
            for w in wells:
                all_metadata[w] = metadata
            all_data = pd.concat([all_data, collected_data])
            all_time_gradients = pd.concat([all_time_gradients, time_gradients])
            full_processed_log += processed_cell_log

    if plot:
        plot_all(all_data, all_wells, all_labels, r2_plot=r2_plot)
        plot_time_gradients(all_time_gradients, wells=all_wells, labels=all_labels)

    return full_processed_log, all_metadata, all_data, all_wells, all_labels


def plot_all(collected_data, wells, labels, r2_plot=0.92):
    wl = [
        v
        for i, v in enumerate(zip(wells, labels))
        if v not in list(zip(wells, labels))[:i]
    ]
    # wl = list(set(zip(wells, labels)))
    plot_collected_data_comparison(
        collected_data, "radius_value", "Radius/nm", wells, labels=labels
    )
    plot_collected_data_comparison(
        collected_data, "sigma_value", "Tension/1e-7 N/m", wells, labels=labels
    )
    plot_collected_data_comparison(
        collected_data, "sigma_corrected", "Tension/1e-7 N/m", wells, labels=labels
    )
    plt.title("With time correction")
    plot_collected_data_comparison(
        collected_data, "kappa_value", "Bending modulus/1e-21 J", wells, labels=labels
    )
    plot_collected_data_comparison(
        collected_data,
        "kappa_corrected",
        "Bending modulus/1e-21 J",
        wells,
        labels=labels,
    )
    plt.title("With time correction")
    plot_collected_data_comparison(
        collected_data, "time", "Time/s", wells, labels=labels
    )

    if "contour_variance" in collected_data.columns:
        plot_collected_data_comparison(
            collected_data,
            "contour_variance",
            "Contour variance/nm",
            wells,
            labels=labels,
        )
        plt.ylim((5, 60))
        plot_collected_data_comparison(
            collected_data,
            "contour_variance_butter",
            "Contour variance (filtered)/nm",
            wells,
            labels=labels,
        )
        plt.ylim((5, 30))

    plot_collected_data_time(
        collected_data, "radius_value", "Radius/nm", wells, labels=labels
    )
    plot_collected_data_time(
        collected_data,
        "sigma_value",
        "Tension/1e-7 N/m",
        wells,
        err_column="sigma_error",
        labels=labels,
    )
    plot_collected_data_time(
        collected_data,
        "kappa_value",
        "Bending modulus/1e-21 J",
        wells,
        err_column="kappa_error",
        labels=labels,
    )
    if "contour_variance" in collected_data.columns:
        plot_collected_data_time(
            collected_data,
            "contour_variance",
            "Contour variance/nm",
            wells,
            labels=labels,
        )
        plt.ylim((5, 30))
        plot_collected_data_time(
            collected_data,
            "contour_variance_butter",
            "Contour variance (filtered)/nm",
            wells,
            labels=labels,
        )
        plt.ylim((5, 30))

    grouped = get_group_results(collected_data)

    plot_cell_numbers(
        grouped, wells, labels, collected_data, r2_plot
    )  # TODO: get r2 from config


# TODO: unify use process_folders
def process_folder(
    folder,
    fit_start=8,
    fit_end=21,
    well_labels=None,
    temperature_c=25,
    filters=None,
    method="rautu",
    keep_order=True,
    cluster_client=None,
    contour_preprocessor=None,
):
    collected_data, wells, processed_cell_log, metadata = get_data_from_folder(
        folder,
        fit_start,
        fit_end,
        temperature_c,
        filters,
        method,
        cluster_client=cluster_client,
        contour_preprocessor=contour_preprocessor,
    )
    wells = list(wells)
    if well_labels is not None:
        if keep_order:  # TODO: this relies on order in dict
            wells = list(well_labels.keys())
            labels = list(well_labels.values())
        else:
            labels = remap_labels(wells, well_labels)
    else:
        labels = wells

    time_gradients = get_time_gradients(collected_data)
    collected_data = apply_time_correction(collected_data, time_gradients, False)

    plot_all(collected_data, wells, labels)
    plot_time_gradients(time_gradients, wells=wells, labels=labels)

    return processed_cell_log, metadata, collected_data


def remove_outlier_wells_by_gradient(
    collected_data,
    filter_by_sigma=True,
    filter_by_kappa=False,
    max_sigma_gradient_per_hour=None,
    max_kappa_gradient_per_hour=None,
    std_dev_threshold=2.0,
):
    """
    Removes outlier wells based on the consistency of tension (sigma) and bending modulus (kappa) gradients.
    Calculates gradients internally, converts them to hourly rates, and is optimized for small sample sizes.

    Args:
        collected_data (pd.DataFrame): DataFrame containing the collected data, must include 'well' and 'sample' columns.
        max_sigma_gradient_per_hour (float, optional): Maximum allowed absolute hourly gradient for sigma. Wells exceeding this will be removed.
        max_kappa_gradient_per_hour (float, optional): Maximum allowed absolute hourly gradient for kappa. Wells exceeding this will be removed.
        std_dev_threshold (float, optional): The number of standard deviations from the sample's mean gradient
                                             to consider a well an outlier (used for >3 wells). Defaults to 2.0.

    Returns:
        pd.DataFrame: A copy of the collected_data with outlier wells removed.
    """
    if "sample" not in collected_data.columns:
        logging.warning(
            "No 'sample' column in collected_data. Cannot perform outlier detection based on sample groups."
        )
        return collected_data.copy()

    # Internal gradient calculation
    time_gradients = get_time_gradients(collected_data)

    # Convert gradients to per-hour rates
    for col in ["sigma_slope", "kappa_slope"]:
        time_gradients[col] *= 3600

    # Create a mapping from well to sample
    well_to_sample = collected_data[["well", "sample"]].drop_duplicates()
    gradients_with_sample = pd.merge(time_gradients, well_to_sample, on="well")

    outlier_wells = set()
    initial_well_counts = gradients_with_sample.groupby("sample")["well"].count()

    # Check for absolute max gradients
    if max_sigma_gradient_per_hour is not None:
        outliers = gradients_with_sample[
            np.abs(gradients_with_sample["sigma_slope"]) > max_sigma_gradient_per_hour
        ]
        for _, row in outliers.iterrows():
            #            print(f"Well '{row['well']}' ({row['sample']}) removed: sigma gradient/h {row['sigma_slope']:.2e} exceeds max {max_sigma_gradient_per_hour:.2e}")
            outlier_wells.add(row["well"])

    if max_kappa_gradient_per_hour is not None:
        outliers = gradients_with_sample[
            np.abs(gradients_with_sample["kappa_slope"]) > max_kappa_gradient_per_hour
        ]
        for _, row in outliers.iterrows():
            #            print(f"Well '{row['well']}' ({row['sample']}) removed: kappa gradient/h {row['kappa_slope']:.2e} exceeds max {max_kappa_gradient_per_hour:.2e}")
            outlier_wells.add(row["well"])

    # Check for consistency within samples
    samples = gradients_with_sample["sample"].unique()
    for sample in samples:
        sample_grads = gradients_with_sample[
            gradients_with_sample["sample"] == sample
        ].copy()

        if len(sample_grads) < 3:
            continue  # Not enough data to find outliers

        grad_types = []
        if filter_by_sigma:
            grad_types.append("sigma_slope")
        if filter_by_kappa:
            grad_types.append("kappa_slope")

        for grad_type in grad_types:
            if len(sample_grads) == 3:
                # For exactly 3 wells, find the one furthest from the median
                median = sample_grads[grad_type].median()
                sample_grads["abs_dev"] = np.abs(sample_grads[grad_type] - median)
                outlier_index = sample_grads["abs_dev"].idxmax()
                outlier_well = sample_grads.loc[outlier_index]

                sorted_devs = (
                    sample_grads["abs_dev"].sort_values(ascending=False).to_numpy()
                )
                if (
                    len(sorted_devs) > 1
                    and sorted_devs[0] > std_dev_threshold * sorted_devs[1]
                ):
                    if outlier_well["well"] not in outlier_wells:
                        print(
                            f"Well '{outlier_well['well']}' removed as outlier for sample '{sample}' (n=3): {grad_type}/h of {outlier_well[grad_type]:.2e} is far from median {median:.2e}"
                        )
                        outlier_wells.add(outlier_well["well"])

            else:  # More than 3 wells, use std dev method
                mean = sample_grads[grad_type].mean()
                std = sample_grads[grad_type].std()
                if std == 0:
                    continue

                is_outlier = (
                    np.abs(sample_grads[grad_type] - mean) > std_dev_threshold * std
                )
                outliers = sample_grads[is_outlier]
                for _, row in outliers.iterrows():
                    if row["well"] not in outlier_wells:
                        print(
                            f"Well '{row['well']}' removed as outlier for sample '{sample}': {grad_type}/h of {row[grad_type]:.2e} is far from mean {mean:.2e} (std: {std:.2e})"
                        )
                        outlier_wells.add(row["well"])

    # Check for majority removal
    if outlier_wells:
        removed_well_counts = (
            collected_data[collected_data["well"].isin(outlier_wells)]
            .groupby("sample")["well"]
            .nunique()
        )
        for sample, removed_count in removed_well_counts.items():
            if removed_count >= initial_well_counts[sample] / 2:
                logging.warning(
                    f"Majority of wells ({removed_count}/{initial_well_counts[sample]}) removed for sample '{sample}'."
                )

    if not outlier_wells:
        #        print("No outlier wells detected based on gradients.")
        return collected_data.copy()

    #    print(f"\nRemoving data for {len(outlier_wells)} outlier wells: {', '.join(sorted(list(outlier_wells)))}")
    return collected_data[~collected_data["well"].isin(outlier_wells)].copy()


# chatgpt so very chatty, fixed
def get_time_gradients(collected_data, include_sample=False):
    # Ensure columns are numeric to prevent TypeErrors when using np.isnan or comparison operators
    collected_data = collected_data.copy()
    collected_data["time"] = pd.to_numeric(collected_data["time"], errors="coerce")
    collected_data["sigma_value"] = pd.to_numeric(collected_data["sigma_value"], errors="coerce")
    collected_data["kappa_value"] = pd.to_numeric(collected_data["kappa_value"], errors="coerce")

    # Extract global minimum time
    global_min_time = collected_data["time"].min()

    # Linear fit function
    def linear_fit(x, y):
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_filtered = x[mask].to_numpy()
        y_filtered = y[mask].to_numpy()
        if len(x_filtered) < 2:
            return np.nan, np.nan, np.nan, np.nan
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)
            intercept_err = std_err * np.sqrt(np.mean(x_filtered**2))
            return slope, intercept, std_err, intercept_err
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    # Get unique wells
    wells = collected_data["well"].unique()

    # DataFrame to store the results
    if include_sample:
        gradients_df = pd.DataFrame(
            columns=[
                "well",
                "sample",
                "sigma_slope",
                "sigma_slope_err",
                "sigma_intercept",
                "sigma_intercept_err",
                "kappa_slope",
                "kappa_slope_err",
                "kappa_intercept",
                "kappa_intercept_err",
            ]
        )
    else:
        gradients_df = pd.DataFrame(
            columns=[
                "well",
                "sigma_slope",
                "sigma_slope_err",
                "sigma_intercept",
                "sigma_intercept_err",
                "kappa_slope",
                "kappa_slope_err",
                "kappa_intercept",
                "kappa_intercept_err",
            ]
        )

    # Iterate through each well
    for well in wells:
        well_df = collected_data[collected_data["well"] == well]
        # Fit sigma_value against time for the current well
        if np.sum(well_df["sigma_value"] > 0) < 2:
            logging.warning(f"Not enough data to fit for well {well}")
            continue
        sigma_slope, sigma_intercept, sigma_slope_err, sigma_intercept_err = linear_fit(
            well_df["time"], well_df["sigma_value"]
        )

        # Fit kappa_value against time for the current well
        kappa_slope, kappa_intercept, kappa_slope_err, kappa_intercept_err = linear_fit(
            well_df["time"], well_df["kappa_value"]
        )

        # Append the results to the DataFrame
        data = {
            "well": well,
            "sigma_slope": sigma_slope,
            "sigma_slope_err": sigma_slope_err,
            "sigma_intercept": sigma_intercept,
            "sigma_intercept_err": sigma_intercept_err,
            "kappa_slope": kappa_slope,
            "kappa_slope_err": kappa_slope_err,
            "kappa_intercept": kappa_intercept,
            "kappa_intercept_err": kappa_intercept_err,
        }
        if include_sample:
            data["sample"] = well_df["sample"].iloc[0]

        gradients_df = pd.concat(
            [gradients_df, pd.DataFrame([data])], ignore_index=True
        )

    # Fit sigma_value against time for all data
    (
        global_sigma_slope,
        global_sigma_intercept,
        global_sigma_slope_err,
        global_sigma_intercept_err,
    ) = linear_fit(collected_data["time"], collected_data["sigma_value"])

    # Fit kappa_value against time for all data
    (
        global_kappa_slope,
        global_kappa_intercept,
        global_kappa_slope_err,
        global_kappa_intercept_err,
    ) = linear_fit(collected_data["time"], collected_data["kappa_value"])

    # Add the global gradients to the DataFrame
    gradients_df = pd.concat(
        [
            gradients_df,
            pd.DataFrame(
                [
                    {
                        "well": "GLOBAL",
                        "sigma_slope": global_sigma_slope,
                        "sigma_slope_err": global_sigma_slope_err,
                        "sigma_intercept": global_sigma_intercept,
                        "sigma_intercept_err": global_sigma_intercept_err,
                        "kappa_slope": global_kappa_slope,
                        "kappa_slope_err": global_kappa_slope_err,
                        "kappa_intercept": global_kappa_intercept,
                        "kappa_intercept_err": global_kappa_intercept_err,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    return gradients_df


# chatgpt
def apply_time_correction(df, gradients_df, per_well=False):
    # Extract global minimum time
    global_min_time = df["time"].min()

    if per_well:
        # Initialize corrected columns
        df["sigma_corrected"] = np.nan
        df["kappa_corrected"] = np.nan

        # Iterate through each well
        for index, row in gradients_df.iterrows():
            well = row["well"]
            if well == "GLOBAL":
                continue

            sigma_slope = row["sigma_slope"]
            kappa_slope = row["kappa_slope"]

            # Correct sigma_value and kappa_value to the global minimum time for the current well
            df.loc[df["well"] == well, "sigma_corrected"] = df.loc[
                df["well"] == well, "sigma_value"
            ] - sigma_slope * (df.loc[df["well"] == well, "time"] - global_min_time)
            df.loc[df["well"] == well, "kappa_corrected"] = df.loc[
                df["well"] == well, "kappa_value"
            ] - kappa_slope * (df.loc[df["well"] == well, "time"] - global_min_time)
    else:
        # Use global gradients for correction
        global_gradients = gradients_df[gradients_df["well"] == "GLOBAL"].iloc[0]
        global_sigma_slope = global_gradients["sigma_slope"]
        global_kappa_slope = global_gradients["kappa_slope"]

        # Correct sigma_value to the global minimum time
        df["sigma_corrected"] = df["sigma_value"] - global_sigma_slope * (
            df["time"] - global_min_time
        )

        # Correct kappa_value to the global minimum time
        df["kappa_corrected"] = df["kappa_value"] - global_kappa_slope * (
            df["time"] - global_min_time
        )

    return df


def check_sample_distinguishability(
    dataset: pd.DataFrame,
    control_label: str,
    target_label: str,
    column: str = "sigma_value",
    p_val: float = 0.05,
    k_target: int = 1,
    r_control: int = 1,
    print_p_values_grid: bool = False,
) -> tuple[
    bool, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, float, tuple[float, float]
]:  # Updated return type
    """
    Checks the statistical distinguishability between target and control sample groups,
    returning a boolean result, grids of pairwise p-values, mean differences, and
    standard errors of differences, along with a count of distinguishable target wells.
    It also returns the overall mean difference between the target and control samples
    and its 95% confidence interval.

    Args:
        dataset (pd.DataFrame): DataFrame with 'well', 'sample', and `column` as columns.
        control_label (str): The label (from the 'sample' column) for the control group.
        target_label (str): The label (from the 'sample' column) for the target group.
        column (str, optional): The name of the column containing numeric values for comparison.
                                Defaults to "sigma_value".
        p_val (float, optional): The p-value threshold for distinguishability. Defaults to 0.05.
        k_target (int, optional): The minimum number of target wells that must be
                                  distinguishable for the overall boolean result to be True.
                                  A target well is considered distinguishable if it meets
                                  the `r_control` criterion. Defaults to 1.
        r_control (int, optional): The minimum number of control wells a single target well
                                   must be distinguishable from (p < p_val) for that target
                                   well to count towards `k_target`. Defaults to 1.
        print_p_values_grid (bool, optional): If True, prints color-coded DataFrames of
                                            pairwise p-values, mean differences, and
                                            standard errors of differences. Defaults to False.

    Returns:
        tuple[bool, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, float, tuple[float, float]]:
            - bool: True if at least `k_target` target wells are found to be
                    distinguishable from at least `r_control` control wells; False otherwise.
            - pd.DataFrame: A DataFrame where rows are target wells, columns are control wells,
                            and values are the p-values from t-tests comparing each pair.
            - pd.DataFrame: A DataFrame where rows are target wells, columns are control wells,
                            and values are the mean differences (Target Mean - Control Mean).
            - pd.DataFrame: A DataFrame where rows are target wells, columns are control wells,
                            and values are the standard errors of the mean differences.
            - int: The actual count of target wells that were distinguishable from
                   at least `r_control` control wells.
            - float: The overall mean difference between the target and control samples.
            - tuple[float, float]: The 95% confidence interval for the overall mean difference.
    """
    # --- Input Validation ---
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Input 'dataset' must be a pandas DataFrame.")
    if "well" not in dataset.columns:
        raise ValueError("The 'dataset' DataFrame must contain a 'well' column.")
    if "sample" not in dataset.columns:
        raise ValueError("The 'dataset' DataFrame must contain a 'sample' column.")
    if column not in dataset.columns:
        raise ValueError(f"The 'dataset' DataFrame must contain a '{column}' column.")
    if not np.issubdtype(dataset[column].dtype, np.number):
        raise ValueError(f"The '{column}' column must contain numeric data.")
    if control_label not in dataset["sample"].unique():
        raise ValueError(
            f"Control label '{control_label}' not found in 'sample' column."
        )
    if target_label not in dataset["sample"].unique():
        raise ValueError(f"Target label '{target_label}' not found in 'sample' column.")
    if not isinstance(p_val, (int, float)) or not (0 <= p_val <= 1):
        raise ValueError("Input 'p_val' must be a numeric value between 0 and 1.")
    if not isinstance(k_target, int) or k_target < 1:
        raise ValueError("Input 'k_target' must be a positive integer.")
    if not isinstance(r_control, int) or r_control < 1:
        raise ValueError("Input 'r_control' must be a positive integer.")
    if not isinstance(print_p_values_grid, bool):
        raise TypeError("Input 'print_p_values_grid' must be a boolean.")

    # --- Data Filtering and Pre-processing ---
    control_df = dataset[dataset["sample"] == control_label]
    target_df = dataset[dataset["sample"] == target_label]

    if control_df.empty:
        print(
            f"Warning: No data found for control label '{control_label}'. Cannot perform comparisons."
        )
        return (
            False,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            0,
            np.nan,
            (np.nan, np.nan),
        )
    if target_df.empty:
        print(
            f"Warning: No data found for target label '{target_label}'. Cannot perform comparisons."
        )
        return (
            False,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            0,
            np.nan,
            (np.nan, np.nan),
        )

    control_wells = control_df["well"].unique()
    target_wells = target_df["well"].unique()

    if len(control_wells) == 0:
        print(
            f"Warning: No unique wells found for control label '{control_label}'. Cannot perform comparisons."
        )
        return (
            False,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            0,
            np.nan,
            (np.nan, np.nan),
        )
    if len(target_wells) == 0:
        print(
            f"Warning: No unique wells found for target label '{target_label}'. Cannot perform comparisons."
        )
        return (
            False,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            0,
            np.nan,
            (np.nan, np.nan),
        )

    # Store well data for efficient access
    control_well_data = {
        well: control_df[control_df["well"] == well][column].values
        for well in control_wells
    }
    target_well_data = {
        well: target_df[target_df["well"] == well][column].values
        for well in target_wells
    }

    # --- Initialize Matrices for Results ---
    num_target_wells = len(target_wells)
    num_control_wells = len(control_wells)
    p_values_matrix = np.full((num_target_wells, num_control_wells), np.nan)
    mean_diff_matrix = np.full((num_target_wells, num_control_wells), np.nan)
    se_diff_matrix = np.full((num_target_wells, num_control_wells), np.nan)

    # --- Calculate Metrics Grid ---
    for i, target_well in enumerate(target_wells):
        target_values = target_well_data[target_well]
        mean_target_well = np.nanmean(target_values)
        std_target_well = np.nanstd(
            target_values, ddof=1
        )  # ddof=1 for sample standard deviation
        n_target_well = len(target_values)

        sem_target = np.nan  # Initialize as NaN
        if n_target_well >= 2:
            sem_target = std_target_well / np.sqrt(n_target_well)

        for j, control_well in enumerate(control_wells):
            control_values = control_well_data[control_well]
            mean_control_well = np.nanmean(control_values)
            std_control_well = np.nanstd(
                control_values, ddof=1
            )  # ddof=1 for sample standard deviation
            n_control_well = len(control_values)

            sem_control = np.nan  # Initialize as NaN
            if n_control_well >= 2:
                sem_control = std_control_well / np.sqrt(n_control_well)

            # Calculate Mean Difference
            mean_diff_matrix[i, j] = mean_target_well - mean_control_well

            # Calculate Standard Error of the Difference
            if not np.isnan(sem_target) and not np.isnan(sem_control):
                se_diff_matrix[i, j] = np.sqrt(sem_target**2 + sem_control**2)
            else:
                se_diff_matrix[i, j] = np.nan  # If either SEM is NaN, SE_diff is NaN

            # Perform independent t-test (Welch's t-test for unequal variances)
            if n_target_well >= 2 and n_control_well >= 2:
                statistic, pvalue = ttest_ind(
                    target_values, control_values, equal_var=False, nan_policy="omit"
                )
                p_values_matrix[i, j] = pvalue
            else:
                # print(f"Warning: Insufficient data for t-test for T:{target_well} vs C:{control_well}. Needs at least 2 data points per well.")
                p_values_matrix[i, j] = (
                    np.nan
                )  # Cannot perform t-test with less than 2 data points

    p_value_grid_df = pd.DataFrame(
        p_values_matrix, index=target_wells, columns=control_wells
    )
    mean_diff_grid_df = pd.DataFrame(
        mean_diff_matrix, index=target_wells, columns=control_wells
    )
    se_diff_grid_df = pd.DataFrame(
        se_diff_matrix, index=target_wells, columns=control_wells
    )

    # --- Determine Overall Distinguishability (Boolean Result and Count) ---
    distinguishable_target_count = 0

    if r_control > num_control_wells:
        # If r_control is greater than available control wells, no target well can meet the criterion
        overall_distinguishable = False
    else:
        for target_well in target_wells:
            row_p_values = p_value_grid_df.loc[target_well].dropna()
            if not row_p_values.empty:
                if np.sum(row_p_values < p_val) >= r_control:
                    distinguishable_target_count += 1
        overall_distinguishable = distinguishable_target_count >= k_target

    # --- Calculate Overall Group Statistics and Confidence Interval ---
    all_control_values = dataset[dataset["sample"] == control_label][column].values
    all_target_values = dataset[dataset["sample"] == target_label][column].values

    mean_all_control = np.nanmean(all_control_values)
    std_all_control = np.nanstd(all_control_values, ddof=1)
    n_all_control = len(all_control_values)
    sem_all_control = (
        std_all_control / np.sqrt(n_all_control) if n_all_control >= 2 else np.nan
    )

    mean_all_target = np.nanmean(all_target_values)
    std_all_target = np.nanstd(all_target_values, ddof=1)
    n_all_target = len(all_target_values)
    sem_all_target = (
        std_all_target / np.sqrt(n_all_target) if n_all_target >= 2 else np.nan
    )

    overall_mean_difference = mean_all_target - mean_all_control

    overall_se_difference = np.nan
    if not np.isnan(sem_all_target) and not np.isnan(sem_all_control):
        overall_se_difference = np.sqrt(sem_all_target**2 + sem_all_control**2)

    # Calculate 95% Confidence Interval for overall difference (using Z-score for large samples)
    z_score_95_ci = 1.96  # Approximate Z-score for 95% CI
    lower_bound_ci = np.nan
    upper_bound_ci = np.nan

    if not np.isnan(overall_se_difference):
        margin_of_error = z_score_95_ci * overall_se_difference
        lower_bound_ci = overall_mean_difference - margin_of_error
        upper_bound_ci = overall_mean_difference + margin_of_error
    overall_confidence_interval = (lower_bound_ci, upper_bound_ci)

    # --- Optional: Print Color-Coded P-value Grid and other grids ---
    if print_p_values_grid:
        # Define a flexible styling function for general numeric grids (mean diff, SE diff)
        def _style_numeric_grid(val):
            if pd.isna(val):
                return "color: #777;"  # Muted color for NaN
            return "color: white;"  # Default text color for dark mode

        # Helper function for p-value grid, optimized for dark mode
        def _highlight_p_values_dark_mode(val, p_threshold_display):
            if pd.isna(val):
                return "color: #777;"  # Muted color for NaN
            return (
                "background-color: #28a745; color: black;"
                if val < p_threshold_display
                else "background-color: #343a40; color: white;"
            )  # Bolder green for significant, dark background/white text otherwise

        print(f"\n--- P-value Grid (p < {p_val} is significant) ---")
        styled_p_grid = p_value_grid_df.style.applymap(
            lambda x: _highlight_p_values_dark_mode(x, p_val)
        )
        display(styled_p_grid)

        print("\n--- Mean Difference Grid (Target Mean - Control Mean) ---")
        styled_mean_diff_grid = mean_diff_grid_df.style.applymap(_style_numeric_grid)
        display(styled_mean_diff_grid)

        print("\n--- Standard Error of Difference Grid ---")
        styled_se_diff_grid = se_diff_grid_df.style.applymap(_style_numeric_grid)
        display(styled_se_diff_grid)

    return (
        overall_distinguishable,
        p_value_grid_df,
        mean_diff_grid_df,
        se_diff_grid_df,
        distinguishable_target_count,
        overall_mean_difference,
        overall_confidence_interval,
    )


def plot_multirun_analysis_results(
    results: dict, effect_mode: str, column: str, counts: dict
):
    """
    Plots the results of the multi-run analysis, showing mean effect, error bars,
    and significance stars.
    """
    target_labels = [label for label, res in results.items() if res is not None]
    if not target_labels:
        print("No valid results to plot.")
        return

    mean_effects = [results[label]["mean_effect"] for label in target_labels]
    se_effects = [results[label]["se_effect"] for label in target_labels]
    p_values = [results[label]["p_value"] for label in target_labels]

    # Add counts to labels for plotting
    plot_labels = [
        f"{label}\n(n={counts.get(label, 'N/A')})" for label in target_labels
    ]

    fig, ax = plt.subplots(dpi=150, figsize=(max(6, len(target_labels) * 0.8), 5))

    bars = ax.bar(
        plot_labels,
        mean_effects,
        yerr=se_effects,
        capsize=5,
        color="skyblue",
        edgecolor="black",
    )

    # Add significance stars
    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            y_pos = bars[i].get_height() + se_effects[i]
            ax.text(
                bars[i].get_x() + bars[i].get_width() / 2,
                y_pos,
                "*",
                ha="center",
                va="bottom",
                color="red",
                fontsize=16,
            )

    # Set y-axis label and reference line
    if effect_mode == "difference":
        ylabel = f"Mean Effect ({column} Difference)"
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    elif effect_mode == "ratio":
        ylabel = f"Mean Effect ({column} Ratio)"
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
    else:
        ylabel = f"Mean Effect ({effect_mode}) on {column}"

    ax.set_ylabel(ylabel)
    ax.set_title("Multi-run Experiment Analysis (Weighted)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_normalized_analysis_results(
    results: dict,
    column: str,
    counts: dict,
    control_label: str,
    plot_absolute: bool = False,
    dataset: pd.DataFrame = None,
):
    """
    Plots the results of the normalized control analysis, showing mean effect,
    error bars, and significance stars.
    """
    target_labels = [label for label, res in results.items() if res is not None]
    if not target_labels:
        print("No valid normalized results to plot.")
        return

    if plot_absolute:
        if dataset is None:
            raise ValueError("Dataset must be provided when plot_absolute is True.")

        control_mean = dataset[dataset["sample"] == control_label][column].mean()

        # Include control in the plot
        plot_labels = [control_label] + target_labels
        mean_effects = [control_mean] + [
            res["mean_effect"] for res in results.values() if res is not None
        ]
        se_effects = [dataset[dataset["sample"] == control_label][column].sem()] + [
            res["se_effect"] for res in results.values() if res is not None
        ]
        p_values = [1.0] + [
            res["p_value"] for res in results.values() if res is not None
        ]  # Control has no p-value
        colors = ["gray"] + ["skyblue"] * len(target_labels)

        title = f"Absolute {column} by Sample"
        ylabel = f"Mean {column}"
    else:
        # Normalized plot
        plot_labels = target_labels
        mean_effects = [results[label]["mean_effect"] for label in target_labels]
        se_effects = [results[label]["se_effect"] for label in target_labels]
        p_values = [results[label]["p_value"] for label in target_labels]
        colors = ["skyblue"] * len(target_labels)

        title = f"Normalized {column} Effect vs. {control_label}"
        ylabel = f"Mean Effect (Normalized to Control)"

    # Add counts to labels for plotting
    plot_labels = [f"{label}\n(n={counts.get(label, 'N/A')})" for label in plot_labels]

    fig, ax = plt.subplots(dpi=150, figsize=(max(6, len(plot_labels) * 0.8), 5))

    bars = ax.bar(
        plot_labels,
        mean_effects,
        yerr=se_effects,
        capsize=5,
        color=colors,
        edgecolor="black",
    )

    # Add significance stars
    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            y_pos = bars[i].get_height() + se_effects[i]
            ax.text(
                bars[i].get_x() + bars[i].get_width() / 2,
                y_pos,
                "*",
                ha="center",
                va="bottom",
                color="red",
                fontsize=16,
            )

    if plot_absolute:
        # No reference line needed for absolute plot
        pass
    else:
        # Reference line at 1 for normalized plot
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


# plt.show()


def analyse_multirun_experiment_normalized_control(
    dataset: pd.DataFrame,
    control_label: str,
    target_labels: list[str] | str,
    column: str,
    run_column: str = "run",
    p_val: float = 0.05,
    plot: bool = True,
    quiet: bool = False,
    plot_absolute: bool = False,
) -> dict:
    """
    Analyzes a multi-run experiment by normalizing to the control mean within each run.

    This method first calculates the mean of the control group for each independent run.
    It then subtracts this run-specific control mean from all data points (both control
    and target) within that same run. This centers the control data around zero for each run.

    After normalization, all the data is pooled together. A standard independent t-test is
    then performed to compare the (now zero-centered) control group against each target group.
    This method accounts for run-to-run shifts in the baseline control value.

    Args:
        dataset (pd.DataFrame): DataFrame with 'well', 'sample', `column`, and `run_column`.
        control_label (str): The label for the control group in the 'sample' column.
        target_labels (list[str] | str): A list of labels (or a single label) for the
                                         target groups in the 'sample' column.
        column (str): The name of the column containing the numeric values for comparison.
        run_column (str, optional): The name of the column identifying independent runs.
                                    Defaults to "run".
        p_val (float, optional): The p-value threshold for significance. Defaults to 0.05.
        plot (bool, optional): If True, plots the results. Defaults to True.
        quiet (bool, optional): If True, suppresses all print output. Defaults to False.
        plot_absolute (bool, optional): If True, plots the absolute values instead of the
                                        normalized ones. A warning is issued if used with
                                        multiple runs. Defaults to False.

    Returns:
        dict: A dictionary where keys are target labels and values are dictionaries
              containing the analysis results for that target.
    """
    # --- Input Validation ---
    if run_column not in dataset.columns:
        raise ValueError(f"Run column '{run_column}' not found in the dataset.")
    if "sample" not in dataset.columns:
        raise ValueError("The 'dataset' DataFrame must contain a 'sample' column.")
    if column not in dataset.columns:
        raise ValueError(f"The 'dataset' DataFrame must contain a '{column}' column.")

    if isinstance(target_labels, str):
        target_labels = [target_labels]

    runs = dataset[run_column].unique()
    if plot_absolute and len(runs) > 1 and not quiet:
        logging.warning(
            "Plotting absolute values across multiple runs can be misleading due to run-to-run variations."
        )

    # --- Data Normalization ---
    normalized_dfs = []
    normalized_col_name = f"{column}_normalized"

    if not quiet:
        print("--- Normalizing Data to Control Mean within Each Run ---")
    for run in runs:
        run_data = dataset[dataset[run_column] == run].copy()
        control_data_run = run_data[run_data["sample"] == control_label][
            column
        ].dropna()

        if control_data_run.empty:
            logging.warning(f"Skipping run '{run}': no control data found.")
            continue

        run_control_mean = control_data_run.mean()
        if not quiet:
            print(
                f"Run '{run}': Control mean = {run_control_mean:.4f}. Subtracting from all values in this run."
            )

        run_data[normalized_col_name] = run_data[column] - run_control_mean
        normalized_dfs.append(run_data)

    if not normalized_dfs:
        if not quiet:
            print("No data could be normalized. Aborting analysis.")
        return {}

    combined_normalized_df = pd.concat(normalized_dfs, ignore_index=True)

    # --- Pooled Analysis on Normalized Data ---
    if not quiet:
        print("\n--- Analyzing Pooled Normalized Data ---")
    analysis_results = {}

    # Get total entries per sample from the normalized data
    counts = {}
    control_normalized = combined_normalized_df[
        combined_normalized_df["sample"] == control_label
    ][normalized_col_name].dropna()
    counts[control_label] = len(control_normalized)

    if control_normalized.empty or len(control_normalized) < 2:
        if not quiet:
            print("Insufficient control data after normalization. Aborting analysis.")
        return {}

    for target_label in target_labels:
        target_normalized = combined_normalized_df[
            combined_normalized_df["sample"] == target_label
        ][normalized_col_name].dropna()
        counts[target_label] = len(target_normalized)

        if target_normalized.empty or len(target_normalized) < 2:
            logging.warning(
                f"Skipping analysis for target '{target_label}': insufficient data after normalization."
            )
            analysis_results[target_label] = None
            continue

        # Perform Welch's t-test
        t_stat, p_value = ttest_ind(
            target_normalized, control_normalized, equal_var=False
        )

        # Calculate effect size and standard error
        mean_effect = target_normalized.mean()  # Control mean is ~0
        se_effect = target_normalized.sem()

        results = {
            "mean_effect": mean_effect,
            "se_effect": se_effect,
            "p_value": p_value,
            "is_significant": p_value < p_val,
            "t_stat": t_stat,
            "n": counts[target_label],
        }
        analysis_results[target_label] = results
        if not quiet:
            print(
                f"Normalized analysis for '{target_label}': Mean Diff = {mean_effect:.4f}, SE = {se_effect:.4f}, p = {p_value:.4f}"
            )

    # Plot the results
    if plot and analysis_results:
        plot_normalized_analysis_results(
            analysis_results,
            column,
            counts,
            control_label,
            plot_absolute=plot_absolute,
            dataset=dataset,
        )

    return analysis_results


def analyse_multirun_experiment(
    dataset: pd.DataFrame,
    control_label: str,
    target_labels: list[str] | str,
    column: str,
    run_column: str = "run",
    p_val: float = 0.05,
    effect_mode: str = "difference",  # or "ratio"
) -> dict:
    """
    Analyzes a multi-run experiment using two methods:
    1. Inverse-variance weighting meta-analysis (primary method):
       Determines the overall effect by giving more weight to runs with higher precision.
       This is the most robust method for combining data from independent runs.
    2. Raw (pooled) analysis:
       Combines all data for a given sample across all runs and performs a simple
       t-test against the combined control data. This method ignores run-to-run
       variability and is provided for comparison.

    Args:
        dataset (pd.DataFrame): DataFrame with 'well', 'sample', `column`, and `run_column`.
        control_label (str): The label for the control group in the 'sample' column.
        target_labels (list[str] | str): A list of labels (or a single label) for the
                                         target groups in the 'sample' column.
        column (str): The name of the column containing the numeric values for comparison.
        run_column (str, optional): The name of the column identifying independent runs.
                                    Defaults to "run".
        p_val (float, optional): The p-value threshold for significance. Defaults to 0.05.
        effect_mode (str, optional): How to calculate the effect. "difference" (target - control)
                                     or "ratio" (target / control). Defaults to "difference".

    Returns:
        dict: A dictionary where keys are target labels and values are dictionaries
              containing the primary (weighted) analysis results for that target.
    """
    if run_column not in dataset.columns:
        raise ValueError(f"Run column '{run_column}' not found in the dataset.")
    if "sample" not in dataset.columns:
        raise ValueError("The 'dataset' DataFrame must contain a 'sample' column.")
    if column not in dataset.columns:
        raise ValueError(f"The 'dataset' DataFrame must contain a '{column}' column.")

    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Get total entries per sample
    counts = {}
    counts[control_label] = len(
        dataset[dataset["sample"] == control_label][column].dropna()
    )
    for label in target_labels:
        counts[label] = len(dataset[dataset["sample"] == label][column].dropna())

    # Print total entries per sample
    print("--- Total Processed Entries per Sample ---")
    print(f"  - {control_label} (Control): {counts[control_label]}")
    for label in target_labels:
        print(f"  - {label} (Target): {counts[label]}")
    print("-----------------------------------------")

    overall_results = {}

    for target_label in target_labels:
        print(
            f"--- Analyzing Target: {target_label} vs Control: {control_label} (Weighted) ---"
        )
        run_effects = []
        runs = dataset[run_column].unique()

        for run in runs:
            run_data = dataset[dataset[run_column] == run]

            control_data = run_data[run_data["sample"] == control_label][
                column
            ].dropna()
            target_data = run_data[run_data["sample"] == target_label][column].dropna()

            if (
                control_data.empty
                or target_data.empty
                or len(control_data) < 2
                or len(target_data) < 2
            ):
                logging.warning(
                    f"Skipping run '{run}' for target '{target_label}': insufficient data (need at least 2 points per group)."
                )
                continue

            mean_control, var_control, n_control = (
                control_data.mean(),
                control_data.var(ddof=1),
                len(control_data),
            )
            mean_target, var_target, n_target = (
                target_data.mean(),
                target_data.var(ddof=1),
                len(target_data),
            )

            var_mean_control = var_control / n_control
            var_mean_target = var_target / n_target

            if effect_mode == "difference":
                effect = mean_target - mean_control
                var_effect = var_mean_target + var_mean_control
            elif effect_mode == "ratio":
                if mean_control == 0:
                    logging.warning(
                        f"Skipping run '{run}' for ratio calculation: control mean is zero."
                    )
                    continue
                effect = mean_target / mean_control
                # Variance of a ratio using the delta method approximation
                var_effect = (effect**2) * (
                    var_mean_target / mean_target**2
                    + var_mean_control / mean_control**2
                )
            else:
                raise ValueError("effect_mode must be 'difference' or 'ratio'.")

            if var_effect == 0:
                logging.warning(
                    f"Skipping run '{run}' for target '{target_label}': zero variance in effect, cannot calculate weight."
                )
                continue

            run_effects.append(
                {
                    "run": run,
                    "effect": effect,
                    "variance": var_effect,
                    "weight": 1 / var_effect,
                    "mean_control": mean_control,
                    "mean_target": mean_target,
                }
            )

        if not run_effects:
            print(
                f"No valid run effects could be calculated for target '{target_label}'."
            )
            overall_results[target_label] = None
            continue

        effects_df = pd.DataFrame(run_effects)

        # Meta-analysis calculations
        weights = effects_df["weight"]
        effects = effects_df["effect"]

        weighted_mean_effect = np.sum(weights * effects) / np.sum(weights)
        var_weighted_mean = 1 / np.sum(weights)
        se_weighted_mean = np.sqrt(var_weighted_mean)

        # Z-test for significance
        popmean = 0 if effect_mode == "difference" else 1
        z_stat = (weighted_mean_effect - popmean) / se_weighted_mean
        p_value = 2 * norm.sf(np.abs(z_stat))  # Two-tailed p-value

        # Confidence interval
        confidence_level = 1 - p_val
        z_crit = norm.ppf((1 + confidence_level) / 2)
        ci_margin = z_crit * se_weighted_mean
        confidence_interval = (
            weighted_mean_effect - ci_margin,
            weighted_mean_effect + ci_margin,
        )

        results = {
            "per_run_effects": effects_df.set_index("run"),
            "mean_effect": weighted_mean_effect,
            "se_effect": se_weighted_mean,
            "p_value": p_value,
            "confidence_interval": confidence_interval,
            "is_significant": p_value < p_val,
            "z_stat": z_stat,
        }

        print(f"Weighted multi-run analysis for column '{column}':")
        print(
            f"  - Mean Effect ({effect_mode}): {results['mean_effect']:.4f} (SE: {results['se_effect']:.4f})"
        )
        print(
            f"  - Z-statistic: {results['z_stat']:.4f}, P-value: {results['p_value']:.4f}"
        )
        print(
            f"  - {confidence_level*100:.0f}% CI: ({results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f})"
        )
        print(f"  - Significant at p < {p_val}: {results['is_significant']}")

        overall_results[target_label] = results

    # Plot the weighted results
    if overall_results:
        plot_multirun_analysis_results(overall_results, effect_mode, column, counts)

    # --- Raw Analysis (All Runs Combined) ---
    print("\n--- Analyzing Raw Data (All Runs Combined) ---")
    raw_results = {}
    control_all_data = dataset[dataset["sample"] == control_label][column].dropna()

    if control_all_data.empty or len(control_all_data) < 2:
        print("Insufficient data for control group in raw analysis. Skipping.")
    else:
        mean_control_raw, var_control_raw, n_control_raw = (
            control_all_data.mean(),
            control_all_data.var(ddof=1),
            len(control_all_data),
        )
        var_mean_control_raw = var_control_raw / n_control_raw

        for target_label in target_labels:
            target_all_data = dataset[dataset["sample"] == target_label][
                column
            ].dropna()

            if target_all_data.empty or len(target_all_data) < 2:
                logging.warning(
                    f"Skipping raw analysis for target '{target_label}': insufficient data."
                )
                raw_results[target_label] = None
                continue

            mean_target_raw, var_target_raw, n_target_raw = (
                target_all_data.mean(),
                target_all_data.var(ddof=1),
                len(target_all_data),
            )
            var_mean_target_raw = var_target_raw / n_target_raw

            if effect_mode == "difference":
                effect_raw = mean_target_raw - mean_control_raw
                var_effect_raw = var_mean_target_raw + var_mean_control_raw
            elif effect_mode == "ratio":
                if mean_control_raw == 0:
                    logging.warning(
                        f"Skipping raw ratio calculation for target '{target_label}': control mean is zero."
                    )
                    raw_results[target_label] = None
                    continue
                effect_raw = mean_target_raw / mean_control_raw
                var_effect_raw = (effect_raw**2) * (
                    var_mean_target_raw / mean_target_raw**2
                    + var_mean_control_raw / mean_control_raw**2
                )

            se_effect_raw = np.sqrt(var_effect_raw)

            # Perform Welch's t-test for significance
            t_stat, p_value_raw = ttest_ind(
                target_all_data, control_all_data, equal_var=False
            )

            raw_results[target_label] = {
                "mean_effect": effect_raw,
                "se_effect": se_effect_raw,
                "p_value": p_value_raw,
                "is_significant": p_value_raw < p_val,
                "t_stat": t_stat,
            }
            print(
                f"Raw analysis for '{target_label}': Effect = {effect_raw:.4f}, SE = {se_effect_raw:.4f}, p = {p_value_raw:.4f}"
            )

    # Plot the raw results
    if raw_results:
        plot_raw_analysis_results(
            raw_results, effect_mode, column, counts, control_label
        )

    return overall_results


def combine_runs_to_dataframe(runs_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combines pre-loaded DataFrames from multiple experimental runs into a single DataFrame.

    This function iterates through a dictionary of DataFrames, adds a 'run'
    column to each, and then concatenates the results. It assumes the 'sample'
    column already exists in the input DataFrames.

    Args:
        runs_data (dict[str, pd.DataFrame]): A dictionary mapping a run name to its
                                             pre-loaded collected_data DataFrame.
                                                                                                                                                         The DataFrame should already contain a 'sample' column.
                                             Example: {'run1': df1, 'run2': df2}

    Returns:
        pd.DataFrame: A single DataFrame containing the combined and labeled data from all runs.
    """
    all_data = []

    for run_name, collected_data in runs_data.items():
        # Make a copy to avoid modifying the original DataFrame
        data_copy = collected_data.copy()

        # Add run identifier
        data_copy["run"] = run_name

        if "sample" not in data_copy.columns:
            logging.warning(f"No 'sample' column found for run '{run_name}'.")

        all_data.append(data_copy)

    if not all_data:
        print("No data was provided to combine.")
        return pd.DataFrame()

    # Concatenate all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)

    # Reorder columns to have identifiers first
    cols_to_front = ["run", "sample", "well", "cell_id", "global_id"]
    existing_cols = [col for col in cols_to_front if col in combined_df.columns]
    other_cols = [col for col in combined_df.columns if col not in existing_cols]
    combined_df = combined_df[existing_cols + other_cols]

    return combined_df


def plot_mixed_model_results(
    results: dict, column: str, counts: dict, control_label: str
):
    """
    Plots the results of the mixed-effects model analysis.
    """
    target_labels = [label for label, res in results.items() if res is not None]
    if not target_labels:
        print("No valid mixed-model results to plot.")
        return

    mean_effects = [results[label]["mean_effect"] for label in target_labels]
    se_effects = [results[label]["se_effect"] for label in target_labels]
    p_values = [results[label]["p_value"] for label in target_labels]

    plot_labels = [
        f"{label}\n(n={counts.get(label, 'N/A')})" for label in target_labels
    ]

    fig, ax = plt.subplots(dpi=150, figsize=(max(6, len(target_labels) * 0.8), 5))
    bars = ax.bar(
        plot_labels,
        mean_effects,
        yerr=se_effects,
        capsize=5,
        color="gold",
        edgecolor="black",
    )

    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            y_pos = bars[i].get_height() + se_effects[i]
            ax.text(
                bars[i].get_x() + bars[i].get_width() / 2,
                y_pos,
                "*",
                ha="center",
                va="bottom",
                color="red",
                fontsize=16,
            )

    ylabel = f"Mean Difference in {column} (Mixed-Effects Model)"
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Mixed-Effects Model Analysis (vs. Control '{control_label}')")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def analyse_multirun_experiment_mixed_model(
    dataset: pd.DataFrame,
    control_label: str,
    target_labels: list[str] | str,
    column: str,
    run_column: str = "run",
    p_val: float = 0.05,
) -> dict:
    """
    Analyzes a multi-run experiment by normalizing against each control well and using a
    Linear Mixed-Effects Model (LMM) to account for correlated data.

    This method expands the dataset. For each run:
    1. It identifies all control wells.
    2. For every data point (in any well, target or control), it creates new data points
       by subtracting the mean of EACH control well from it.
    3. This explicitly accounts for the variability between control wells within a run.

    The resulting dataset has correlated data (multiple new points from one original point).
    A Linear Mixed-Effects Model is used to analyze this, with the original cell ID treated
    as a random effect to handle the non-independence.

    Args:
        dataset (pd.DataFrame): DataFrame with 'well', 'sample', `column`, and `run_column`.
        control_label (str): The label for the control group.
        target_labels (list[str] | str): Labels for the target groups.
        column (str): The numeric column to analyze.
        run_column (str, optional): Column identifying different runs. Defaults to "run".
        p_val (float, optional): Significance threshold. Defaults to 0.05.

    Returns:
        dict: A dictionary of analysis results for each target label.
    """
    if run_column not in dataset.columns:
        raise ValueError(f"Run column '{run_column}' not found in the dataset.")
    if "sample" not in dataset.columns:
        raise ValueError("The 'dataset' DataFrame must contain a 'sample' column.")
    if column not in dataset.columns:
        raise ValueError(f"The 'dataset' DataFrame must contain a '{column}' column.")

    if isinstance(target_labels, str):
        target_labels = [target_labels]

    print("--- Creating Expanded Dataset for Mixed-Effects Model ---")
    expanded_data = []
    all_samples = [control_label] + target_labels

    # Get original counts
    original_counts = {
        label: len(dataset[dataset["sample"] == label][column].dropna())
        for label in all_samples
    }

    for run in dataset[run_column].unique():
        run_data = dataset[dataset[run_column] == run]

        control_wells_in_run = run_data[run_data["sample"] == control_label][
            "well"
        ].unique()
        if len(control_wells_in_run) == 0:
            logging.warning(f"Skipping run '{run}': No control wells found.")
            continue

        control_well_means = {
            cw: run_data[run_data["well"] == cw][column].mean()
            for cw in control_wells_in_run
        }

        for sample_label in all_samples:
            sample_wells_in_run = run_data[run_data["sample"] == sample_label][
                "well"
            ].unique()
            for well in sample_wells_in_run:
                well_data = run_data[run_data["well"] == well]
                for _, row in well_data.iterrows():
                    if pd.notna(row[column]):
                        for (
                            control_well_ref,
                            control_mean,
                        ) in control_well_means.items():
                            # For control data, avoid subtracting its own mean from itself
                            if well == control_well_ref:
                                continue

                            expanded_data.append(
                                {
                                    "normalized_value": row[column] - control_mean,
                                    "sample": sample_label,
                                    "grouping_id": row[
                                        "global_id"
                                    ],  # Use a unique ID for each cell
                                }
                            )

    if not expanded_data:
        print("Could not generate data for mixed-effects model. Aborting.")
        return {}

    expanded_df = pd.DataFrame(expanded_data)

    # The control group is now centered around 0. We test if the target groups are different from 0.
    print("\n--- Fitting Linear Mixed-Effects Model ---")

    # Ensure 'sample' is a categorical type with the control as the reference level
    expanded_df["sample"] = pd.Categorical(
        expanded_df["sample"], categories=[control_label] + target_labels, ordered=True
    )

    analysis_results = {}
    for target in target_labels:
        print(f"--- Analyzing Target: {target} ---")
        # We model the difference between the target and the control (the intercept)
        model_df = expanded_df[expanded_df["sample"].isin([control_label, target])]

        try:
            # model formula: the normalized value depends on the sample, with correlation by cell
            md = smf.mixedlm(
                "normalized_value ~ sample", model_df, groups=model_df["grouping_id"]
            )
            mdf = md.fit()

            # The p-value for the target sample coefficient
            p_value = mdf.pvalues[f"sample[T.{target}]"]
            mean_effect = mdf.fe_params[f"sample[T.{target}]"]
            se_effect = mdf.bse[f"sample[T.{target}]"]

            analysis_results[target] = {
                "mean_effect": mean_effect,
                "se_effect": se_effect,
                "p_value": p_value,
                "is_significant": p_value < p_val,
                "summary": mdf.summary(),
            }
            print(
                f"Mixed-effects model for '{target}': Mean Diff = {mean_effect:.4f}, SE = {se_effect:.4f}, p = {p_value:.4f}"
            )

        except Exception as e:
            print(
                f"Could not fit mixed-effects model for target '{target}'. Error: {e}"
            )
            analysis_results[target] = None

    if analysis_results:
        plot_mixed_model_results(
            analysis_results, column, original_counts, control_label
        )

    return analysis_results


def plot_pairwise_differences_vs_time(
    df,
    control_sample,
    sample_names=None,
    window_seconds=7200,
    plot=True,
    plot_control_evolution=True,
):
    """
    For each selected sample, computes all pairwise differences in sigma_value and kappa_value
    between control and sample datapoints, along with the time difference (sample_time - control_time).
    Plots rolling mean and std of these differences vs time difference (>0 only), using a time-based window.
    Overlays a line for each sample.
    If plot_control_evolution is True, adds a secondary y-axis showing the smoothed evolution of the control sample itself.
    Returns a DataFrame with columns: [sample, control_time, sample_time, time_diff, sigma_diff, kappa_diff]
    """

    # Select samples
    if sample_names is None:
        sample_names = [s for s in df["sample"].unique() if s != control_sample]
    results = []
    for sample in sample_names:
        control_df = df[df["sample"] == control_sample]
        sample_df = df[df["sample"] == sample]
        # Drop rows with NaN in relevant columns
        control_df = control_df.dropna(subset=["time", "sigma_value", "kappa_value"])
        sample_df = sample_df.dropna(subset=["time", "sigma_value", "kappa_value"])
        control_arr = control_df[["time", "sigma_value", "kappa_value"]].to_numpy()
        sample_arr = sample_df[["time", "sigma_value", "kappa_value"]].to_numpy()
        for s_row in sample_arr:
            s_time, s_sigma, s_kappa = s_row
            for c_row in control_arr:
                c_time, c_sigma, c_kappa = c_row
                # Skip if any value is nan
                if (
                    np.isnan(s_time)
                    or np.isnan(s_sigma)
                    or np.isnan(s_kappa)
                    or np.isnan(c_time)
                    or np.isnan(c_sigma)
                    or np.isnan(c_kappa)
                ):
                    continue
                time_diff = s_time - c_time
                if time_diff > 0:
                    results.append(
                        {
                            "sample": sample,
                            "control_time": c_time,
                            "sample_time": s_time,
                            "time_diff": time_diff,
                            "sigma_diff": s_sigma - c_sigma,
                            "kappa_diff": s_kappa - c_kappa,
                        }
                    )
    diff_df = pd.DataFrame(results)
    if diff_df.empty:
        print("No valid pairs found.")
        return diff_df

    # Plotting
    def rolling_time_window(x, y, window):
        # x: sorted time_diff, y: corresponding values
        mean = np.zeros_like(x)
        std = np.zeros_like(x)
        for i in range(len(x)):
            # Find indices within window_seconds before x[i]
            left = np.searchsorted(x, x[i] - window, side="left")
            window_vals = y[left : i + 1]
            # Ignore nan values in window_vals
            window_vals = window_vals[~np.isnan(window_vals)]
            if len(window_vals) > 0:
                mean[i] = np.mean(window_vals)
                std[i] = np.std(window_vals)
            else:
                mean[i] = np.nan
                std[i] = np.nan
        return mean, std

    if plot:
        control_df_for_plot = df[df["sample"] == control_sample].sort_values("time")

        for value, label in [
            ("sigma_diff", "Sigma Value Difference"),
            ("kappa_diff", "Kappa Value Difference"),
        ]:
            fig, ax1 = plt.subplots(dpi=200)

            # Plot differences on primary axis
            for sample in sample_names:
                sample_df = diff_df[diff_df["sample"] == sample].sort_values(
                    "time_diff"
                )
                x = sample_df["time_diff"].to_numpy()
                y = sample_df[value].to_numpy()
                if len(x) > 0:
                    roll_mean, roll_std = rolling_time_window(x, y, window_seconds)
                    start_idx = np.searchsorted(x, x[0] + window_seconds, side="left")
                    if start_idx < len(x):
                        (line,) = ax1.plot(
                            x[start_idx:], roll_mean[start_idx:], label=f"{sample} mean"
                        )
                        ax1.fill_between(
                            x[start_idx:],
                            roll_mean[start_idx:] - roll_std[start_idx:],
                            roll_mean[start_idx:] + roll_std[start_idx:],
                            alpha=0.2,
                            color=line.get_color(),
                        )

            ax1.set_xlabel("Time Difference (s)")
            ax1.set_ylabel(label, color="black")
            ax1.tick_params(axis="y", labelcolor="black")
            ax1.axhline(0, color="grey", linestyle="--", linewidth=0.8)

            # Plot control evolution on secondary axis
            if plot_control_evolution and not control_df_for_plot.empty:
                ax2 = ax1.twinx()
                control_value_col = "sigma_value" if "sigma" in value else "kappa_value"
                control_label_suffix = "Sigma" if "sigma" in value else "Kappa"

                x_control = control_df_for_plot["time"].to_numpy()
                y_control = control_df_for_plot[control_value_col].to_numpy()

                if len(x_control) > 0:
                    # We need to adjust the rolling window function to work with absolute time
                    # The existing function calculates window from the past, which is fine.
                    # We also need to map time_diff to absolute time for the x-axis.
                    # For simplicity, let's align the start time of control with time_diff=0
                    time_offset = x_control[0]
                    x_control_adjusted = x_control - time_offset

                    roll_mean_control, roll_std_control = rolling_time_window(
                        x_control_adjusted, y_control, window_seconds
                    )
                    start_idx_control = np.searchsorted(
                        x_control_adjusted,
                        x_control_adjusted[0] + window_seconds,
                        side="left",
                    )

                    if start_idx_control < len(x_control_adjusted):
                        ax2.plot(
                            x_control_adjusted[start_idx_control:],
                            roll_mean_control[start_idx_control:],
                            label=f"{control_sample} (control)",
                            color="red",
                            linestyle=":",
                        )
                        ax2.fill_between(
                            x_control_adjusted[start_idx_control:],
                            roll_mean_control[start_idx_control:]
                            - roll_std_control[start_idx_control:],
                            roll_mean_control[start_idx_control:]
                            + roll_std_control[start_idx_control:],
                            alpha=0.1,
                            color="red",
                        )

                ax2.set_ylabel(f"Control Value ({control_label_suffix})", color="red")
                ax2.tick_params(axis="y", labelcolor="red")

            fig.suptitle(f"{label} vs Time Difference ({control_sample} vs samples)")
            fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
            fig.tight_layout()
            plt.subplots_adjust(top=0.92)  # Adjust for suptitle

    return diff_df


def plot_rolling_gradient(
    collected_data,
    columns=["sigma_value", "kappa_value"],
    window_hours=2,
    plot=True,
    samples_to_plot=None,
    min_points_in_window=5,
):
    """
    Calculates and plots the rolling gradient of specified columns over time for each sample.

    For each data point, it looks back over a defined time window, performs a
    linear regression (slope calculation) on the data within that window, and
    plots this gradient against time. This helps visualize the rate of change
    of a value over time. The time axis is normalized to the start of the experiment.

    Args:
        collected_data (pd.DataFrame): DataFrame containing the experimental data.
                                       Must include 'time' and 'sample' columns, and
                                       the columns specified for analysis.
        columns (list, optional): A list of column names for which to calculate the
                                  rolling gradient. Defaults to ['sigma_value', 'kappa_value'].
        window_hours (int, optional): The size of the rolling window in hours.
                                      Defaults to 2.
        plot (bool, optional): If True, generates and displays the plots.
                               Defaults to True.
        samples_to_plot (list, optional): A list of sample names to plot. If None, all
                                          samples will be plotted. Defaults to None.
        min_points_in_window (int, optional): The minimum number of data points required
                                              within a rolling window to calculate a gradient.
                                              Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated rolling gradients for each sample,
                      with columns for time, sample, and the gradient of each analyzed column
                      (e.g., 'sigma_value_gradient').
    """
    df = collected_data.copy()

    # --- Pre-processing and Validation ---
    if "time" not in df.columns:
        raise ValueError("Input DataFrame must have a 'time' column.")
    if "sample" not in df.columns:
        raise ValueError("Input DataFrame must have a 'sample' column for grouping.")

    # Ensure 'time' is numeric, then normalize it to the start of the experiment
    df["time"] = pd.to_numeric(df["time"])
    min_time = df["time"].min()
    df["time"] = df["time"] - min_time
    df = df.sort_values("time")

    window_seconds = window_hours * 3600
    all_samples_results = []

    # --- Helper function for slope calculation ---
    def get_slope(x, y, min_points):
        valid_indices = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[valid_indices], y[valid_indices]
        if len(x) < min_points:
            return np.nan
        return np.polyfit(x, y, 1)[0]

    # --- Process each sample ---
    samples = df["sample"].unique()
    for sample in samples:
        sample_df = df[df["sample"] == sample]

        sample_results_df = pd.DataFrame(
            {"time": sorted(sample_df["time"].dropna().unique())}
        )
        sample_results_df["sample"] = sample

        for col in columns:
            temp_df = sample_df[["time", col]].dropna()
            if temp_df.empty:
                sample_results_df[f"{col}_gradient"] = np.nan
                continue

            times = temp_df["time"].to_numpy()
            values = temp_df[col].to_numpy()

            gradients = []
            for t in sample_results_df["time"]:
                window_start_time = t - window_seconds
                mask = (times >= window_start_time) & (times <= t)
                window_times = times[mask]
                window_values = values[mask]
                slope = get_slope(window_times, window_values, min_points_in_window)
                gradients.append(slope)

            sample_results_df[f"{col}_gradient"] = gradients

        all_samples_results.append(sample_results_df)

    if not all_samples_results:
        print("No data to process.")
        return pd.DataFrame()

    final_results_df = pd.concat(all_samples_results, ignore_index=True)

    # --- Plotting ---
    if plot:
        plotting_samples = samples
        if samples_to_plot is not None:
            plotting_samples = [s for s in samples if s in samples_to_plot]

        for col in columns:
            grad_col_name = f"{col}_gradient"
            if grad_col_name not in final_results_df.columns:
                continue

            plt.figure(dpi=150)

            for sample in plotting_samples:
                sample_plot_df = final_results_df[
                    final_results_df["sample"] == sample
                ].copy()

                # Filter out NaN gradients for plotting
                plot_data = sample_plot_df.dropna(subset=[grad_col_name])

                if not plot_data.empty:
                    time_hours = plot_data["time"] / 3600
                    gradient_per_hour = plot_data[grad_col_name] * 3600

                    # Plot individual gradient points
                    (line,) = plt.plot(
                        time_hours,
                        gradient_per_hour,
                        "o",
                        markersize=2,
                        alpha=0.4,
                        label=f"{sample} points",
                    )

                    # Plot a smoothed line (e.g., rolling mean of the calculated gradients)
                    # This provides a clearer trend line over the noisy points
                    smoothed_grad = gradient_per_hour.rolling(
                        window=max(1, int(len(gradient_per_hour) / 10)), center=True
                    ).mean()
                    # plt.plot(time_hours, smoothed_grad, color=line.get_color(), label=f'{sample} smoothed')

            plt.xlabel("Time (hours from start)")
            plt.ylabel(f"Gradient of {col} (units/hour)")
            plt.title(
                f"Rolling Gradient of {col} (Window: {window_hours} hours, Min Points: {min_points_in_window})"
            )
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.axhline(0, color="grey", linestyle="--")
            plt.legend()
            plt.tight_layout()
            # plt.show()

    return final_results_df
