import numpy as np
import scipy
from scipy.interpolate import interp2d, interp1d, RegularGridInterpolator
from scipy.signal import correlate, correlation_lags, windows
import itertools
import cv2
import matplotlib.pyplot as plt
from time import time
from glob import glob
from multiprocessing import Pool
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
import pandas as pd
import numpy as np
from scipy.stats import linregress
import warnings
from flickering.utils.process_multiwell import *
from flickering.utils.process_multiwell import (
    get_repeat_from_global_id,
    get_group_from_global_id,
)

warnings.filterwarnings("ignore")
from PIL import Image


# chatgpt
def combine_images(jpeg_files, max_width):
    # Open all images
    images = []
    for file in jpeg_files:
        try:
            images.append(Image.open(file))
        except:
            print(f"Failed to open preview {file}")
        #    images.append(None)
    # Calculate the number of rows needed
    num_images = len(images)
    if num_images == 0:
        return False
    num_rows = (num_images - 1) // (max_width // min(img.width for img in images)) + 1

    # Calculate total height for the combined image
    img_height = max(img.height for img in images)
    combined_width = min(max_width, max(img.width for img in images)) * min(
        num_images, max_width // min(img.width for img in images)
    )
    # chatgpt missed this...
    total_height = num_rows * img_height

    # Create a blank image with the calculated dimensions
    combined_image = Image.new("RGB", (combined_width, total_height), (255, 255, 255))

    # Paste images onto the combined image
    y_offset = 0
    x_offset = 0
    last_img = None
    for img in images:
        if img is not None:
            combined_image.paste(img, (x_offset, y_offset))
            last_img = img
        if last_img is not None:
            x_offset += last_img.width
            if x_offset >= max_width or x_offset >= combined_width:
                y_offset += last_img.height
                x_offset = 0

    # Convert the combined image to a numpy array
    combined_array = np.array(combined_image)

    return combined_array


def plot_individual_cells(data, ids, cell_log):
    # data = np.array(data)
    base_time = float(data["time"].min())
    plt.rcParams["figure.dpi"] = 200
    for cell_id in tqdm(ids):
        cell_data = data[data["cell_id"] == cell_id]
        images = []
        times = []
        for row in cell_log:
            if row["cell_id"] == cell_id:
                images.append(row["preview_file"])
                times.append(float(row["start_time"]))
        images_sorted = [x for _, x in sorted(zip(times, images))]
        # print(cell_id)
        full_image = combine_images(images_sorted, 2000)
        if full_image is not False:
            # cell_data = cell_data[:,:-3].astype(float)
            plt.figure(dpi=200)
            plt.title(f"Cell {cell_id}")
            plt.imshow(full_image)
        plt.figure()
        plt.title(f"Cell {cell_id}, Tension")
        plt.errorbar(
            (cell_data["time"] - base_time) / 3600,
            cell_data["sigma_value"],
            yerr=cell_data["sigma_error"],
            fmt="o",
        )
        plt.ylim((0, 130))
        plt.ylabel("Tension / 10^{-7} Nm")
        plt.xlabel("Time/h")
        plt.figure()
        kT = 4.07291455  # e-21 J
        plt.title(f"Cell {cell_id}, Bending")
        plt.ylabel("Bending modulus / kT")
        plt.xlabel("Time/h")
        plt.errorbar(
            (cell_data["time"] - base_time) / 3600,
            cell_data["kappa_value"] / kT,
            yerr=cell_data["kappa_error"] / kT,
            fmt="o",
        )
        # plt.ylim((-10,50))
        plt.figure()
        plt.title(f"Cell {cell_id}, R2, valid")
        plt.ylabel("R^2, valid %")
        plt.xlabel("Time/h")
        plt.errorbar(
            (cell_data["time"] - base_time) / 3600,
            cell_data["valid_p"],
            fmt="o",
            label="Valid contours",
        )
        plt.errorbar(
            (cell_data["time"] - base_time) / 3600,
            cell_data["r2"],
            fmt="o",
            label="R2 fit",
        )
        plt.legend()

        plt.figure()
        plt.title(f"Cell {cell_id}, radius")
        plt.ylabel("Radius/nm")
        plt.xlabel("Time/h")
        plt.errorbar(
            (cell_data["time"] - base_time) / 3600,
            cell_data["radius_value"],
            fmt="o",
            label="Cell radius",
        )
        plt.legend()


# np.array(data)#[np.array(data)[:,0]==cell_id,:]


def normalise_values(collected_data):
    # obtain base values for normalised results
    base_values = {}
    nans = 0

    # range for intial normalised values, first value in range used
    min_repeat_n = 0
    max_repeat_n = 1

    for r_id in range(min_repeat_n, max_repeat_n):
        for i, row in collected_data.iterrows():
            if np.isnan(float(row["sigma_value"])):
                # nans+=1
                continue
            if row["repeat"] == r_id and row["cell_id"] not in base_values:
                base_values[row["cell_id"]] = row

    normalised_data = []
    # del collected_data["datetime"]
    # TODO: columns might be inconsistent?
    for index, row in collected_data.iterrows():
        if row["cell_id"] not in base_values or row["repeat"] < 0:
            continue
        base_val = base_values[row["cell_id"]]
        non_num = [
            "well",
            "sample",
            "sample_run",
            "run",
            "cell_id",
            "global_id",
            "radius",
            "fit_status",
        ]
        divided = row.drop(non_num).astype(float) / base_val.drop(non_num).astype(float)
        # base_val =  pd.DataFrame(data=[base_val], columns=columns)
        # 'well', 'cell_id', 'global_id'
        normed_row = [] + list(
            row[
                [
                    "well",
                    "sample",
                    "run",
                    "cell_id",
                    "sample_run",
                    "global_id",
                    "time",
                    "pfs_offset",
                    "radius",
                    "valid_p",
                    "fit_status",
                ]
            ]
        )
        normed_row.append(divided["sigma_value"])
        normed_row.append(
            float(
                divided["sigma_value"]
                * np.sqrt(
                    (base_val["sigma_error"] / base_val["sigma_value"]) ** 2
                    + (row["sigma_error"] / row["sigma_value"]) ** 2
                )
            )
        )
        # ignore crazy uncertainty
        if normed_row[-1] > 3:
            continue
        normed_row.append(divided["kappa_value"])
        normed_row.append(
            float(
                divided["kappa_value"]
                * np.sqrt(
                    (base_val["kappa_error"] / base_val["kappa_value"]) ** 2
                    + (row["kappa_error"] / row["kappa_value"]) ** 2
                )
            )
        )
        if normed_row[-1] > 3:
            continue

        normed_row.append(divided["r2"])
        normed_row.append(divided["radius_value"])
        normed_row.append(row["fit_p"])
        normed_row.append(row["fit_cross_rate"])
        normed_row.append(row["repeat"])
        if "contour_variance" in row:
            normed_row.append(divided["contour_variance"])
            normed_row.append(divided["contour_variance_butter"])

        # normed_row.append(
        #    float(divided["radius_value"]*np.sqrt(
        #        (base_val["kappa_error"]/base_val["kappa_value"])**2+(row["kappa_error"]/row["kappa_value"])**2
        #    ))
        # )
        normalised_data.append(normed_row)

    return pd.DataFrame(data=normalised_data, columns=collected_data.columns)


# normalised_data_df = normalise_values(collected_data)


def plot_normalised_results(
    normalised_data_df,
    column,
    label="Normalised tension/AU",
    rolling="2H",
    merge_sample=False,
):
    if "repeat" not in normalised_data_df.columns:
        normalised_data_df["repeat"] = list(
            map(get_repeat_from_global_id, normalised_data_df["global_id"])
        )
        normalised_data_df["group"] = list(
            map(get_group_from_global_id, normalised_data_df["global_id"])
        )

    start_time = np.min(normalised_data_df["time"])
    roll_time = rolling
    if merge_sample:
        normalised_data_df = normalised_data_df.copy()
        normalised_data_df["well"] = normalised_data_df["sample"]

    wells = normalised_data_df["well"].unique().tolist()

    # Check if we have groups
    if "group" not in normalised_data_df.columns:
        normalised_data_df["group"] = 0

    plt.figure(dpi=200)
    # print(wells)
    colors = [
        "green",
        "blue",
        "red",
        "yellow",
        "magenta",
        "gold",
        "darkred",
        "aqua",
        "darkolivegreen",
        "indigo",
        "slategray",
        "gainsboro",
        "sienna",
        "darkslategray",
    ]  # TODO
    i = 0
    for well in wells:
        df_well = normalised_data_df[normalised_data_df["well"] == well]
        groups = sorted(
            df_well["group"].unique(),
            key=lambda x: (
                (0, float(x)) if isinstance(x, (int, float, np.number)) else (1, str(x))
            ),
        )

        for g in groups:
            normalised_data_df_well = df_well[df_well["group"] == g]

            normalised_data_df_well["datetime"] = pd.to_datetime(
                normalised_data_df_well["time"], unit="s"
            )
            rolled_mean = (
                normalised_data_df_well.select_dtypes(exclude=object)
                .sort_values("datetime")
                .rolling(roll_time, min_periods=3, on="datetime")
                .apply(np.nanmean)
            )
            rolled_std = (
                normalised_data_df_well.select_dtypes(exclude=object)
                .sort_values("datetime")
                .rolling(roll_time, min_periods=3, on="datetime")
                .apply(np.nanstd)
            )
            rolled_counts = (
                normalised_data_df_well.select_dtypes(exclude=object)
                .sort_values("datetime")
                .rolling(roll_time, min_periods=3, on="datetime")
                .apply(np.count_nonzero)
            )

            # rolled_mean["sigma_value"]/=rolled_counts["sigma_value"]
            grouped_mean = (
                normalised_data_df_well.select_dtypes(exclude=object)
                .groupby("repeat")
                .mean()
            )
            grouped_std = (
                normalised_data_df_well.select_dtypes(exclude=object)
                .groupby("repeat")
                .std()
            )

            # plt.errorbar(normalised_data_df["time"]-start_time, normalised_data_df["sigma_value"], yerr=normalised_data_df["sigma_error"], fmt="+", capsize=2, c="orange", label="Recordings")

            # plt.errorbar(normalised_data[:,2], normalised_data[:,4], yerr=normalised_data[:,5].astype(float), fmt="+", capsize=2)

            # plt.errorbar(grouped_mean["time"]-start_time, grouped_mean["sigma_value"],yerr=grouped_std["sigma_value"], capsize=2, fmt="o", label="Repeat mean",c="green")

            label_suffix = "" if len(groups) <= 1 and g == 0 else f" G{g}"

            # Use color index, cycle if needed
            c = colors[i % len(colors)]

            plt.plot(
                (rolled_mean["time"] - start_time).to_numpy(),
                rolled_mean[column].to_numpy(),
                c=c,
                label=f"{well}{label_suffix}: 2 hour rolling mean",
            )
            stds = rolled_std[column]
            # mean_errs = rolled_std[column]/np.sqrt(rolled_counts[column]-1)
            use_err = stds
            plt.fill_between(
                (rolled_mean["time"] - start_time).to_numpy(),
                (rolled_mean[column] - use_err).to_numpy(),
                (rolled_mean[column] + use_err).to_numpy(),
                color=c,
                alpha=0.3,
            )
            plt.xlabel("Time/s")
            plt.errorbar(
                (grouped_mean["time"] - start_time).to_numpy(),
                grouped_mean[column].to_numpy(),
                yerr=grouped_std[column].to_numpy(),
                capsize=2,
                fmt="+",
                c=c,
            )
            # plt.plot((rolled_mean["time"]-start_time).to_numpy(), rolled_counts[column].to_numpy(), label=f"{well}: Repeat mean", c=colors[i])

            i += 1
        plt.ylabel(label)

    plt.legend()
    # plt.xlim((0,200000))
    # plt.xlim((0,30000))


def plot_mean_counts(normalised_data_df, column, label="Normalised tension/AU"):
    start_time = np.min(normalised_data_df["time"])
    roll_time = "6H"

    wells = normalised_data_df["well"].unique().tolist()

    plt.figure(dpi=200)
    # print(wells)
    colors = [
        "green",
        "blue",
        "red",
        "yellow",
        "magenta",
        "gold",
        "darkred",
        "aqua",
        "darkolivegreen",
        "indigo",
        "slategray",
        "gainsboro",
        "sienna",
        "darkslategray",
    ]  # TODO
    i = 0
    for well in wells:
        normalised_data_df_well = normalised_data_df[normalised_data_df["well"] == well]
        normalised_data_df_well["datetime"] = pd.to_datetime(
            normalised_data_df_well["time"], unit="s"
        )
        rolled_mean = (
            normalised_data_df_well.select_dtypes(exclude=object)
            .sort_values("datetime")
            .rolling(roll_time, min_periods=3, on="datetime")
            .apply(np.nanmean)
        )
        # rolled_std = normalised_data_df_well.select_dtypes(exclude=object).sort_values("datetime").rolling(roll_time, min_periods=3, on="datetime").apply(np.nanstd)
        rolled_counts = (
            normalised_data_df_well.select_dtypes(exclude=object)
            .sort_values("datetime")
            .rolling(roll_time, min_periods=3, on="datetime")
            .apply(np.count_nonzero)
        )

        # rolled_mean["sigma_value"]/=rolled_counts["sigma_value"]
        grouped_mean = (
            normalised_data_df_well.select_dtypes(exclude=object)
            .groupby("repeat")
            .mean()
        )
        grouped_std = (
            normalised_data_df_well.select_dtypes(exclude=object)
            .groupby("repeat")
            .std()
        )

        # plt.errorbar(normalised_data_df["time"]-start_time, normalised_data_df["sigma_value"], yerr=normalised_data_df["sigma_error"], fmt="+", capsize=2, c="orange", label="Recordings")

        # plt.errorbar(normalised_data[:,2], normalised_data[:,4], yerr=normalised_data[:,5].astype(float), fmt="+", capsize=2)

        # plt.errorbar(grouped_mean["time"]-start_time, grouped_mean["sigma_value"],yerr=grouped_std["sigma_value"], capsize=2, fmt="o", label="Repeat mean",c="green")

        plt.plot(
            (rolled_mean["time"] - start_time).to_numpy(),
            rolled_counts[column].to_numpy(),
            c=colors[i],
            label=f"{well}: 2 hour rolling count",
        )
        plt.xlabel("Time/s")
        # plt.errorbar((grouped_mean["time"]-start_time).to_numpy(), grouped_mean[column].to_numpy(),yerr=grouped_std[column].to_numpy(), capsize=2, fmt="+", label=f"{well}: Repeat mean", c=colors[i])
        i += 1
        plt.ylabel(label)

    plt.legend()
    # plt.xlim((0,200000))
    # plt.xlim((0,30000))


# visualise cell status including tracking and fit results
def visualise_well_results(cell_log, well="F3"):
    cmap = {
        "No contour file": [255, 150, 0],
        "Autofocus fail": [0, 255, 255],  # [255,80,0],
        "Poor fit": [100, 0, 255],
        "Fit error": [0, 0, 255],
        "Contour invalid": [255, 255, 0],
        "OK": [0, 255, 0],
        "Not found": [255, 0, 0],
    }

    relevant_cell_ids = list()
    relevant_repeats = set()
    for cell in cell_log:
        if cell["well"] == well:
            if cell["cell_id"] not in relevant_cell_ids:
                relevant_cell_ids.append(cell["cell_id"])
            relevant_repeats.add(get_repeat_from_global_id(cell["global_id"]))

    # relevant_cell_ids = list(relevant_cell_ids)
    relevant_repeats = list(relevant_repeats)

    # explicit RGB
    image = np.zeros((len(relevant_cell_ids), len(relevant_repeats), 3))
    image += cmap["Not found"]

    for cell in cell_log:
        value = [255, 255, 255]  # cmap["Not found"] #not found
        if cell["well"] == well:
            if "result" not in cell or cell["result"] == "preprocesor-fail":
                value = cmap["Autofocus fail"]  # red-orange
            else:  # success doesn't have result entry, this is a bug
                if not "contour_file" in cell:
                    value = cmap["No contour file"]  # orange
                elif "fit_status" in cell:
                    match cell["fit_status"]:
                        case "success":
                            value = cmap["OK"]
                        case "r2-fail":
                            value = cmap["Poor fit"]  # blue-purple
                        case "contour-invalid":
                            value = cmap["Contour invalid"]
                        case "error":
                            value = cmap["Fit error"]
                        case "filter_reject":  # different color? there is a lot already
                            value = cmap["Poor fit"]
                else:
                    value = cmap["Fit error"]

            repeat_i = relevant_repeats.index(
                get_repeat_from_global_id(cell["global_id"])
            )
            cell_id_i = relevant_cell_ids.index(cell["cell_id"])
            image[cell_id_i, repeat_i] = np.array(value, dtype=int)

    plt.figure(dpi=200)
    plt.imshow(image, interpolation="nearest", aspect="auto")
    plt.title(f"Cell finding results {well}")

    plt.figure(dpi=200)
    label_img = np.reshape(list(cmap.values()), (-1, 1, 3))
    plt.imshow(label_img)
    plt.yticks(np.arange(len(cmap.values())), list(cmap.keys()))
    plt.xticks([])
    plt.xlabel("Repeat number")
    plt.ylabel("Cell number")


def plot_focus_results(processed_cell_log):
    focus_results = []
    wells = set()
    for row in processed_cell_log:
        repeat_n = get_repeat_from_global_id(row["global_id"])
        focus_line = [row["well"], repeat_n, 0, 0, 0]
        if (
            "result" not in row
            or row["result"] == "preprocesor-fail"
            or row["autofocus-status"] == "fail"
        ):
            focus_line[4] = 1
        elif row["autofocus-status"] == "skip":
            focus_line[3] = 1
        if "result" in row and row["result"] == "success":
            focus_line[2] = 1
        focus_results.append(focus_line)
        wells.add(row["well"])
    focus_results = pd.DataFrame(
        data=focus_results, columns=["well", "repeat", "success", "skip", "fail"]
    )
    focus_results.groupby(["well", "repeat"])
    # focus
    for well in list(wells):
        relevant = focus_results[focus_results["well"] == well]
        sums = relevant.groupby(["repeat"], as_index=False).sum()
        nums = relevant.groupby(["repeat"], as_index=False).count()

        fig, ax1 = plt.subplots(dpi=200)
        plt.plot(
            sums["repeat"].to_numpy(),
            (sums["skip"] / nums["well"]).to_numpy(),
            label="Autofocus skip rate",
        )
        plt.plot(
            sums["repeat"].to_numpy(),
            (sums["fail"] / nums["well"]).to_numpy(),
            label="Autofocus fail rate",
        )
        plt.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel("Total imaged cells")
        ax2.plot(
            sums["repeat"].to_numpy(),
            sums["success"].to_numpy(),
            label="Recorded cells",
            color="green",
        )
        ax1.set_xlabel("Repeat number")
        ax1.set_ylabel("Skip/Fail Rate")
        plt.title(well)


def process_repeats(
    folders,
    folder_labels,
    well_labels=None,
    fit_start=5,
    fit_end=18,
    temperature_c=25,
    individual_plots=False,
    filters=None,
    method="rautu",
    reprocess=False,
    end_count=None,
    fitter=None,
    cluster_client=None,
    merge_sample=False,
):
    # isinstance(temperature_c, list)
    full_processed_log = []
    all_metadata = {}
    all_wells = []
    all_labels = []
    all_data = pd.DataFrame([])
    filters_list = filters is not None and isinstance(filters, list)

    for i in tqdm(range(len(folders))):
        collected_data, wells, processed_cell_log, metadata = get_data_from_folder(
            folders[i],
            fit_start if not isinstance(fit_start, list) else fit_start[i],
            fit_end if not isinstance(fit_end, list) else fit_end[i],
            temperature_c if not isinstance(temperature_c, list) else temperature_c[i],
            filters if not filters_list else filters[i],
            method,
            reprocess,
            fitter=fitter,
            cluster_client=cluster_client,
        )

        # If merge_sample is True, use sample names from well_labels for merging
        if merge_sample and well_labels is not None and well_labels[i] is not None:
            sample_map = well_labels[i]
            labels = list(sample_map.values())
            wells_col = [
                folder_labels[i] + sample_map.get(w, w) for w in collected_data["well"]
            ]
            collected_data["well"] = wells_col
            wells = [folder_labels[i] + sample_map.get(w, w) for w in wells]
        else:
            sample_map = well_labels[i]
            labels = list(sample_map.values())
            labels = list(map(lambda x: folder_labels[i] + x, labels))
            wells_col = list(
                map(lambda x: folder_labels[i] + x, collected_data["well"])
            )
            collected_data["well"] = wells_col
            wells = list(map(lambda x: folder_labels[i] + x, wells))
        all_labels += labels
        all_wells += wells
        for w in wells:
            all_metadata[w] = metadata

        well_entry_label = list(
            map(
                lambda x: (
                    x
                    if well_labels[i] is None or x not in well_labels[i]
                    else well_labels[i][x]
                ),
                collected_data["well"],
            )
        )
        well_run_label = list(
            map(
                lambda x: folder_labels[i]
                + (
                    x
                    if well_labels[i] is None or x not in well_labels[i]
                    else well_labels[i][x]
                ),
                collected_data["well"],
            )
        )
        collected_data["well"] = wells_col
        collected_data.insert(1, "sample", well_entry_label)
        collected_data.insert(3, "sample_run", well_run_label)
        collected_data.insert(2, "run", folder_labels[i])
        all_data = pd.concat([all_data, collected_data])  # , ignore_index=True)

        for row in processed_cell_log:
            # If merge_sample is True, update well in cell_log to sample name
            if merge_sample and well_labels is not None and well_labels[i] is not None:
                sample_map = well_labels[i]
                row["well"] = folder_labels[i] + sample_map.get(
                    row["well"], row["well"]
                )
            else:
                row["well"] = folder_labels[i] + row["well"]

            full_processed_log.append(row)

        # full_processed_log += processed_cell_log

        ids = set()
        if individual_plots:
            # TODO: check this still works - haven't used on a while
            plot_individual_cells(collected_data, ids, processed_cell_log)

    all_data["repeat"] = list(map(get_repeat_from_global_id, all_data["global_id"]))
    all_data["group"] = list(map(get_group_from_global_id, all_data["global_id"]))
    if False:
        for well in all_wells:
            visualise_well_results(full_processed_log, well)

        plot_focus_results(full_processed_log)

    if end_count is not None:
        # we want to terminate when the number of successful cells in a repeat reaches end_count
        # then we only process cells which are present in this final loop
        # TODO: should we do cells which are always present instead?
        sucess_counts = all_data.groupby(["well", "repeat"], as_index=False).count()[
            ["well", "repeat", "sigma_value"]
        ]
        end_repeats = (
            sucess_counts[sucess_counts["sigma_value"] > 7]
            .groupby("well")
            .max()["repeat"]
        )
        selected_cells = {}
        for well in all_wells:
            selected_cells[well] = all_data[
                np.logical_and(
                    np.logical_and(
                        all_data["well"] == well,
                        all_data["repeat"] == end_repeats[well],
                    ),
                    ~np.isnan(all_data["sigma_value"]),
                )
            ]["cell_id"]

        filtered_log = []

        for row in full_processed_log:
            selected = list(selected_cells[row["well"]])
            repeat_n = get_repeat_from_global_id(row["global_id"])

            if row["cell_id"] in selected and repeat_n <= end_repeats[row["well"]]:
                filtered_log.append(row)
        all_data = all_data[
            all_data.apply(
                lambda r: r["cell_id"] in list(selected_cells[r["well"]])
                and get_repeat_from_global_id(r["global_id"]) <= end_repeats[r["well"]],
                axis=1,
            )
        ]

        full_processed_log = filtered_log

    if True:
        normalised_data_df = normalise_values(all_data)
        plot_normalised_results(normalised_data_df, "sigma_value")
        # plt.ylim((0,3))
        plot_normalised_results(
            normalised_data_df, "kappa_value", "Normalised Bending modulus/AU"
        )
        # plt.ylim((0,3))
        plot_normalised_results(normalised_data_df, "radius_value", label="Radius/AU")
        plt.ylim((0.7, 1.3))
    # print(collected_data.columns)
    # print(all_data.columns)

    plot_normalised_results(
        all_data, "sigma_value", label="Tension/1e-7 N/m", merge_sample=merge_sample
    )
    # plt.ylim((0,30))
    plot_normalised_results(
        all_data,
        "kappa_value",
        label="Bending modulus/1e-21 J",
        merge_sample=merge_sample,
    )
    # plt.ylim((0,800))
    plot_normalised_results(
        all_data, "radius_value", label="Radius/nm", merge_sample=merge_sample
    )

    plt.figure(dpi=200)
    all_wells = list(set(all_wells))
    for well in all_wells:
        relevant = all_data[all_data["well"] == well]
        grouped = relevant.groupby("repeat", as_index=False).count()
        plt.plot(
            grouped["repeat"].to_numpy(), grouped["sigma_value"].to_numpy(), label=well
        )
    plt.xlabel("Repeat")
    plt.ylabel("Number of cells")
    plt.legend()

    plot_collected_data_time(
        all_data,
        "sigma_value",
        "Tension/1e-7 N/m",
        all_wells,
        err_column="sigma_error",
        unit="*10^{-7} N/m",
    )
    plot_collected_data_time(
        all_data,
        "kappa_value",
        "Bending modulus/1e-21 J",
        all_wells,
        err_column="kappa_error",
        unit="*10^{-21} J",
    )
    plot_collected_data_time(
        all_data, "radius_value", "Radius/nm", all_wells, unit="nm"
    )

    # rolled_std["sigma_value"][-50:]

    return all_data, normalised_data_df, full_processed_log, all_metadata
