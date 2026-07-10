import numpy as np
import scipy
from scipy.interpolate import interp2d, interp1d, RegularGridInterpolator
from scipy.signal import correlate, correlation_lags, windows
import itertools
import cv2
import matplotlib.pyplot as plt
from time import time
from glob import glob
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
import autoimager.CorrelationContourTracker as CCT
from flickering.tracking.contour_io import ContourIO
from tqdm.auto import tqdm
from time import time
from copy import deepcopy
from functools import partial
from flickering.utils.standard_configs import default_fitter, default_tracker
#TODO: make conditional somehow
from IPython.display import Video
import matplotlib.pyplot as plt
from PIL import Image
from flickering.utils.visualisation import *
from datetime import datetime
from flickering.utils.encoder import MultiJsonEncoder

def rebuild_minimal_log(folder):
    existing_gids = []

    # List to store the log entries
    log_entries = []
    current_log ={}
    try:
        current_log = load_cell_log(folder)
        log_entries = current_log["cells"]
        if current_log["description"].startswith("Recovered"):
            print("Log already recovered, aborting")
            return []
        for entry in current_log["cells"]:
            existing_gids.append(entry["global_id"])
        print(f"Recovered existing log: {len(existing_gids)} entries")
    except:
        print("No cell log found, creating new one")
        # Create a new log if it doesn't exist

    # Placeholder values for the log
    default_values = {
        'field': -1,
        'stage_position': [np.nan, np.nan],
        'image_offset': [np.nan, np.nan],
        'center_position': [np.nan, np.nan],
        'video_center_position': [128.5, 128.5],
        'radius': np.nan,
        'pfs_offset': np.nan,
        'z_position':np.nan,
        'mf_cell': [np.nan, np.nan, np.nan],
        'mf_metric_values': [],
        'mf_pfs_offsets': [],
        'autofocus-status': 'success',
        'autofocus-elapsed': np.nan,
        'autofocus-skip-metrics': [],
        'fps': 660,
        'duration': 20,
        'shutter': 0.0008,
        'result': 'success',
    }


    # Parse the .movie files
    for filename in os.listdir(folder):
        if filename.endswith(".movie"):
            # Extract parts of the filename
            parts = filename.split("-")
            well = parts[0]
            cell_id = int(parts[1])
            repeat_and_time = parts[2].split(".")
            repeat_part = repeat_and_time[0]
            if "_" in repeat_part:
                repeat_str, group_suffix = repeat_part.split("_", 1)
                repeat = int(repeat_str)
                global_id = f"{well}-{cell_id}-{repeat}_{group_suffix}"
            else:
                repeat = int(repeat_part)
                global_id = f"{well}-{cell_id}-{repeat}"
            timestamp_str = repeat_and_time[1:-1]  # Extract the timestamp part (e.g., "02Apr2025_22.14.50")

            # Parse the timestamp from the filename
            timestamp_format = "%d%b%Y_%H.%M.%S"  # Format of the timestamp in the filename
            start_time = datetime.strptime(".".join(timestamp_str), timestamp_format).timestamp()

            # Calculate finish_time using the duration
            finish_time = start_time + default_values['duration']

            # Create the log entry
            log_entry = {
                'well': well,
                'field': default_values['field'],
                'stage_position': default_values['stage_position'],
                'image_offset': default_values['image_offset'],
                'center_position': default_values['center_position'],
                'video_center_position': default_values['video_center_position'],
                'radius': default_values['radius'],
                'global_id': global_id,
                'cell_id': cell_id,
                'pfs_offset': default_values['pfs_offset'],
                'start_time': start_time,
                'z_position': default_values['z_position'],
                'mf_cell': default_values['mf_cell'],
                'mf_metric_values': default_values['mf_metric_values'],
                'mf_pfs_offsets': default_values['mf_pfs_offsets'],
                'autofocus-status': default_values['autofocus-status'],
                'autofocus-elapsed': default_values['autofocus-elapsed'],
                'autofocus-skip-metrics': default_values['autofocus-skip-metrics'],
                'fps': default_values['fps'],
                'duration': default_values['duration'],
                'shutter': default_values['shutter'],
                'result': default_values['result'],
                'finish_time': finish_time,
            }
            if global_id in existing_gids:
                # If the global_id already exists, skip adding it
                print(f"Skipping existing global_id: {global_id}")
                continue
            # Add the log entry to the list
            log_entries.append(log_entry)

    with open(f"{folder}/{time():.0f}.json", "w") as logfile:
        logfile.write(json.dumps(
            {"cells": log_entries,
                "metadata": {} if not "metadata" in current_log else current_log["metadata"],
                "description": "Recovered log" if not "description" in current_log else "Recovered. "+current_log["description"],
                "info": {} if not "info" in current_log else current_log["info"]
            }, cls=MultiJsonEncoder#,
                                #default=lambda o: f"<not serializable: {str(type(o))}>" #this can cause missing info
            ))

    print(f"Rebuilt log with {len(log_entries)} entries")

    return log_entries

def load_cell_log(folder):
    def parse_fname(f):
        try:
            return int(f.split("/")[-1].replace(".json",""))
        except:
            return 0
    fname = str(max(map(parse_fname, glob(folder+"/*.json"))))+".json"
    with open(f"{folder}/{fname}", "r") as f:
        return json.load(f)

def plot_debug(log_row, metadata, contour_video = False, cf = None):
    #TODO: this assumes default_cf was used, we it should ideally recover cf from metadata somehow
    if cf is None:
        cf = default_cf(metadata)
        cf.min_mode = metadata["fit_start"]
        cf.max_mode = metadata["fit_end"]
    if "fit_results" not in log_row:
        print("No fit results")
        return False
    if "preview_file" in log_row:
        img = Image.open(log_row["preview_file"])
        plt.figure(dpi=200)
        plt.imshow(img, cmap="gray")
    if "contour_file" in log_row and "preview_file" in log_row:
        cio = ContourIO(log_row["contour_file"])
        plt.figure(dpi=200)
        if cio.mode == "XY":
            plt.imshow(draw_xy_contour(np.array(img), cio.contours[0]))
        else:
            xy_contour = CCT.CorrelationContourTracker.convert_contour_xy(cio.contours[0], cio.centers[0])
            plt.imshow(draw_xy_contour(np.array(img), xy_contour))

        if contour_video and "movie_f" in log_row:
            video_name = "/tmp/"+log_row["global_id"]+".mp4"
            generate_contour_video(movie_file=log_row["movie_f"], contour_file=log_row["contour_file"], output_file=video_name)
            Video(video_name)

    plt.figure(dpi=200)
    cf.plot_spectrum(np.array(log_row["fit_results"]["mps"]),np.array(log_row["fit_results"]["mps_err"]), log_row["fit_results"], np.array(log_row["fit_results"]["alpha_beta"]), np.array(log_row["fit_results"]["decay_times"]["value"]), cf.delta(log_row["fit_results"]["radius"]["value"]))
    plt.title(log_row["global_id"])
    return True

def default_cf(metadata):
    cf = default_fitter()
    if "temperature_c" in metadata:
        cf.set_temperature(metadata["temperature_c"]+273)
    if "method" in metadata:
        logging.warning(f"Overriding method to {metadata['method']}")
        cf.fitting_method = metadata["method"]
    if "fps" in metadata:
        cf.delay_between_frames_ms = 1000/metadata["fps"]
    if "shutter" in metadata:
        cf.exposure_time_ms = metadata["shutter"]*1000 #TODO: read from metada
    #print(cf.exposure_time_ms)
    #print(cf.delay_between_frames_ms)
    #cf.radius_correction_nm = 0
    if "min_mode" in metadata:
        logging.warning(f"Overriding min_mode to {metadata['min_mode']}")
        cf.min_mode = metadata["fit_start"]
    if "max_mode" in metadata:
        logging.warning(f"Overriding max_mode to {metadata['max_mode']}")
        cf.max_mode = metadata["fit_end"]

    return cf

def find_cell_entries(log, search_dict, limit = None):
    result=[]
    for entry in log:
        match = True
        for key, value in search_dict.items():
            key_parts = key.split("|")
            sub_entry = entry
            #work with nested keys
            for kp in key_parts:
                if kp not in sub_entry:
                    match = False
                    break
                sub_entry = sub_entry[kp]
            if not match:
                break

            if type(value) == str:
                if value.startswith("<"):
                    value = value[1:]
                    if sub_entry >= float(value):
                        match = False
                        break
                elif value.startswith(">"):
                    value = value[1:]
                    if sub_entry <= float(value):
                        match = False
                        break
                elif value.startswith("!"):
                    value = value[1:]
                    if sub_entry == value:
                        match = False
                        break
                elif sub_entry != value:
                    match = False
                    break
            elif sub_entry != value:
                match = False
                break

        if match:
            result.append(entry)
        if limit is not None and len(result) >= limit:
            break

    return result

