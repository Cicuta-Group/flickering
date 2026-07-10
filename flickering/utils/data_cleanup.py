import os
import concurrent.futures
import logging
import cv2
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Callable, Dict, Tuple, Union, Any
from flickering.utils.process_multiwell import process_folder_caching
from flickering.analysis.fitter import ContourFitter
from flickering.tracking.correlation_tracker import CorrelationContourTracker
from flickering.utils.standard_configs import default_fitter
from flickering.utils.movie_reader import get_movie_reader
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Lazy imports for heavy dependencies
RandomForestAutofocus = None
RandomForestCellValidator = None
CellFindingExperiment = None
TemikaMicroscope = None

# Global singletons for process-based parallelism
_GLOBAL_RF_AUTOFOCUS = None
_GLOBAL_RF_VALIDATOR = None
_GLOBAL_EXPERIMENT_DUMMY = None


def get_global_rf_autofocus():
    global _GLOBAL_RF_AUTOFOCUS, _GLOBAL_EXPERIMENT_DUMMY, RandomForestAutofocus, CellFindingExperiment, TemikaMicroscope
    if _GLOBAL_RF_AUTOFOCUS is None:
        if RandomForestAutofocus is None:
            from flickering.acquisition.autofocus.RandomForestAutofocus import RandomForestAutofocus
            from flickering.acquisition.autoimager import CellFindingExperiment
            from temika.microscope import TemikaMicroscope

        if _GLOBAL_EXPERIMENT_DUMMY is None:
            _GLOBAL_EXPERIMENT_DUMMY = CellFindingExperiment(None)

        _GLOBAL_RF_AUTOFOCUS = RandomForestAutofocus(_GLOBAL_EXPERIMENT_DUMMY)
    return _GLOBAL_RF_AUTOFOCUS


def get_global_rf_validator():
    global _GLOBAL_RF_VALIDATOR, RandomForestCellValidator
    if _GLOBAL_RF_VALIDATOR is None:
        if RandomForestCellValidator is None:
            from flickering.acquisition.autofocus.RandomForestAutofocus import RandomForestCellValidator

        _GLOBAL_RF_VALIDATOR = RandomForestCellValidator(None)
        _GLOBAL_RF_VALIDATOR.cell_preprocessor = get_global_rf_autofocus()
        _GLOBAL_RF_VALIDATOR.initialise_validator()
    return _GLOBAL_RF_VALIDATOR


def read_frame_zero(movie_file: str) -> Union[np.ndarray, None]:
    """Top-level function to read frame 0, safe for multiprocessing."""
    if not movie_file or not os.path.exists(movie_file):
        return None

    try:
        if movie_file.endswith(".movie"):
            try:
                # Use custom Movie reader
                mov = get_movie_reader(movie_file)
                frame = mov.get_frame(0)
                mov.destroy()
                if frame is not None:
                    return frame
            except Exception as e:
                print(f"pytmk failed for {movie_file}: {e}. Trying cv2 fallback.")

        # Try cv2 for other formats OR as fallback for .movie
        cap = cv2.VideoCapture(movie_file)
        ret, frame = cap.read()
        cap.release()
        if ret:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        return None
    except Exception as e:
        # In multiprocessing, logging might be tricky, but we can print or ignore
        print(f"Error reading {movie_file}: {e}")
        return None


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    elif isinstance(obj, (np.ndarray,)):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    return obj


def process_row_task(row, exclusions, filters):
    """Top-level worker function for processing a row."""
    movie_file = row.get("movie_f")
    global_id = str(row.get("global_id", "Unknown"))

    # Define frame loader callback
    def frame_loader():
        return read_frame_zero(movie_file)

    keep = True
    reasons = []
    filter_data = {}
    protected = False

    # Check exclusions first
    if global_id in exclusions:
        protected = True
        reasons.append("Manual Exclusion")

    for filter_func, desc in filters:
        # filter_func is picklable (top-level function)
        result = filter_func(row, frame_loader)
        if len(result) == 3:
            status, reason, metadata = result
        else:
            status, reason = result
            metadata = {}

        if metadata:
            if "metric" in metadata:
                filter_data[metadata["metric"]] = metadata
            else:
                filter_data.update(metadata)

        if status == "PROTECT":
            protected = True
            reasons.append(f"PROTECTED: {reason}")
        elif status is False:
            keep = False
            reasons.append(reason)

    # Create a slim row for the report
    slim_row = {
        "global_id": global_id,
        "movie_f": movie_file,
        "preview_file": row.get("preview_file"),
        "filter_data": filter_data,
        "cleanup_reasons": reasons,
        "fit_status": row.get("fit_status"),
    }

    space_freed = 0
    if protected:
        slim_row["cleanup_status"] = "PROTECTED"
        if global_id in exclusions:
            slim_row["cleanup_status"] = "KEEP_EXCLUDED"
    elif not keep:
        slim_row["cleanup_status"] = "DELETE"
        if movie_file and os.path.exists(movie_file):
            space_freed = os.path.getsize(movie_file)
    else:
        slim_row["cleanup_status"] = "KEEP"

    return slim_row, space_freed


def generate_training_data_task(row, output_dir, n_frames):
    """Top-level worker function for generating training data."""
    movie_file = row.get("movie_f")
    global_id = row.get("global_id", "unknown")
    status = row.get("cleanup_status", "KEEP")
    reasons = row.get("cleanup_reasons", [])
    filter_data = row.get("filter_data", {})

    if not movie_file or not os.path.exists(movie_file):
        return None

    try:
        saved_files = []
        if movie_file.endswith(".movie"):
            mov = get_movie_reader(movie_file)
            total_frames = mov.n_frames

            if n_frames == 1:
                indices = [0]
            else:
                indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

            for i, frame_idx in enumerate(indices):
                frame = mov.get_frame(int(frame_idx))
                if frame is not None:
                    filename = f"frame_{global_id}_{i}.npz"
                    filepath = os.path.join(output_dir, filename)
                    np.savez_compressed(filepath, frame=frame)
                    saved_files.append(filename)

            mov.destroy()

        else:
            frame = read_frame_zero(movie_file)
            if frame is not None:
                filename = f"frame_{global_id}_0.npz"
                filepath = os.path.join(output_dir, filename)
                np.savez_compressed(filepath, frame=frame)
                saved_files.append(filename)

        if saved_files:
            return {
                "global_id": global_id,
                "filenames": saved_files,
                "status": status,
                "reasons": reasons,
                "filter_data": filter_data,
                "original_movie": movie_file,
            }
    except Exception as e:
        print(f"Error processing {movie_file} for training data: {e}")
    return None


class DataCleanup:
    def __init__(self, folder: str, fitter: Union[ContourFitter, None] = None):
        self.folder = folder
        self.logger = logging.getLogger("DataCleanup")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.fitter = fitter if fitter is not None else default_fitter()
        self.filters: List[Tuple[Callable, str]] = []
        self.cell_log = []
        self.metadata = {}

        # Lazy loaded instances
        self._rf_autofocus = None
        self._rf_validator = None
        self._experiment_dummy = None

        # Cache for frame 0 (only used in interactive/main process now)
        self._frame_cache = {}

    def _get_rf_autofocus(self):
        return get_global_rf_autofocus()

    def _get_rf_validator(self):
        return get_global_rf_validator()

    def load_data(self, load_last_cache=True):
        self.logger.info(f"Loading data from {self.folder}")
        self.cell_log, self.metadata = process_folder_caching(
            self.folder,
            cf=self.fitter,
            load_last_cache=load_last_cache,  # Prefer loading existing cache to save time
        )
        self._fix_paths()
        self.logger.info(f"Loaded {len(self.cell_log)} entries")

    def _fix_paths(self):
        """
        Fix paths in cell_log if they point to non-existent locations
        (e.g. from a different mount point).
        """
        count = 0
        for row in tqdm(self.cell_log, desc="Fixing paths"):
            for key in ["movie_f", "preview_file", "contour_file"]:
                try:
                    if key in row and row[key]:
                        val = row[key]
                        if isinstance(val, list):
                            new_list = []
                            for item in val:
                                if item:
                                    if self.folder not in item or not os.path.exists(item):
                                        basename = os.path.basename(item)
                                        new_path = os.path.join(self.folder, basename)
                                        if os.path.exists(new_path):
                                            new_list.append(new_path)
                                            count += 1
                                        else:
                                            new_list.append(item)
                                    else:
                                        new_list.append(item)
                            row[key] = new_list
                        else:
                            if self.folder not in val or not os.path.exists(val):
                                basename = os.path.basename(val)
                                new_path = os.path.join(self.folder, basename)
                                if os.path.exists(new_path):
                                    row[key] = new_path
                                    count += 1
                except Exception as e:
                    self.logger.error(f"Error fixing path for key {key} in row: {e}")
        if count > 0:
            self.logger.info(f"Fixed {count} paths to match current folder.")

    def add_filter(
        self,
        filter_func: Callable[[Dict, Callable], Tuple[Union[bool, str], str]],
        description: str,
    ):
        """
        Register a filter.
        filter_func: (row, frame_loader) -> (status, reason)
        status: True (Keep), False (Delete), "PROTECT" (Force Keep)
        frame_loader: () -> np.ndarray (loads frame 0 of the movie)
        """
        self.filters.append((filter_func, description))

    def get_frame_zero(self, movie_file: str) -> Union[np.ndarray, None]:
        if movie_file is None:
            return None

        with self._lock:
            if movie_file in self._frame_cache:
                return self._frame_cache[movie_file]

        try:
            # We lock the file reading as well, just in case pytmk or cv2 has thread safety issues
            # This might reduce parallelism but ensures correctness.
            # If performance is too slow, we can try narrowing the lock scope later.
            with self._lock:
                if movie_file.endswith(".movie"):
                    # Use custom Movie reader
                    mov = get_movie_reader(movie_file)
                    frame = mov.get_frame(0)
                    mov.destroy()
                    if frame is not None:
                        self._frame_cache[movie_file] = frame
                        return frame
                    else:
                        self.logger.warning(
                            f"Could not read frame 0 from {movie_file} using MovieReader"
                        )
                        return None
                else:
                    # Try cv2 for other formats (e.g. .mkv, .avi)
                    cap = cv2.VideoCapture(movie_file)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        if len(frame.shape) == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        self._frame_cache[movie_file] = frame
                        return frame
                    else:
                        self.logger.warning(f"Could not read frame 0 from {movie_file}")
                        return None
        except Exception as e:
            self.logger.error(f"Error reading {movie_file}: {e}")
            return None

    def check_contrast(self, n_samples=20, output_path="contrast_check.jpg"):
        """Generate a grid of random images with their contrast values."""
        if not self.cell_log:
            self.load_data()

        import random

        samples = random.sample(self.cell_log, min(n_samples, len(self.cell_log)))

        # Prepare data for grid generation
        grid_data = []
        for row in samples:
            movie_file = row.get("movie_f")
            global_id = str(row.get("global_id", "Unknown"))

            frame = self.get_frame_zero(movie_file)
            contrast = 0
            if frame is not None:
                contrast = contrast_metric(frame)

            # Create a dummy row for generate_preview_grid
            dummy_row = {
                "global_id": global_id,
                "movie_f": movie_file,
                "preview_file": row.get("preview_file"),
                "cleanup_reasons": [f"Contrast: {contrast:.2f}"],
                "cleanup_status": "CHECK",
            }
            grid_data.append(dummy_row)

    def reduce_memory_usage(self, keys_to_drop: List[str]):
        """
        Drop specified keys from cell_log entries to save memory.
        Useful before multiprocessing which pickles data.
        """
        if not keys_to_drop:
            return

        self.logger.info(f"Reducing memory usage by dropping keys: {keys_to_drop}")
        count = 0
        for row in self.cell_log:
            if "fit_results" in row:
                for key in keys_to_drop:
                    if key in row["fit_results"]:
                        del row["fit_results"][key]
                        count += 1
                # Also check inside nested dicts if needed, but usually top-level is enough
                # For fit_results, we might want to drop specific sub-keys?
                # For now, just top-level keys.

        # Force garbage collection
        import gc

        gc.collect()
        self.logger.info(f"Dropped {count} items. GC collected.")

    def run(
        self,
        dry_run=True,
        delete=False,
        interactive=False,
        generate_training_data=False,
        training_frames=1,
        exclude_file=None,
        resume_from=None,
        check_contrast=None,
        workers=None,
        drop_keys=None,
        auto_confirm=False,
        load_last_cache=True,
    ):
        if check_contrast:
            self.check_contrast(check_contrast)
            return

        if resume_from:
            self.logger.info(f"Resuming from report: {resume_from}")
            if not os.path.exists(resume_from):
                self.logger.error(f"Report file not found: {resume_from}")
                return
            with open(resume_from, "r") as f:
                processed_rows = json.load(f)

            # Apply exclusions if provided
            exclusions = set()
            if exclude_file and os.path.exists(exclude_file):
                with open(exclude_file, "r") as f:
                    exclusions = set(line.strip() for line in f if line.strip())
                self.logger.info(f"Loaded {len(exclusions)} exclusions.")

            # Update status based on exclusions
            files_to_delete = []
            for row in processed_rows:
                gid = row.get("global_id")
                if gid in exclusions:
                    row["cleanup_status"] = "KEEP_EXCLUDED"

                if row.get("cleanup_status") == "DELETE":
                    # Only count as "to delete" if the file actually exists
                    if row.get("movie_f") and os.path.exists(row["movie_f"]):
                        files_to_delete.append(row)

            self.logger.info(
                f"Resumed with {len(files_to_delete)} files marked for deletion."
            )

        else:
            if not self.cell_log:
                self.load_data(load_last_cache=load_last_cache)

            # Reduce memory usage if requested
            if drop_keys:
                self.reduce_memory_usage(drop_keys)

            # Load exclusions
            exclusions = set()
            if exclude_file and os.path.exists(exclude_file):
                with open(exclude_file, "r") as f:
                    exclusions = set(line.strip() for line in f if line.strip())
                self.logger.info(f"Loaded {len(exclusions)} exclusions.")

            self.logger.info(
                f"Running filters with {workers or os.cpu_count()} workers..."
            )
            processed_rows = []
            files_to_delete = []
            files_protected = 0
            space_to_free = 0

            # Use ProcessPoolExecutor for parallel processing
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers
            ) as executor:
                # Submit tasks
                futures = {
                    executor.submit(
                        process_row_task, row, exclusions, self.filters
                    ): row
                    for row in self.cell_log
                }

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(self.cell_log),
                    desc="Filtering",
                ):
                    try:
                        slim_row, space_freed = future.result()
                        processed_rows.append(slim_row)

                        if slim_row["cleanup_status"] == "DELETE":
                            # Only count as "to delete" if the file actually exists
                            if slim_row.get("movie_f") and os.path.exists(
                                slim_row["movie_f"]
                            ):
                                files_to_delete.append(slim_row)
                                space_to_free += space_freed
                        elif slim_row["cleanup_status"] in [
                            "PROTECTED",
                            "KEEP_EXCLUDED",
                        ]:
                            files_protected += 1
                    except Exception as e:
                        self.logger.error(f"Error processing row: {e}")

            # Save report
            report_file = os.path.join(self.folder, "cleanup_report.json")
            with open(report_file, "w") as f:
                sanitized_rows = sanitize_for_json(processed_rows)
                json.dump(sanitized_rows, f, indent=4)
            self.logger.info(f"Saved cleanup report to {report_file}")

            self.logger.info(
                f"Identified {len(files_to_delete)} files for deletion (existing movies)."
            )
            self.logger.info(f"Protected {files_protected} files.")
            self.logger.info(
                f"Estimated space to free: {space_to_free / (1024*1024*1024):.2f} GB"
            )

        # Generate preview grid for files to delete
        if files_to_delete:
            self.generate_preview_grid(
                files_to_delete,
                output_path=os.path.join(self.folder, "cleanup_preview.jpg"),
            )

        if generate_training_data:
            self.generate_training_dataset(
                processed_rows, n_frames=training_frames, workers=workers
            )

        if interactive:
            self.run_interactive(files_to_delete)
        elif delete:
            if dry_run:
                self.logger.info("Dry run enabled. Skipping deletion.")
            else:
                if not auto_confirm:
                    confirm = input(
                        f"Proceed to delete {len(files_to_delete)} files? [y/N]: "
                    )
                else:
                    confirm = "y"
                if confirm.lower() == "y":
                    self.run_delete(files_to_delete)
                else:
                    self.logger.info("Deletion cancelled.")
        else:
            self.logger.info("Dry-run complete. No files deleted.")

    def generate_preview_grid(
        self, files: List[Dict], output_path="cleanup_preview.jpg"
    ):
        # Filter out entries with no movie or preview file (rejected before recording)
        valid_files = [
            f
            for f in files
            if (f.get("movie_f") and os.path.exists(f["movie_f"]))
            or (f.get("preview_file") and os.path.exists(f["preview_file"]))
        ]

        if not valid_files:
            self.logger.info("No valid files with content to preview.")
            return

        files = valid_files  # Use filtered list

        # Limit to 50 files for preview to avoid huge images
        batch_size = 50
        batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

        for i, batch in enumerate(batches):
            batch_output_path = output_path.replace(".jpg", f"_{i}.jpg")
            self.logger.info(f"Generating preview grid {i+1}/{len(batches)}...")

            # Grid dimensions
            cols = 8
            rows = (len(batch) + cols - 1) // cols

            cell_w, cell_h = 200, 200
            grid_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

            for idx, row in enumerate(batch):
                r, c = divmod(idx, cols)

                # Get image
                img = None
                # Try preview file first
                if row.get("preview_file") and os.path.exists(row["preview_file"]):
                    try:
                        img = cv2.imread(row["preview_file"])
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to read preview {row['preview_file']}: {e}"
                        )

                # Fallback to movie frame
                if img is None and row.get("movie_f"):
                    try:
                        frame = self.get_frame_zero(row["movie_f"])
                        if frame is not None:
                            # Normalize for display
                            if frame.dtype != np.uint8:
                                frame_norm = cv2.normalize(
                                    frame, None, 0, 255, cv2.NORM_MINMAX
                                )
                                frame_uint8 = frame_norm.astype(np.uint8)
                            else:
                                frame_uint8 = frame
                            img = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to read movie frame {row['movie_f']}: {e}"
                        )

                if img is None:
                    img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    cv2.putText(
                        img,
                        "No Image",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                else:
                    img = cv2.resize(img, (cell_w, cell_h))

                # Overlay info
                global_id = str(row.get("global_id", "Unknown"))
                reasons = row.get("cleanup_reasons", [])

                # Draw ID
                cv2.putText(
                    img,
                    global_id,
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Draw Reasons
                y = 40
                if reasons:
                    for reason in reasons[:3]:  # Limit reasons
                        cv2.putText(
                            img,
                            str(reason),
                            (5, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                        )
                        y += 15
                else:
                    cv2.putText(
                        img,
                        "No Reason",
                        (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )

                grid_img[
                    r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w
                ] = img

            cv2.imwrite(batch_output_path, grid_img)
            self.logger.info(f"Saved preview grid to {batch_output_path}")

    def generate_training_dataset(self, files: List[Dict], n_frames=1, workers=None):
        output_dir = os.path.join(self.folder, "training_dataset")
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(
            f"Generating training dataset in {output_dir} with {n_frames} frames per file using {workers or os.cpu_count()} workers..."
        )

        dataset_metadata = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    generate_training_data_task, row, output_dir, n_frames
                ): row
                for row in files
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(files),
                desc="Generating Dataset",
            ):
                try:
                    result = future.result()
                    if result:
                        dataset_metadata.append(result)
                except Exception as e:
                    self.logger.error(f"Error generating training data: {e}")

        # Save metadata
        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, "w") as f:
            sanitized_metadata = sanitize_for_json(dataset_metadata)
            json.dump(sanitized_metadata, f, indent=4)

        self.logger.info(
            f"Saved training dataset with {len(dataset_metadata)} samples."
        )

    def run_interactive(self, files: List[Dict]):
        self.logger.info("Starting interactive mode...")
        self.logger.info("Controls: [d] Delete, [k] Keep, [b] Back, [q] Quit")

        idx = 0
        use_matplotlib = False

        while 0 <= idx < len(files):
            row = files[idx]
            global_id = str(row.get("global_id", "Unknown"))
            reasons = row.get("cleanup_reasons", [])

            # Get image
            img = None
            if row.get("preview_file") and os.path.exists(row["preview_file"]):
                img = cv2.imread(row["preview_file"])
            elif row.get("movie_f"):
                frame = self.get_frame_zero(row["movie_f"])
                if frame is not None:
                    # Normalize for display
                    if frame.dtype != np.uint8:
                        frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                        frame_uint8 = frame_norm.astype(np.uint8)
                    else:
                        frame_uint8 = frame
                    img = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

            if img is None:
                img = np.zeros((400, 400, 3), dtype=np.uint8)
                cv2.putText(
                    img,
                    "No Image",
                    (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            # Overlay info
            status = row.get("cleanup_status", "DELETE")
            color = (0, 0, 255) if status == "DELETE" else (0, 255, 0)

            cv2.putText(
                img,
                f"ID: {global_id} ({idx+1}/{len(files)})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"Status: {status}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            y = 90
            for r in reasons:
                cv2.putText(
                    img, r, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )
                y += 20

            # Display logic
            key = None
            if not use_matplotlib:
                try:
                    cv2.imshow("Data Cleanup", img)
                    key = cv2.waitKey(0) & 0xFF
                except cv2.error:
                    self.logger.warning(
                        "cv2.imshow failed (likely headless). Switching to matplotlib."
                    )
                    use_matplotlib = True
                    cv2.destroyAllWindows()

            if use_matplotlib:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"ID: {global_id} - {status}")
                plt.axis("off")
                plt.draw()
                # Wait for key press
                # plt.waitforbuttonpress is blocking but returns True for key, False for mouse
                # We need to capture the key value.
                # This is tricky in non-interactive backends, but let's try.
                print(f"ID: {global_id} | Status: {status} | Reasons: {reasons}")
                print("Press [d]elete, [k]eep, [b]ack, [q]uit in terminal + Enter")
                # Fallback to terminal input for safety
                user_input = input("Action [d/k/b/q]: ").lower()
                if user_input == "d":
                    key = ord("d")
                elif user_input == "k":
                    key = ord("k")
                elif user_input == "b":
                    key = ord("b")
                elif user_input == "q":
                    key = ord("q")
                else:
                    key = 0  # Invalid

                plt.close()

            if key == ord("d"):
                row["cleanup_status"] = "DELETE"
                idx += 1
            elif key == ord("k"):
                row["cleanup_status"] = "KEEP"
                idx += 1
            elif key == ord("b"):
                idx = max(0, idx - 1)
            elif key == ord("q"):
                break

        if not use_matplotlib:
            cv2.destroyAllWindows()
        else:
            plt.close("all")

        # Re-evaluate files to delete based on interactive choices
        final_delete = [f for f in files if f.get("cleanup_status") == "DELETE"]
        self.logger.info(
            f"Interactive mode finished. {len(final_delete)} files confirmed for deletion."
        )

        if final_delete:
            confirm = input(f"Proceed to delete {len(final_delete)} files? [y/N]: ")
            if confirm.lower() == "y":
                self.run_delete(final_delete)
            else:
                self.logger.info("Deletion cancelled.")

    def run_delete(self, files: List[Dict]):
        self.logger.info(f"Deleting {len(files)} files...")
        deleted_count = 0
        freed_space = 0

        report = []

        for row in files:
            if row.get("cleanup_status") != "DELETE":
                continue

            movie_file = row.get("movie_f")
            if movie_file and os.path.exists(movie_file):
                try:
                    size = os.path.getsize(movie_file)
                    os.remove(movie_file)
                    freed_space += size
                    deleted_count += 1
                    self.logger.info(f"Deleted {movie_file}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {movie_file}: {e}")

            report.append(
                {
                    "global_id": row.get("global_id"),
                    "movie_file": movie_file,
                    "reasons": row.get("cleanup_reasons"),
                }
            )

        self.logger.info(
            f"Deleted {deleted_count} files. Freed {freed_space / (1024*1024*1024):.2f} GB."
        )

        # Save report
        report_file = os.path.join(
            self.folder,
            f"cleanup_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        pd.DataFrame(report).to_json(report_file, orient="records", indent=4)
        self.logger.info(f"Deletion report saved to {report_file}")


# Standard Filters
def filter_fit_status(row, loader):
    status = row.get("fit_status")
    meta = {"metric": "fit_status", "value": status}
    if status == "success":
        return True, "OK", meta
    return False, f"Fit: {status}", meta


class R2Filter:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def __call__(self, row, loader):
        val = None
        if "fit_results" in row and "r2" in row["fit_results"]:
            val = row["fit_results"]["r2"]

        meta = {"metric": "r2", "value": val, "threshold": self.threshold}

        if val is not None:
            if val > self.threshold:
                return True, "OK", meta
            return False, f"R2 {val:.2f} < {self.threshold}", meta
        return False, "No R2", meta


def filter_r2(threshold=0.8):
    return R2Filter(threshold)


class SigmaFilter:
    def __init__(self, min_val=0, max_val=100):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, row, loader):
        val = None
        if "fit_results" in row and "sigma" in row["fit_results"]:
            val = row["fit_results"]["sigma"]["value"]

        meta = {
            "metric": "sigma",
            "value": val,
            "min": self.min_val,
            "max": self.max_val,
        }

        if val is not None:
            if self.min_val <= val <= self.max_val:
                return True, "OK", meta
            return False, f"Sigma {val:.1f} out of range", meta
        return False, "No Sigma", meta


def filter_sigma(min_val=0, max_val=100):
    return SigmaFilter(min_val, max_val)


class KappaFilter:
    def __init__(self, min_val=0, max_val=1000):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, row, loader):
        val = None
        if "fit_results" in row and "kappa" in row["fit_results"]:
            val = row["fit_results"]["kappa"]["value"]

        meta = {
            "metric": "kappa",
            "value": val,
            "min": self.min_val,
            "max": self.max_val,
        }

        if val is not None:
            if self.min_val <= val <= self.max_val:
                return True, "OK", meta
            return False, f"Kappa {val:.1f} out of range", meta
        return False, "No Kappa", meta


def filter_kappa(min_val=0, max_val=1000):
    return KappaFilter(min_val, max_val)


class SigmaErrorFilter:
    def __call__(self, row, loader):
        val = None
        err = None
        if "fit_results" in row and "sigma" in row["fit_results"]:
            val = row["fit_results"]["sigma"]["value"]
            err = row["fit_results"]["sigma"]["error"]

        meta = {"metric": "sigma_error", "value": val, "error": err}

        if val is not None:
            if err is None or val > err:
                return True, "OK", meta
            return False, f"Sigma {val:.1f} < Err {err:.1f}", meta
        return False, "No Sigma", meta


def filter_sigma_error():
    return SigmaErrorFilter()


class KappaErrorFilter:
    def __call__(self, row, loader):
        val = None
        err = None
        if "fit_results" in row and "kappa" in row["fit_results"]:
            val = row["fit_results"]["kappa"]["value"]
            err = row["fit_results"]["kappa"]["error"]

        meta = {"metric": "kappa_error", "value": val, "error": err}

        if val is not None:
            if err is None or val > err:
                return True, "OK", meta
            return False, f"Kappa {val:.1f} < Err {err:.1f}", meta
        return False, "No Kappa", meta


def filter_kappa_error():
    return KappaErrorFilter()


class ValidContourFilter:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, row, loader):
        val = row.get("valid_contour_rate")
        meta = {
            "metric": "valid_contour_rate",
            "value": val,
            "threshold": self.threshold,
        }

        if val is not None:
            if val > self.threshold:
                return True, "OK", meta
            return False, f"ValidContour {val:.2f} < {self.threshold}", meta
        return False, "No Contour Rate", meta


def filter_valid_contour(threshold=0.5):
    return ValidContourFilter(threshold)


class ImageFocusFilter:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, row, loader):
        meta = {"metric": "focus_prob", "value": None, "threshold": self.threshold}

        frame = loader()
        if frame is None:
            return False, "No Frame", meta

        # Use global accessor for multiprocessing safety
        af = get_global_rf_autofocus()
        is_focused, metrics = af.determine_abs_focus(frame, skip_normalise=False)

        prob = metrics[-1] if len(metrics) > 0 else 0
        meta["value"] = prob

        if prob > self.threshold:
            return True, "OK", meta
        return False, f"Focus Prob {prob:.2f} < {self.threshold}", meta


def filter_image_focus(threshold=0.5, cleaner=None):
    return ImageFocusFilter(threshold)


class ImageValidationFilter:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, row, loader):
        meta = {
            "metric": "validation_passed",
            "value": None,
            "threshold": self.threshold,
        }

        frame = loader()
        if frame is None:
            return False, "No Frame", meta

        # Use global accessor for multiprocessing safety
        validator = get_global_rf_validator()
        validator.validation_threshold = self.threshold

        is_valid = validator.validate_cell(frame, final=True)
        meta["value"] = is_valid

        if is_valid:
            return True, "OK", meta
        return False, "Validation Failed", meta


def filter_image_validation(threshold=0.5, cleaner=None):
    return ImageValidationFilter(threshold)


class ContrastFilter:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, row, loader):
        frame = loader()
        if frame is None:
            return False, "No Frame", {"metric": "contrast", "value": None}

        val = np.std(frame)
        if val >= self.threshold:
            return (
                True,
                "OK",
                {"metric": "contrast", "value": val, "threshold": self.threshold},
            )
        return (
            False,
            f"Contrast {val:.2f} < {self.threshold}",
            {"metric": "contrast", "value": val, "threshold": self.threshold},
        )


def filter_contrast(threshold):
    return ContrastFilter(threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Cleanup Utility")
    parser.add_argument("folders", nargs="+", help="Folders to process")
    parser.add_argument(
        "--dry-run", action="store_true", default=True, help="Dry run (default)"
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete files")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--config", action="store_true", help="Use standard configuration"
    )
    parser.add_argument(
        "--generate-training-data",
        action="store_true",
        help="Generate training dataset (npz frames)",
    )
    parser.add_argument(
        "--training-frames",
        type=int,
        default=1,
        help="Number of frames per file for training data",
    )
    parser.add_argument(
        "--exclude-file",
        type=str,
        help="Text file containing list of global_ids to exclude from deletion (force keep)",
    )
    parser.add_argument(
        "--resume-from", type=str, help="Resume from a JSON report file (skip filters)"
    )
    parser.add_argument(
        "--check-contrast",
        type=int,
        help="Generate a grid of N random images with contrast values",
    )
    parser.add_argument(
        "--filter-contrast", type=float, help="Filter by contrast threshold (std dev)"
    )
    parser.add_argument(
        "--workers", type=int, help="Number of worker threads (default: CPU count)"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Automatically confirm deletion"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force reprocessing of data (ignore cache)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # If delete or interactive is specified, disable dry-run (unless explicitly set, but argparse defaults handle this)
    if args.delete or args.interactive:
        args.dry_run = False

    for folder in args.folders:
        print(f"Processing {folder}...")
        cleaner = DataCleanup(folder)

        if args.filter_contrast is not None:
            cleaner.add_filter(
                filter_contrast(args.filter_contrast),
                f"Contrast > {args.filter_contrast}",
            )

        if args.config:
            cleaner.add_filter(filter_fit_status, "Fit Success")
            # cleaner.add_filter(filter_r2(0.85), "R2 > 0.85")
            cleaner.add_filter(filter_valid_contour(0.5), "Valid Contour > 50%")
            # cleaner.add_filter(filter_sigma_error(), "Sigma > Error")
            # cleaner.add_filter(filter_kappa_error(), "Kappa > Error")
            # Optional heavy filters
            # cleaner.add_filter(filter_image_focus(0.5, cleaner), "Focus Check")
            # cleaner.add_filter(filter_image_validation(0.5, cleaner), "Validation Check")

        cleaner.run(
            dry_run=args.dry_run,
            delete=args.delete,
            interactive=args.interactive,
            generate_training_data=args.generate_training_data,
            training_frames=args.training_frames,
            exclude_file=args.exclude_file,
            resume_from=args.resume_from,
            check_contrast=args.check_contrast,
            workers=args.workers,
            drop_keys=["autocorrelation_function"],
            auto_confirm=args.yes,
            load_last_cache=not args.no_cache,
        )
