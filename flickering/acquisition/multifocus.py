from __future__ import annotations
from typing import List, Union
from flickering.acquisition.autoimager import *
import os
import pickle
import sklearn
from flickering.acquisition.autoimager import *
from flickering.acquisition.autoimager import CellFindingExperiment
from flickering.acquisition.autofocus.RandomForestAutofocus import RandomForestAutofocus
from flickering.tracking.correlation_tracker import CorrelationContourTracker as CCT
import pandas as pd
import warnings
from flickering.acquisition.microscope import Microscope
from scipy.interpolate import LinearNDInterpolator
from time import time
import cv2
from pathos.multiprocessing import ThreadPool as Pool
from pathos.multiprocessing import ProcessPool
from functools import partial

from itertools import product


class MultiFocus(RepeatsMultiWellCellFindingExperiment):
    """Run autofocus on entire FoV. Provides valid_cells and full cell process.
    Beware of MRO.

    Needs MultiCellAutofocus as preprocessor. This should probably be configured as autofocus_
    """

    def __init__(
        self,
        microscope: Microscope,
        wells: List,
        max_fields=0,
        reference_well=None,
        move_by=np.array([100, 100]),
        contour_tracker=None,
        data_folder="./",
        repeats_n=1000,
        well_rotation=0,
    ):
        self._run_focus_on_valid = True
        self.repeats_search_multiple_zs = True
        self.allow_no_fit = False
        self.min_valid_cells = (
            2  # skip FoV if it looks like it has fever than this number of cells
        )
        self.post_focus_min_cells = None  # UNTESTED!
        super().__init__(
            microscope,
            wells,
            max_fields,
            reference_well,
            move_by,
            contour_tracker,
            data_folder,
            repeats_n,
            well_rotation=well_rotation,
        )

    def valid_cells(self, image, debug_index=0, return_preview=False):
        try:
            if return_preview:
                cells, _, preview = super().valid_cells(
                    image, debug_index, return_preview=return_preview
                )
            else:
                cells, _ = super().valid_cells(
                    image, debug_index, return_preview=return_preview
                )

            self.cell_preprocessor: MultiCellAutofocus
            if self._run_focus_on_valid:
                if len(cells) < self.min_valid_cells:
                    self.logger.info(
                        f"Field of view appears to have {len(cells)}, rejecting"
                    )
                    return [], []
                cell_data = self.cell_preprocessor.run_multicell_focus(cells)
                if self.post_focus_min_cells is not None:
                    count = 0
                    for cell in cell_data:
                        if cell["valid_on_best"]:
                            count += 1
                    if count < self.post_focus_min_cells:
                        self.logger.info(
                            f"Field of view appears to have {count} valid cells after focus, rejecting"
                        )
                        return [], []
            else:
                cell_data = None
            if return_preview:
                return cells, cell_data, preview
            return cells, cell_data
        except Exception as e:
            self.logger.warning("Cell finding/focus failed with exception:", exc_info=e)
            if return_preview:
                return [], [], None
            return [], []

    def full_cell_process(self, cell, well_name, well_field, cell_id, cell_extras={}):
        if cell_extras and "mf_cell" in cell_extras:
            if (
                "mf_fit_position" in cell_extras
                and cell_extras["mf_fit_position"] is not None
            ):
                best_focus = cell_extras["mf_fit_position"]
            elif (
                "mf_max_position" in cell_extras
                and cell_extras["mf_max_position"] is not None
            ):
                best_focus = cell_extras["mf_max_position"]
                self.logger.warning("No fit result found for cell, using max position")
                found = False
                for p, state in zip(
                    cell_extras["mf_pfs_offsets"], cell_extras["mf_focus_states"]
                ):
                    if p == best_focus:
                        if state and self.allow_no_fit:
                            self.logger.info(
                                "Cell was deemed in focus at this position, continueing"
                            )
                            found = True
                            break
                        else:
                            self.logger.warning(
                                "Cell was not in focus at this position, skipping"
                            )
                            info = self.generate_base_cell_info(
                                cell, well_name, well_field, cell_id, cell_extras
                            )
                            info["autofocus-status"] = "preprocesor-fail"
                            return False, info
                if not found:
                    self.logger.warning(
                        "Could not verify expected focus state, skippin cell"
                    )
                    info = self.generate_base_cell_info(
                        cell, well_name, well_field, cell_id, cell_extras
                    )
                    info["autofocus-status"] = "preprocesor-fail"
                    return False, info
            else:
                # no focus data, probably exception in validation/focus metrics etc.
                self.logger.warning("Skipping cell due to missing focus data")
                info = self.generate_base_cell_info(
                    cell, well_name, well_field, cell_id, cell_extras
                )
                info["autofocus-status"] = "preprocesor-fail"
                return False, info

            best_focus += self.cell_preprocessor.shift_from_max
            best_focus = int(best_focus)
            self.logger.info(f"Moving PFS to {best_focus} based on autofocus result")
            self.microscope.move_pfs(best_focus, True)
            sleep(2 * self.cell_preprocessor.large_move_delay)  # bit hacky
        else:
            self.logger.warning(
                f"Missing cell focus data, this is a bug! {cell_extras}"
            )
        # return back should be handled between full FoVs
        return super().full_cell_process(
            cell, well_name, well_field, cell_id, cell_extras
        )

    def find_matching_cells(self, cells, relevant_offsets, key, retries=2):
        if self.repeats_search_multiple_zs:
            self._run_focus_on_valid = False
            cells_found = super().find_matching_cells(
                cells, relevant_offsets, key, retries
            )
            self._run_focus_on_valid = True
            cells_list = []
            for cell_id, c in cells_found.items():
                cells_list.append(c["_last_match"])
            cell_data = self.cell_preprocessor.run_multicell_focus(cells_list)
            i = 0
            for k, v in cells_found.items():
                cells_found[k] = cells_found[k] | cell_data[i]
                cells_found[k]["extras"] = cell_data[i]
                i += 1

            return cells_found
        else:
            cells_found = {}
            offset = np.array(list(relevant_offsets)).mean()
            self.microscope.move_pfs(int(offset), True)

            sleep(self.delay_pre_record / 3)  # TODO
            image = self.microscope.get_image()
            vcs, extra_infos = self.valid_cells(
                image, f"{key}-{self.repeat_index}-{offset}"
            )
            for cell in cells:
                if cell["cell_id"] in cells_found:
                    continue

                matched_cell = self.find_nearest_cell(
                    vcs, cell["center_position"], self.max_cell_shift_cell_finding
                )  # TODO: configurable max distance
                if matched_cell is not None:
                    # this sucks
                    for vc, extra in zip(vcs, extra_infos):
                        if np.allclose(vc, matched_cell):
                            cell["extras"] = extra
                            break
                    cell["_last_match"] = matched_cell
                    cells_found[cell["cell_id"]] = cell

        return cells_found


class MultiCellAutofocus(RandomForestAutofocus):
    def __init__(self, experiment: CellFindingExperiment, threshold=0.6, threads=8):
        self.threads = threads
        self.skip_post_check = True  # completely skip final checks, use best focus position from initial processing
        self.ignore_post_result = True  # record cell even if final result is false, this can still recenter cell
        self.use_temp_file = False
        super().__init__(experiment, threshold)

    def run_multicell_focus(self, cells):
        # TODO: a bit of code duplication here
        start_pos = self.experiment.microscope.get_pfs_offset()
        min_point = start_pos - (self.steps - 1) / 2.0 * self.step_size
        max_point = start_pos + (self.steps - 1) / 2.0 * self.step_size
        positions = np.linspace(min_point, max_point, self.steps)

        prep_frames = []

        def analyse_cell(cell_position_index, frame=None):
            cell_id, position_id = cell_position_index
            try:
                if frame is None:
                    frame = prep_frames[position_id]
                cell = cells[cell_id]
                center_location = cell[0:2]
                offset = self.experiment.adjust_offset_to_limits(
                    (center_location - self.experiment.frame_size // 2).astype(int)
                )
                cell_img = frame[
                    offset[0] : offset[0] + self.experiment.frame_size[0],
                    offset[1] : offset[1] + self.experiment.frame_size[1],
                ]
                cell_img = CCT.normalise_image_values(cell_img)
                in_frame_center = center_location - offset
                focus_state, metrics = self.determine_abs_focus(
                    cell_img, center=in_frame_center, radius=cell[2]
                )
                metric_value = metrics[0]
            except Exception as e:
                self.logger.warning("Cell analysis failed with exception: ", exc_info=e)
                return cell_id, position_id, None, None
            return cell_id, position_id, metric_value, (focus_state, metrics)

        def validate_cell_at_best(frame, cell):
            center_location = cell[0:2]
            offset = self.experiment.adjust_offset_to_limits(
                (center_location - self.experiment.frame_size // 2).astype(int)
            )
            cell_img = frame[
                offset[0] : offset[0] + self.experiment.frame_size[0],
                offset[1] : offset[1] + self.experiment.frame_size[1],
            ]
            cell_img = CCT.normalise_image_values(cell_img)
            return self.experiment.validate_cell(
                cell_img, True
            )  # TODO: should we do final here?

        if self.use_temp_file:
            script_send, script_return, move_start = self.record_temp_file(positions)

            movie = Movie(self.temp_filename + ".movie")
            # we need to cut-up the image and process each region individually
            if movie.n_frames != len(positions):
                self.logger.error(
                    f"Recorded file frames ({movie.n_frames}) != steps ({self.steps}), this might cause focus failure"
                )
                if movie.n_frames > 100:
                    return False, []

            # should this be part of the multiprocessing fn instead?
            for f in movie.frames():
                prep_frames.append(f)
            movie_read = time()

            # TODO: multithread fitting step?
            with Pool(self.threads) as p:
                results = p.map(
                    analyse_cell,
                    product(list(range(len(cells))), list(range(len(positions)))),
                )

        else:
            results = []
            result_objects = []
            script_send = time()
            with Pool(self.threads) as p:
                self.experiment.microscope.move_pfs(positions[0], absolute=True)
                sleep(self.large_move_delay)  # large move
                for i, offset in enumerate(positions):
                    self.experiment.microscope.move_pfs(offset, absolute=True)
                    sleep(self.move_delay)
                    frame = self.experiment.microscope.get_image()
                    prep_frames.append(frame)
                    analyse_frame = partial(analyse_cell, frame=frame)
                    result_objects.append(
                        p.map_async(
                            analyse_frame, product(list(range(len(cells))), [i])
                        )
                    )

                script_return = time()

                move_start = time()
                self.experiment.microscope.move_pfs(start_pos, True)

                for ro in result_objects:
                    results += ro.get()
                movie_read = time()

        cell_data = []
        for cell in cells:
            cell_data.append(
                {
                    "mf_cell": cell,
                    "mf_metric_values": [],
                    "mf_pfs_offsets": [],
                    "mf_focus_states": [],
                }
            )

        for r in results:
            cell_id, position_id, metric_value, focus_state = r
            if metric_value is None:
                continue
            cell_data[cell_id]["mf_metric_values"].append(metric_value)
            cell_data[cell_id]["mf_focus_states"].append(focus_state)
            cell_data[cell_id]["mf_pfs_offsets"].append(positions[position_id])

        for cd in cell_data:
            if not "mf_pfs_offsets" in cd:
                continue
            max_position, fit_position = self.focus_scores_to_offset(
                cd["mf_pfs_offsets"], cd["mf_metric_values"]
            )
            if fit_position:
                cd["mf_fit_position"] = fit_position
            cd["mf_max_position"] = max_position
            cd["mf_max_index"] = cd["mf_pfs_offsets"].index(max_position)

        # TODO: run validation on mf_max_position images
        if self.experiment.post_focus_min_cells is not None:
            for cell in cell_data:
                if "mf_fit_position" in cell and cell["mf_fit_position"] is not None:
                    cell["valid_on_best"] = validate_cell_at_best(
                        prep_frames[cell["mf_max_index"]], cell["mf_cell"]
                    )
                    cell["valid_on_best"] = True
                else:
                    cell["valid_on_best"] = False

        comp = time()
        self.logger.info(
            f"Focus timings: script:{script_return-script_send:.2f}s,ms:{move_start-script_return:.2f}s,read:{movie_read-script_return:.2f}s,post:{comp-movie_read:.2f}s"
        )

        return cell_data

    def run(self):
        if self.skip_post_check:
            if self.readjust_cell_camera:
                info = {}
                af_start = time()
                test_image = CCT.normalise_image_values(
                    self.experiment.microscope.get_image()
                )
                res, info, test_image = self.readjust_camera(test_image, info, af_start)
                if not res:
                    return False, info

            if self.experiment.cell_validation:
                if not self.readjust_cell_camera:
                    test_image = CCT.normalise_image_values(
                        self.experiment.microscope.get_image()
                    )
                is_valid = self.experiment.validate_cell(test_image, final=True)
            return True, {"autofocus-status": "post-disabled"}

        self.disable_autofocus = True
        is_valid, info = super().run()

        if self.ignore_post_result and not is_valid:
            self.logger.warning(
                "Cell not in focus or not valid after multifocus, recording anyway"
            )
            info["autofocus-status"] = "overridden"
            return True, info

        return is_valid, info
