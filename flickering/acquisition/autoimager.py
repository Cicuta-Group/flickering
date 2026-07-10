from __future__ import annotations
import numpy as np
import scipy
import cv2
import logging
from time import sleep, time
import flickering.utils.visualisation
from flickering.utils.movie_reader import get_movie_reader
from flickering.acquisition.microscope import Microscope
from copy import deepcopy
from tqdm.auto import tqdm
from flickering.utils.encoder import MultiJsonEncoder
from typing import List, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from TemikaXML.SamplePlatform import Pad

# from flickering.utils.visualisation import *
from flickering.tracking.correlation_tracker import CorrelationContourTracker as CCT
from scipy.optimize import curve_fit

# logging.basicConfig(
#    filename="/dev/stderr",
#    encoding="utf-8",
#    level=logging.INFO,
#    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
# )
from typing import List, Union
from flickering.acquisition.spiral import SpiralMoves
import itertools
from numpyencoder import NumpyEncoder
from flickering.utils.standard_configs import default_tracker


# TODO: add json output+json experiment config data
class CellFindingExperiment:

    def __init__(
        self, microscope: Microscope, contour_tracker=None, data_folder="./"
    ):
        if contour_tracker is None:
            # initialise default contour tracker
            contour_tracker = default_tracker()
            contour_tracker.mask_refinement = True
            contour_tracker.refine_correlation = True
            contour_tracker.hough_param_1 = 50
            contour_tracker.hough_param_2 = 33
            contour_tracker.hough_min_r = 40
            contour_tracker.hough_max_r = 100
            contour_tracker.hough_min_d = 100
            contour_tracker.mask_width = 17  # select this many pixels around the x direction max to correlate with
            contour_tracker.correlate_width = (
                80  # CORRELATE_WIDTH": 150,  # maximum deviation
            )
            contour_tracker.ignore_center_rad = 0

        self.data_folder = data_folder
        self.microscope = microscope
        self.contour_tracker = contour_tracker

        self.display = False
        self.debug = False

        self.correct_drift = False
        self.frame_size = np.array([256, 256], dtype=int)
        self.cell_preprocessor = None
        if microscope is not None:
            self.analysis_offset = microscope.get_pfs_offset()
        self.cell_validation = True
        self.analysis_shutter = 0.0014  # seconds
        self.cell_log = []
        #        self.repeats_log = []

        self.delay_pre_record = 0.5  # seconds
        self.timeout = None
        self.interwell_speed = 2500.0  # speed in um/s for inter well moves. Typical: 10000 (fastest), 1111 (slowest)

        # TODO: how to handle this with repeats?
        self.statistics = {}
        self.last_statistics = np.array(
            [0, 0, 0, 0, 0]
        )  # all, close, messy, invalid, accepted

        self.thresholds = {
            "CELL_MIN_SEPARATION": 50 / 2,
            "CELL_CLEAR_AREA_FROM": 5,
            "CELL_CLEAR_AREA_WIDTH": 10,  # width of the annulus to check for variation
            "CELL_CLEAR_AREA_2_98_VAR": 0.5,  # maximum (98percentile-2percentile)/mean in the clear annulus
            "CELL_MIN_CONTRAST": 0.05,
        }

        self.description = ""
        self.info = {}

        self.logger = logging.getLogger("EXPERIMENT")
        contour_tracker.logger.setLevel(logging.ERROR)
        contour_tracker.use_fixed_mask = False

    def valid_cells(self, image, debug_index=0, return_preview=False):
        """
        Find useable cells and return their positions in images TODO
        returns positions in pixel coordinates in image

        Args:
            image (np.array): image to process
        """
        self.last_statistics = np.array(
            [0, 0, 0, 0, 0]
        )  # all, close, messy, invalid, accepted
        # image = Image.fromarray(image.astype(np.uint8))
        image = CCT.normalise_image_values(image)
        denoised = cv2.fastNlMeansDenoising(
            (image * 255.0).astype(np.uint8), None, 10, 21, 5
        )
        image8 = cv2.normalize(
            denoised, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )  # image.astype(np.uint8) #cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)# (normalise_image_values(image)*255.0).astype(np.uint8
        # plt.imshow(((image/255.0).astype(np.float64)))
        # plt.show()
        # print(image8.min())
        # print(image8.max())
        # image8 = image8.astype(np.uint8)
        disp = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR)
        # hough params are in correlation contour
        # TODO: clean up the configuration

        circles = cv2.HoughCircles(
            image8,
            cv2.HOUGH_GRADIENT,
            self.contour_tracker.hough_dp,
            self.contour_tracker.hough_min_d,
            param1=self.contour_tracker.hough_param_1,
            param2=self.contour_tracker.hough_param_2,
            minRadius=self.contour_tracker.hough_min_r,
            maxRadius=self.contour_tracker.hough_max_r,
        )
        cells = []
        if circles is None:
            self.logger.debug(f"No circles detected in frame {debug_index}")
            if self.display and self.debug:
                cv2.imshow("microscope_view", disp)
                cv2.waitKey(0)
            cv2.imwrite(
                f"{self.data_folder}/detected_cells-{debug_index}.jpeg", disp
            )
            if return_preview:
                return [], None, disp
            return [], None

        # cells appear to x,y order here!
        for i in circles[0, :]:
            # remove cells too close to image edge
            if (
                i[0] < self.frame_size[0] / 2
                or i[1] < self.frame_size[1] / 2
                or image.shape[1] - i[0] < self.frame_size[0] / 2
                or image.shape[0] - i[1] < self.frame_size[1] / 2
            ):
                continue
            cells.append([i[0], i[1], i[2]])

        self.last_statistics[0] = len(cells)

        # we need to remove cells which are too close
        # try the naive method first, might be fine
        separated_cells = []
        cells_too_close = []
        cells = np.array(cells)
        for cell in cells:
            is_clean = True
            for cell2 in cells:
                if np.array_equal(cell2, cell):
                    continue
                min_d = np.linalg.norm(cell[0:2] - cell2[0:2]) - cell2[2] - cell[2]
                if min_d < self.thresholds["CELL_MIN_SEPARATION"]:
                    is_clean = False
                    self.logger.debug(f"Cells too close: {min_d}")
                    break
                self.logger.debug(f"Cell min_d:{min_d}")
            if is_clean:
                separated_cells.append(cell)
            else:
                cells_too_close.append(cell)

        self.last_statistics[1] = len(cells_too_close)
        self.draw_cells_on_image(disp, cells_too_close, True, (0, 0, 255))

        clean_cells = []
        unclean_cells = []
        # now we check the area around the cells for any issues
        for cell in separated_cells:
            # these cells are not close to the edge
            # this is the same as below, might want to merge the loops
            # we only check small images around the cell
            cell_img = image[
                int(cell[1])
                - self.frame_size[0] // 2 : int(cell[1])
                + self.frame_size[0] // 2,
                int(cell[0])
                - self.frame_size[1] // 2 : int(cell[0])
                + self.frame_size[1] // 2,
            ]
            xs = np.arange(cell_img.shape[1])
            ys = np.arange(cell_img.shape[0])
            dist = (xs[np.newaxis, :] - self.frame_size[0] // 2) ** 2 + (
                ys[:, np.newaxis] - self.frame_size[1] // 2
            ) ** 2
            mask = np.logical_and(
                dist > np.power(cell[2] + self.thresholds["CELL_CLEAR_AREA_FROM"], 2),
                dist
                <= np.power(
                    cell[2]
                    + self.thresholds["CELL_CLEAR_AREA_WIDTH"]
                    + self.thresholds["CELL_CLEAR_AREA_FROM"],
                    2,
                ),
            )
            values = np.reshape(cell_img[mask], (-1))
            p10 = np.percentile(values, 2)
            p90 = np.percentile(values, 98)
            mean = np.average(values)
            var = (p90 - p10) / mean
            if var > self.thresholds["CELL_CLEAR_AREA_2_98_VAR"]:
                self.logger.debug(f"Excessive variance near cell: {var}")
                unclean_cells.append(cell)
            else:
                clean_cells.append(cell)

        self.last_statistics[2] = len(unclean_cells)
        self.draw_cells_on_image(disp, unclean_cells, True, (255, 165, 0))

        validated_cells = []
        for cell in clean_cells:
            cell_img = image[
                int(cell[1])
                - self.frame_size[0] // 2 : int(cell[1])
                + self.frame_size[0] // 2,
                int(cell[0])
                - self.frame_size[1] // 2 : int(cell[0])
                + self.frame_size[1] // 2,
            ]
            center = None
            contour = None
            try:
                center, contour = self.contour_tracker.get_contour(
                    cell_img,
                    center=np.array([self.frame_size[0] // 2, self.frame_size[1] // 2]),
                    radius=cell[2],
                )
            except Exception as e:
                self.logger.debug("Contour tracking failed:", exc_info=e)

            if (
                center is not None
                and contour is not None
                and self.validate_cell(cell_img, center=center, contour=contour)
            ):
                flickering.utils.visualisation.draw_contour(
                    disp, contour, cell[0:2], img_value=(0, 255, 0), display=False
                )
                validated_cells.append(cell)
            elif contour is not None and len(contour) > 0:
                flickering.utils.visualisation.draw_contour(
                    disp,
                    contour,
                    cell[0:2]
                    + center
                    - np.array([self.frame_size[0] // 2, self.frame_size[1] // 2]),
                    img_value=(255, 0, 0),
                    display=False,
                )
            else:
                self.logger.debug("No contour detected")

        self.last_statistics[3] = len(clean_cells) - len(validated_cells)
        self.last_statistics[4] = len(validated_cells)

        # self.draw_cells_on_image(disp, validated_cells, True, (0, 255, 0))

        # disp = cv2.resize(disp, (disp.shape[1] // 2, disp.shape[0] // 2))
        if self.display:
            cv2.imshow("detected_cells", disp)
            if self.debug:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        cv2.imwrite(
            f"{self.data_folder}/detected_cells-{debug_index}.jpeg", disp
        )  # TODO this needs field ID and Run id to not overwrite
        # cv2.destroyAllWindows()

        # REORDERING x,y => y,x here!
        final_cells = []
        for c in validated_cells:
            final_cells.append([c[1], c[0], c[2]])
        if return_preview:
            return np.array(final_cells), None, disp

        return np.array(final_cells), None

    def draw_cells_on_image(self, image, cells, reorder=True, colour=(0, 255, 0)):
        final_cells = []
        for c in cells:  # clean_cells
            final_cells.append([c[1], c[0], c[2]])

        if reorder:
            cells = np.uint16(np.around(final_cells))
        else:
            cells = np.uint16(np.around(cells))

        for i in cells:
            # draw the outer circle
            cv2.circle(image, (i[1], i[0]), i[2], colour, 2)
            # draw the center of the circle
            cv2.circle(image, (i[1], i[0]), 2, colour, 3)

    def validate_cell(self, image, final=False, center=None, contour=None):
        """After moving the stage and reducing image size, checks if contour can be detected in the image
        (quality control step in Guil's implementation)

        Args:
            image (np.array): image of the cell to check for contour
        """
        normalised_image = image
        rms_contrast = normalised_image.std()
        if rms_contrast < self.thresholds["CELL_MIN_CONTRAST"]:
            self.logger.info(f"Cell has insufficient contrast: {rms_contrast:.3f}")
            return False
        try:
            if center is None or contour is None:
                center, contour = (
                    self.contour_tracker.get_contour(  # TODO: Pass center from above?
                        image  # , [self.frame_size[0] // 2, self.frame_size[1] // 2]
                    )
                )
            distance_from_center = np.linalg.norm(
                np.array(center)
                - np.array([self.frame_size[0] // 2, self.frame_size[1] // 2])
            )
            if distance_from_center > 20:
                self.logger.warning(
                    f"Large distance from video center: {distance_from_center:.1f} px"
                )
        except Exception as e:
            self.logger.debug("Contour tracking failed:", exc_info=e)

        return self.contour_tracker.validate_contour(contour)

    def adjust_offset_to_limits(self, offset):
        max_offset = self.microscope.max_image_size() - self.frame_size
        min_offset = np.array([0, 0])
        offset = np.minimum(max_offset, offset)
        offset = np.maximum(min_offset, offset)

        return offset

    def camera_cell(self, cell):
        center_location = cell[0:2]
        offset = self.adjust_offset_to_limits(
            (center_location - self.frame_size // 2).astype(int)
        )
        camera_settings = {
            "image_size": self.frame_size,
            "image_offset": offset,
            "shutter": self.analysis_shutter,  # TODO make this easier to configure
        }
        self.microscope.configure_camera(camera_settings)
        return (
            offset,
            np.array(center_location) - offset,
        )  # center location in smaller image

    def camera_full(self):
        self.microscope.restore_camera_defaults()

    def run_at_cell(self, global_id):
        """
        Implement this to record cell,
        will be run once cell preprocessing finishes, if it is successful.

        camera is already configured to record frame_size around cell
        """
        pass

    # try to recover PFS lock
    def attempt_pfs_recovery(self, step_size, steps):
        if self.microscope.pfs_focus(True):
            if self.microscope.set_pfs(True):
                return True

        self.microscope.move_z(
            -steps // 2 * step_size, wait=False
        )  # doesn't wait for XY
        for step in range(steps):
            if self.microscope.pfs_focus(update=True):
                self.microscope.set_pfs(True)
                sleep(0.5)
                if self.microscope.pfs_focus(update=True):
                    self.logger.info(
                        f"Focus recovered with a move of {(-steps//2+step)*step_size}um"
                    )
                    return True
            self.microscope.move_z(step_size)

        if not self.microscope.pfs_focus(True):  # return to start
            self.microscope.move_z(-steps // 2 * step_size)

        return self.microscope.pfs_focus(update=True)

    def find_nearest_cell(self, valid_cells, cell, max_dist=150):
        current_best = None
        for vc in valid_cells:
            dist = np.linalg.norm(vc[:2] - cell[:2])
            if dist < max_dist:
                current_best = vc
                max_dist = dist

        if current_best is None:
            self.logger.debug("Failed to find cell")
        else:
            self.logger.info(f"Best match cell with distance {max_dist}")

        return current_best

    def get_configuration_dict(self):
        config_dict = {"class_name": str(self.__class__), "time": time()}
        config_dict = config_dict | self.__dict__
        if self.cell_preprocessor is not None:
            config_dict = config_dict | self.cell_preprocessor.get_configuration_dict()

        for entry in ["microscope", "logger", "cell_preprocessor", "contour_tracker"]:
            del config_dict[entry]
        return config_dict

    def write_log(self):
        # this fails to serialize lists
        with open(f"{self.data_folder}/{time():.0f}.json", "w") as logfile:
            logfile.write(
                json.dumps(
                    {
                        "cells": self.cell_log,
                        "metadata": self.get_configuration_dict(),
                        "description": self.description,
                        "info": self.info,
                    },
                    cls=MultiJsonEncoder,  # ,
                    # default=lambda o: f"<not serializable: {str(type(o))}>" #this can cause missing info
                )
            )


class BaseMultiwellCellFindingExperiment(CellFindingExperiment):
    def __init__(
        self,
        microscope: Microscope,
        wells: List[Union[Pad, np.ndarray]],
        max_fields=0,
        reference_well: Union[Pad, np.ndarray, None] = None,
        move_by=np.array([100, 100]),
        contour_tracker=None,
        data_folder="./",
        well_rotation=0,
    ):
        super().__init__(microscope, contour_tracker, data_folder)
        if microscope is not None:
            self.reference_position = microscope.get_stage_position()
        if reference_well is None and len(wells) > 0:
            self.reference_well = wells[0]
        else:
            self.reference_well = reference_well
        self.max_fields = max_fields
        self.move_by = move_by
        self.spirals: List[SpiralMoves] = []
        self.well_centers = []
        self.cells_per_well = 100  # total number of cells to find in each well
        self.max_dwell_time = (
            0  # Maximum time to spend on a single well, 0 for unlimited
        )
        self.wells = wells
        self.pfs_off_between_wells = False
        self.well_z_positions = {}  # last successful z positions per well

        for i, pad in enumerate(wells):
            p_center = getattr(pad, "pad_center", pad)
            p_name = getattr(pad, "name", f"well_{i}")
            ref_center = getattr(self.reference_well, "pad_center", self.reference_well)

            pad_center = (
                np.array(p_center, dtype=float)
                - np.array(ref_center)
                + self.reference_position
            )
            self.well_centers.append(pad_center)  # potentially useful?
            self.spirals.append(
                SpiralMoves(
                    move_by,
                    max_fields,
                    pad_center,
                    True,
                    name=p_name,
                    rotation=well_rotation,
                )
            )
            self.well_z_positions[p_name] = None
            self.statistics[p_name] = self.last_statistics.copy()

        self.use_progressbars = True
        self.edge_well_time_multiplier = 2  # how much longer to spend on first and last well in pattern (going back and forth otherwise spends longer on middle)
        # self.well_progressbars = None
        # self.overall_pbar = None
        self.start_z_position = None
        self.well_analysis_offsets = None
        # current cell being processed
        self.cell_id = 0
        self.cell_log = (
            []
        )  # we want to log info about cells to potentially revisit them

    def check_center_positions(self, wait_input=False, wait_time=5, get_zs=True):
        for i, spiral in enumerate(self.spirals):
            print(f"Move to {spiral.position}")
            self.microscope.move_stage(spiral.position, absolute=True, speed=10000.0)
            self.logger.info(f"At {spiral.name}")
            if wait_input:
                input(f"At {spiral.name}, waiting for input")
            else:
                sleep(wait_time)
                if not self.microscope.pfs_focus(True):
                    print("No pfs focus, adjust z position")
                    while not self.microscope.pfs_focus(True):
                        sleep(0.2)
                    sleep(0.5)
                    while not self.microscope.pfs_focus(True):
                        sleep(0.2)
                    # wait for the PFS move to settle
                    sleep(1)
                self.logger.info(f"PFS lock at {self.microscope._z} um")
            self.well_z_positions[spiral.name] = self.microscope._z
            # TODO: should we adjust offsets too (in the manual mode)

        self.microscope.move_stage(
            self.spirals[0].position, absolute=True, speed=10000.0
        )
        self.logger.info(f"At {self.spirals[0].name}")

    def process_cell(self, global_id, cell_info={}):
        preprocessor_result = False
        preprocessor_data = {}
        pre_cell_pfs = self.microscope.get_pfs_offset()
        if self.cell_preprocessor is not None:
            try:
                preprocessor_result, preprocessor_data = (
                    self.cell_preprocessor.run()
                )  # this has microscope
            except Exception as e:
                self.logger.error(f"Preprocessor exception for {global_id}", exc_info=e)
        if not preprocessor_result:
            self.logger.warning(f"Cell {global_id} rejected by preprocessor result")
            preprocessor_data["result"] = "preprocesor-fail"
            return False, preprocessor_data

        run_result = False
        run_data = {}
        try:
            run_result, run_data = self.run_at_cell(global_id)
        except Exception as e:
            self.logger.error(f"Exception processing {global_id}", exc_info=e)

        # TODO: should we return to the pre-focus state?
        # if we stay there is a higher chance of other cells being in focus
        # but if we fail we can drift completely out
        # for now moving back
        self.microscope.move_pfs(pre_cell_pfs, True)

        if not run_result:
            run_data["result"] = "run-fail"
            self.logger.warning(f"Cell {global_id} failed processing")

        run_data["result"] = "success"
        run_data["finish_time"] = time()
        return run_result, preprocessor_data | run_data

    def generate_global_id(self, well_name, well_field, cell_n):
        return f"{well_name}-{cell_n}"

    def generate_base_cell_info(
        self, cell, well_name, well_field, cell_id, cell_extras={}
    ):
        global_id = self.generate_global_id(well_name, well_field, cell_id)
        cell_info = {
            "well": well_name,
            "field": well_field,
            "stage_position": self.microscope.get_stage_position(False),
            "center_position": cell[0:2],
            "radius": cell[2],
            "global_id": global_id,
            "cell_id": cell_id,
            "pfs_offset": self.microscope.get_pfs_offset(False),
            "start_time": time(),
            "z_position": self.microscope._z,
        }

        return cell_info

    def full_cell_process(self, cell, well_name, well_field, cell_id, cell_extras={}):
        cell_info = self.generate_base_cell_info(
            cell, well_name, well_field, cell_id, cell_extras
        )
        offset, in_image_center_position = self.camera_cell(cell)
        cell_info["image_offset"] = offset
        cell_info["video_center_position"] = in_image_center_position

        global_id = cell_info["global_id"]
        cell_info = {
            "well": well_name,
            "field": well_field,
            "stage_position": self.microscope.get_stage_position(False),
            "image_offset": offset,
            "center_position": cell[0:2],
            "video_center_position": in_image_center_position,
            "radius": cell[2],
            "global_id": global_id,
            "cell_id": cell_id,
            "pfs_offset": self.microscope.get_pfs_offset(False),
            "start_time": time(),
            "z_position": self.microscope._z,
        }

        cell_info = cell_info | cell_extras
        process_cell_result, process_cell_data = self.process_cell(global_id, cell_info)
        cell_info = cell_info | process_cell_data
        if process_cell_result is not False:
            self.on_cell_success(cell_info)
            return True, cell_info

        return False, cell_info

    def update_offset_to_well(self, well_name):
        if (
            self.well_analysis_offsets is None
            or well_name not in self.well_analysis_offsets
        ):
            return
        self.analysis_offset = self.well_analysis_offsets[well_name]

    def get_fov_id(self, spiral):
        return f"{spiral.name}-{spiral.index}"

    def full_pfs_recovery(self, planes={}):
        failpoint_z = self.microscope._z
        self.microscope.set_pfs(True)
        if self.microscope.pfs_focus(update=True) and (
            sleep(5) or self.microscope.pfs_focus(update=True)
        ):
            self.logger.info(f"Focus recovered")
            return True

        # just try the planes without anything fancy
        for name, plane in planes.items():
            if plane is None:
                continue
            if np.abs(failpoint_z - plane) > 2000:
                self.logger.warning(
                    f"Large shift from initial position: {failpoint_z - self.start_z_position} um"
                )
                if (
                    failpoint_z > self.start_z_position
                    and failpoint_z - self.start_z_position > 2000
                ):
                    raise NotImplementedError(
                        "Recovery from this position (2mm above start) is too dangerous!"
                    )

            self.logger.info(f"Trying {name} ({plane-failpoint_z} um)")
            self.microscope.move_z(plane, True)
            sleep(0.4)
            self.microscope.set_pfs(True)  # this shouldn't be needed?
            sleep(0.4)
            if self.microscope.pfs_focus(update=True):
                recovered_at = self.microscope._z
                self.logger.info(
                    f"Focus recovered at {name} ({recovered_at-failpoint_z} um shift)"
                )
                return True

        for name, plane in planes.items():
            if plane is None:
                continue

            self.microscope.move_z(plane, True)
            if self.attempt_pfs_recovery(10, 30):
                recovered_at = self.microscope._z
                self.logger.info(
                    f"Focus recovered at {name} ({recovered_at-failpoint_z} um shift)"
                )
                return True

        self.logger.warning(f"Standard PFS recovery failed, attempting larger z-range")
        for name, plane in planes.items():
            if plane is None:
                continue

            self.microscope.move_z(plane, True)
            if self.attempt_pfs_recovery(15, 50):  # a bit scary
                recovered_at = self.microscope._z
                self.logger.info(
                    f"Focus recovered at {recovered_at} ({plane-failpoint_z} um shift)"
                )
                return True

        self.logger.error(
            f"PFS lock irrecoverable!"
        )  # TODO should this raise an exception?

        return False

    def run(self):
        overall_pbar = None
        well_progressbars = {}
        self.start_z_position = self.microscope._z

        start_time = time()
        if self.timeout is not None:
            self.timeout_time = start_time + self.timeout
        else:
            self.timeout_time = None
        cell_count = 0

        finished_wells = set()
        spiral_n = 0
        total_fields = 0
        direction = -1
        direction_changed = True
        if len(self.spirals) == 1:
            direction = 0

        if self.use_progressbars:
            overall_pbar = tqdm(
                desc="Total", total=len(self.spirals) * self.cells_per_well
            )
            for spiral in self.spirals:
                well_progressbars[spiral.name] = tqdm(
                    desc=spiral.name, total=self.cells_per_well
                )

        # self.well_progressbars = well_progressbars
        # self.overall_pbar = overall_pbar
        last_spiral = None
        while len(finished_wells) < len(self.spirals):
            if self.abort_requested:
                self.logger.info("Experiment aborted by user")
                break
            # going back and forth
            if spiral_n == len(self.spirals) - 1 or spiral_n == 0:
                direction *= -1
                direction_changed = True
            else:
                direction_changed = False

            spiral = self.spirals[spiral_n]
            self.update_offset_to_well(spiral.name)
            # applied in next loop
            spiral_n += direction

            if (
                spiral.processed_cells >= self.cells_per_well
                or spiral.field >= spiral.max_fields
            ):  # TODO possible off-by-one
                finished_wells.add(spiral.name)
                continue

            self.logger.info(f"Moving to well {spiral.name}")

            self.well_start_time = time()
            well_start_time = self.well_start_time
            speed = self.interwell_speed
            restart_pfs = False
            if self.pfs_off_between_wells and total_fields > 0:
                self.microscope.set_pfs(False)
                restart_pfs = True

            if self.max_dwell_time > 0:
                if direction_changed:
                    well_timeout = (
                        well_start_time
                        + self.edge_well_time_multiplier * self.max_dwell_time
                    )
                else:
                    well_timeout = well_start_time + self.max_dwell_time
            else:
                well_timeout = well_start_time + 1e9

            for position in spiral:
                if self.abort_requested:
                    break
                if well_timeout < time():
                    self.logger.info("Reached well timeout, going to next well")
                    break
                self.pre_fov_change(last_spiral)
                if not self.microscope.move_stage(
                    position, True, speed=speed
                ):  # TODO speed? should be fast enough but avoid losing pfs
                    self.logger.error("Microscope move failed!")
                    break
                last_spiral = spiral
                speed = 3

                if restart_pfs:
                    self.microscope.set_pfs(True)

                self.microscope.update_status()
                if not self.microscope.pfs_focus():
                    self.logger.error("PFS LOST FOCUS!")
                    # small recovery range around the current position
                    if not self.full_pfs_recovery(
                        {
                            "well_pos": self.well_z_positions[spiral.name],
                            "start": self.start_z_position,
                        }
                    ):
                        self.logger.error(
                            f"PFS lock irrecoverable, skipping well"
                        )  # TODO should this raise an exception?
                        break  # this is a little scary
                        # self.close_progressbars(overall_pbar, well_progressbars)
                        return self.cell_log

                #                sleep(self.delay_pre_record)
                #                self.microscope.update_status()
                sleep(self.delay_pre_record)
                image = self.microscope.get_image()
                # cv2.waitKey(0)
                total_fields += 1
                vcs, all_cells_extras = self.valid_cells(image, self.get_fov_id(spiral))
                self.statistics[spiral.name] += self.last_statistics

                self.logger.info(f"Found {len(vcs)} valid cells")
                cell_offsets = []
                if self.timeout is not None and self.timeout_time < time():
                    self.logger.warning(f"Processing aborted by timeout")
                    self.close_progressbars(overall_pbar, well_progressbars)

                    return self.cell_log

                for cell_num_tmp in range(len(vcs)):
                    cell = vcs[cell_num_tmp]
                    if all_cells_extras and len(all_cells_extras) > cell_num_tmp:
                        cell_extra = all_cells_extras[cell_num_tmp]
                    else:
                        cell_extra = {}
                    process_result, cell_info = self.full_cell_process(
                        cell, spiral.name, spiral.field, self.cell_id, cell_extra
                    )
                    if cell_extra:
                        cell_info = cell_info | cell_extra
                    if process_result:
                        cell_offsets.append(
                            self.microscope.get_pfs_offset(update=False)
                        )
                        spiral.processed_cells += 1
                        cell_count += 1
                        cell_recorded_z = cell_info["z_position"]
                        if np.abs(cell_recorded_z - self.start_z_position) > 1000:
                            self.logger.warning(
                                f"Cell recorded >1mm from start positon {cell_recorded_z}"
                            )
                        else:
                            self.well_z_positions[spiral.name] = cell_recorded_z
                        if self.use_progressbars:
                            overall_pbar.update()
                            well_progressbars[spiral.name].update()
                    self.cell_log.append(cell_info)
                    self.cell_id += 1

                # TODO: different analysis position in each well?
                # TODO: this is untested
                # TODO: remove?
                # if there is a drift, the analysis position needs to be updated
                if self.correct_drift and len(cell_offsets) > 0:
                    new_analysis_offset = int(np.mean(cell_offsets))
                    max_shift = 30 * len(cell_offsets)  # used std instead for larger n?

                    if abs(new_analysis_offset - self.analysis_offset) > max_shift:
                        self.logger.info(f"Clipping analysis offset shift")
                        new_analysis_offset = np.clip(
                            new_analysis_offset,
                            self.analysis_offset - max_shift,
                            self.analysis_offset + max_shift,
                        )

                    self.logger.info(
                        f"Moving analysis offset by ({new_analysis_offset-self.analysis_offset})"
                    )
                    self.microscope.move_pfs(new_analysis_offset, absolute=True)
                    self.analysis_offset = new_analysis_offset

                self.microscope.move_pfs(self.analysis_offset, absolute=True)
                self.camera_full()
                if spiral.processed_cells >= self.cells_per_well:
                    break

        self.logger.info(f"Processed {cell_count} cells in {total_fields} fields")
        self.close_progressbars(overall_pbar, well_progressbars)

        return self.cell_log

    def on_cell_success(self, cell_info):
        pass

    def pre_fov_change(self, spiral):
        pass

    def close_progressbars(self, overall_pbar, well_progressbars):
        if self.use_progressbars:
            overall_pbar.close()
            for spiral in self.spirals:
                well_progressbars[spiral.name].close()

    def get_configuration_dict(self):
        full = super().get_configuration_dict()
        for entry in ["spirals", "reference_well", "wells"]:
            del full[entry]
        return full


class RepeatsMultiWellCellFindingExperiment(BaseMultiwellCellFindingExperiment):
    def __init__(
        self,
        microscope: Microscope,
        wells: List[Pad],
        max_fields=0,
        reference_well: Union[Pad, None] = None,
        move_by=np.array([100, 100]),
        contour_tracker=None,
        data_folder="./",
        repeats_n=1000,
        well_rotation=0,
    ):
        super().__init__(
            microscope,
            wells,
            max_fields,
            reference_well,
            move_by,
            contour_tracker,
            data_folder,
            well_rotation=well_rotation,
        )
        self.abort_requested = False
        self.repeats_n = repeats_n
        self.repeat_index = 0
        self.use_last_offset = 0  # average of this many last cell pfs offsets used instead of first offset, 0 for always use analysis_offset
        self.contour_tracker.logger.setLevel(
            logging.INFO
        )  # we expect contour tracking failures - it's part of the cell filter
        self.tracking = True
        self.max_cell_shift_cell_finding = 100

    def generate_global_id(self, well_name, well_field, cell_n):
        return f"{well_name}-{cell_n}-{self.repeat_index}"

    def get_fov_id(self, spiral):
        return super().get_fov_id(spiral) + f"-{self.repeat_index}"

    def continue_interrupted(self, logfile):
        # TODO find last logfile

        with open(logfile, "r") as f:
            last_run = json.load(f)
        self.cell_log = last_run["cells"]
        first_repeat_log = []
        max_repeat = 0
        for row in last_run["cells"]:
            repeat = int(row["global_id"].split("-")[-1])
            if repeat == 0:
                first_repeat_log.append(row)

            if repeat > max_repeat:
                max_repeat = repeat

        self.repeat_index = max_repeat + 1
        self.start_z_position = self.microscope._z  # TODO: reuse?

        start_time = time()
        if self.timeout is not None:
            self.timeout_time = start_time + self.timeout
        else:
            self.timeout_time = None

        self.logger.info(
            f"Resuming run @repeat {max_repeat+1} with {len(first_repeat_log)} cells"
        )
        new_log = self.run_repeats(first_repeat_log)
        return last_run["cells"] + new_log

    def run(self):
        start_position = self.microscope.get_stage_position()
        start_z = self.microscope._z
        start_pfs = self.microscope.get_pfs_offset()
        original_cell_log = super().run()
        self.write_log()
        self.logger.info("Initial processing complete, entering repeats loop")
        self.repeat_index = 1
        repeats_log = []
        if self.tracking:
            repeats_log = self.run_repeats(original_cell_log)
        else:
            for i in range(self.repeat_index, self.repeats_n):
                self.microscope.move_stage(start_position, True, 6)
                if not self.microscope.pfs_focus(True):
                    self.microscope.move_z(start_z, True)
                sleep(1)
                self.microscope.move_pfs(start_pfs, True)
                if not self.microscope.pfs_focus(True):
                    # TODO: code duplication!
                    if not self.full_pfs_recovery(
                        planes={
                            "spiral0": self.well_z_positions[self.spirals[0].name],
                            "start": self.start_z_position,
                        }
                    ):
                        self.logger.error(
                            f"PFS lock irrecoverable, aborting"
                        )  # TODO should this raise an exception?
                        return original_cell_log + repeats_log
                for spiral in self.spirals:
                    spiral.reset()
                repeats_log += super().run()
                self.repeat_index += 1

        return original_cell_log + repeats_log

    def image_cells_from_log(self, cell_log):
        new_log = []

        grouped_by_stage = {}
        for cell in cell_log:
            stage_p = cell["stage_position"]
            key = cell["well"] + "-" + str(cell["field"])
            if key in grouped_by_stage:
                grouped_by_stage[key].append(cell)
            else:
                grouped_by_stage[key] = [cell]

        if self.use_progressbars:
            fovs_pbar = tqdm(desc="FoVs", total=len(grouped_by_stage))
            cells_pbar = tqdm(desc="Cells", total=len(cell_log))

        last_well = None
        for key, cells in grouped_by_stage.items():
            try:
                if self.timeout_time is not None and time() > self.timeout_time:
                    self.logger.info("Timeout reached, aborting run")
                    break

                if len(cells) == 0:
                    # this should never happen
                    continue

                stage_position = cells[0]["stage_position"]
                speed = 3
                if (
                    self.pfs_off_between_wells
                    and last_well is None
                    or last_well != cells[0]["well"]
                ):
                    self.microscope.set_pfs(False)
                    speed = self.interwell_speed

                last_well = cells[0]["well"]
                self.pre_fov_change(None)
                self.microscope.move_stage(stage_position, True, speed=speed)
                if self.pfs_off_between_wells:
                    self.microscope.set_pfs(True)
                    sleep(0.5)

                if not self.microscope.pfs_focus():
                    self.logger.error("PFS LOST FOCUS!")
                    if not self.full_pfs_recovery(
                        planes={
                            "well_pos": self.well_z_positions[last_well],
                            "start": self.start_z_position,
                        }
                    ):
                        self.logger.error(
                            f"PFS lock irrecoverable, aborting"
                        )  # TODO should this raise an exception?
                        # TODO: close pbars?
                        return new_log

                relevant_offsets = []
                for cell in cells:
                    if self.use_last_offset > 0:
                        if not "repeats_offsets" in cell:
                            cell["repeats_offsets"] = [cell["pfs_offset"]]

                        offsets = np.array(cell["repeats_offsets"])
                        if len(offsets) > self.use_last_offset:
                            cell_analysis_offset = offsets[
                                -self.use_last_offset :
                            ].mean()
                        else:
                            cell_analysis_offset = offsets.mean()
                        relevant_offsets.append(cell_analysis_offset)
                        cell["_use_offset"] = cell_analysis_offset

                    else:
                        relevant_offsets.append(cell["pfs_offset"])
                        cell["_use_offset"] = cell["pfs_offset"]
                        cell["repeats_offsets"] = [cell["pfs_offset"]]

                relevant_offsets.append(self.analysis_offset)

                cells_found = {}
                if len(relevant_offsets) < 4:
                    retries = 2
                else:
                    retries = 1
                relevant_offsets = set(relevant_offsets)
                self.logger.info(
                    f"Attempting {len(relevant_offsets)} offsets in cell finding"
                )

                self.microscope.restore_camera_defaults()

                cells_found = self.find_matching_cells(
                    cells, relevant_offsets, key, retries
                )
                self.logger.info(f"Found {len(cells_found)}/{len(cells)} in FoV")

                for cell_id, cell in cells_found.items():
                    self.microscope.move_pfs(cell["_use_offset"], True)
                    matched_cell = cell["_last_match"]

                    dist = np.linalg.norm(
                        np.array(cell["center_position"][:2])
                        - np.array(matched_cell[:2])
                    )
                    sleep(0.1)
                    result, new_cell_info = self.full_cell_process(
                        matched_cell,
                        cell["well"],
                        cell["field"],
                        cell["cell_id"],
                        {} if not "extras" in cell else cell["extras"],
                    )
                    if result and new_cell_info is not None:
                        # cell_offsets.append(microscope._pfs_offset)
                        new_cell_info["distance"] = dist
                        # new_cell_info["match_attempt"] = find_attempt
                        if "pfs_offset" in new_cell_info:
                            cell["repeats_offsets"].append(new_cell_info["pfs_offset"])
                        self.cell_log.append(new_cell_info)
                        new_log.append(new_cell_info)
                        if self.use_progressbars:
                            cells_pbar.update()

            except Exception as e:
                self.logger.error("Error processing loop:", exc_info=e)

            if self.use_progressbars:
                fovs_pbar.update()

        if self.use_progressbars:
            cells_pbar.close()

        return new_log

    def find_matching_cells(self, cells, relevant_offsets, key, retries=2):
        cells_found = {}
        for retry_index in range(retries):
            for offset in set(
                relevant_offsets
            ):  # try all relevant offsets (1/cell in FoV)
                if len(cells_found) == len(cells):
                    break

                self.microscope.move_pfs(offset, True)
                sleep(self.delay_pre_record / 3)  # TODO
                image = self.microscope.get_image()
                vcs, extra_info = self.valid_cells(
                    image, f"{key}-{self.repeat_index}-{offset}"
                )
                for cell in cells:
                    if cell["cell_id"] in cells_found:
                        continue

                    matched_cell = self.find_nearest_cell(
                        vcs, cell["center_position"], self.max_cell_shift_cell_finding
                    )  # TODO: configurable max distance
                    if matched_cell is not None:
                        cell["_last_match"] = matched_cell
                        cells_found[cell["cell_id"]] = cell
        return cells_found

    #    def write_log(self):
    #        original_log = self.cell_log
    #        #TODO check this works correctly and doesn't change in place
    #        self.cell_log = self.cell_log + self.repeats_log
    #        super().write_log()
    #        self.cell_log = original_log

    def run_repeats(self, cell_log):
        cell_log = deepcopy(cell_log)  # make sure this doesn't refer to self.cell_log
        filtered_log = []
        for cell in cell_log:
            if "result" in cell and cell["result"] == "success":
                filtered_log.append(cell)

        repeats_log = []
        repeats = (
            tqdm(range(self.repeat_index, self.repeats_n))
            if self.use_progressbars
            else range(self.repeat_index, self.repeats_n)
        )
        for repeat_n in repeats:
            self.repeat_index = repeat_n
            repeats_log = repeats_log + self.image_cells_from_log(filtered_log)
            if self.timeout_time is not None and time() > self.timeout_time:
                self.logger.info("Timeout reached, aborting run")
                break
            self.write_log()  # for resilience to failures
        return repeats_log

    def get_configuration_dict(self):
        full = super().get_configuration_dict()
        for entry in ["cell_log"]:
            del full[entry]
        return full


class CustomPositionsRepeatsMultiWellCellFindingExperiment(
    RepeatsMultiWellCellFindingExperiment
):
    def interactive_position_init(self, well_names):
        self.well_analysis_offsets = {}
        # based on the first one?
        self.analysis_offset = self.microscope.get_pfs_offset()
        for well_name in well_names:
            input(f"Move to {well_name} center and press any key")
            pad_center = np.array(self.microscope.get_stage_position())
            self.well_centers.append(pad_center)  # potentially useful?
            self.spirals.append(
                SpiralMoves(
                    self.move_by, self.max_fields, pad_center, True, name=well_name
                )
            )
            self.well_z_positions[well_name] = self.microscope._z
            self.well_analysis_offsets[well_name] = self.microscope.get_pfs_offset()
            self.statistics[well_name] = self.last_statistics.copy()


class ThresholdAutoFocusPreprocessor:
    def __init__(
        self,
        experiment: CellFindingExperiment,
    ):
        self.steps = 9
        self.step_size = 110
        self.experiment = experiment
        self.move_delay = 0.2
        self.large_move_delay = 1

        self.threshold_mean_max_grad = 0.13
        self.threshold_unrolled_max_grad_variance = 0.025
        self.threshold_unrolled_sobel = 1850
        self.logger = logging.getLogger("AUTOFOCUS")

        self.retry_focus = False
        self.shift_from_max = 0
        self.disable_autofocus = False  # Only use cell if already in focus
        self.readjust_cell_camera = (
            True  # check if cell is in the center and try to get it there
        )
        self.verify_focus = False  # use determine_abs_focus as last step, return false if not in focus after autofocus
        self.save_rejected_previews = True
        self.rejected_counter = 0

    def get_configuration_dict(self):
        config_dict = {
            "class_name": str(self.__class__),
        }
        config_dict = config_dict | self.__dict__

        for entry in ["experiment", "logger"]:
            del config_dict[entry]
        return config_dict

    def focus_metric(self, image):
        """
        Override this to change metric
        """
        return self.unrolled_mean_max_grad(image)

    def determine_abs_focus(self, image):
        mean = image.mean()
        mmg = self.unrolled_mean_max_grad(image) / mean
        if mmg < self.threshold_mean_max_grad:
            self.logger.info(f"Mean max grad: {mmg:.3f} -> rejected")
            return False, [mmg, np.nan, np.nan]
        mgv = self.unrolled_max_grad_variance(image) / mean
        if mgv > self.threshold_unrolled_max_grad_variance:
            self.logger.info(f"Max grad variance: {mgv:.3f} -> rejected")
            return False, [mmg, mgv, np.nan]

        us = self.unrolled_sobel_metric(image) / mean * 100
        if us < self.threshold_unrolled_sobel:
            self.logger.info(f"Radial laplacian: {us:.3f} -> rejected")
            return False, [mmg, mgv, us]

        self.logger.info(
            f"Skipping autofocus (mmg: {mmg:.3f}, mgv:{mgv:.3f}, us:{us:.3f})"
        )
        # TODO: adjust parameters, add other checks???
        return True, [mmg, mgv, us]

    def determine_abs_focus_final(self, image):
        return self.determine_abs_focus(image)

    def readjust_camera(self, test_image, info, af_start):
        center, radius = self.experiment.contour_tracker.find_cell_center(test_image)
        if center is None:
            self.logger.info("Failed to find cell in view, skipping")
            info["autofocus-status"] = "lost"
            info["autofocus-elapsed"] = time() - af_start
            info["autofocus-skip-metrics"] = None
            return False, info, None

        frame_center = self.experiment.frame_size // 2
        shift = np.array(frame_center) - np.array(center)
        if np.linalg.norm(shift) > 25:
            self.logger.info(f"Cell shifted {shift} from center attempting shift")
            target_offset = self.experiment.microscope._image_offset - shift
            new_offset = self.experiment.adjust_offset_to_limits(target_offset)
            new_center_in_image = target_offset - new_offset + frame_center
            info["image_offset"] = new_offset
            info["cell_shift"] = shift
            info["video_center_position"] = new_center_in_image
            self.experiment.microscope.configure_camera({"image_offset": new_offset})
            test_image = CCT.normalise_image_values(
                self.experiment.microscope.get_image()
            )

            return True, info, test_image

        return True, info, test_image

    def run(self):
        info = {}
        af_start = time()
        start_pos = self.experiment.microscope.get_pfs_offset()
        min_point = start_pos - (self.steps - 1) / 2.0 * self.step_size
        max_point = start_pos + (self.steps - 1) / 2.0 * self.step_size
        positions = np.linspace(min_point, max_point, self.steps)
        test_image_pre_norm = self.experiment.microscope.get_image()
        test_image = CCT.normalise_image_values(test_image_pre_norm)
        if self.readjust_cell_camera:
            res, info, test_image = self.readjust_camera(test_image, info, af_start)
            if not res:
                return False, info
        skipping = False
        is_focused, metric_values = self.determine_abs_focus(test_image)
        if is_focused:  # TODO: this might not be very reliable
            elapsed = time() - af_start
            self.logger.info(f"Skipping autofocus")
            info["autofocus-status"] = "skip"
            info["autofocus-elapsed"] = elapsed
            info["autofocus-skip-metrics"] = metric_values
            skipping = True
            # we might still need to do cell validation here!
            # return True, info
        elif self.disable_autofocus:
            elapsed = time() - af_start
            self.logger.debug(f"Autofocus rejection took seconds:{elapsed:.2f}")
            info["autofocus-status"] = "disabled"
            info["autofocus-elapsed"] = elapsed
            info["autofocus-skip-metrics"] = metric_values
            return False, info
        else:
            focus_result, values = self.autofocus_npoint(positions)
            if not focus_result and self.retry_focus:
                self.logger.warning("Retrying autofocus")
                focus_result, values = self.autofocus_npoint(positions)

            info["autofocus-metric-values"] = values
            elapsed = time() - af_start
            info["autofocus-elapsed"] = elapsed
            if not focus_result:
                self.logger.warning("Autofocus failed")
                info["autofocus-status"] = "fail"
                self.experiment.microscope.move_pfs(start_pos, True)
                return False, info

        is_valid = True
        if self.shift_from_max != 0 and not skipping:
            self.experiment.microscope.move_pfs(self.shift_from_max)

            # TODO this image is also preview, find a way to share it
            sleep(self.move_delay)
            final_image = CCT.normalise_image_values(
                self.experiment.microscope.get_image()
            )

            if self.verify_focus:
                focus_result, final_metrics = self.determine_abs_focus_final(
                    final_image
                )
                info["autofocus-final-metrics"] = final_metrics
                if not focus_result:
                    info["autofocus-status"] = "fail"
                    self.logger.warning(f"Cell seems out of focus after autofocus")
                    cv2.imwrite(
                        self.experiment.data_folder
                        + f"/focus-rejected-{self.rejected_counter}.jpeg",
                        (final_image * 255.0).astype(np.uint8),
                    )
                    self.rejected_counter += 1
                    return False, info
        elif skipping:
            final_image = test_image
        else:
            final_image = CCT.normalise_image_values(
                self.experiment.microscope.get_image()
            )

        if is_valid and self.experiment.cell_validation:
            is_valid = self.experiment.validate_cell(final_image, final=True)

        if is_valid:
            info["autofocus-status"] = "success"
        else:
            info["autofocus-status"] = "invalid-final"
            self.logger.warning(f"Cell contour not valid after focus")
        elapsed = time() - af_start
        info["autofocus-elapsed"] = elapsed
        self.logger.info(f"Autofocus took seconds:{elapsed:.2f}")
        return is_valid, info

    def focus_scores_to_offset(self, positions, values):
        max_position = positions[np.argmax(values)]

        for i in [0, len(values) - 1]:
            if np.argmax(values) == i:
                self.logger.warning("Max value at edge of range")
                # self.experiment.microscope.move_pfs(positions[i], absolute=True)
                return max_position, False  # should we retry?

        # self.experiment.microscope.move_pfs(max_position, absolute=True)
        step_size = positions[1] - positions[0]

        try:
            fit_start = max(0, np.argmax(values) - 2)
            fit_end = min(len(values), np.argmax(values) + 2)
            relevant_values = values[fit_start:fit_end]
            relevant_positions = positions[fit_start:fit_end]
            coeffs = np.polyfit(relevant_positions, relevant_values, 2)
            fit_position = -coeffs[1] / (2 * coeffs[0])
            peak_shift = fit_position - max_position
            if abs(peak_shift) > 2 * step_size:
                self.logger.warning(
                    f"Autofocus fit returned invalid shift {peak_shift}"
                )
                # TODO how to handle this?
                return (
                    max_position,
                    False,
                )  # self.determine_abs_focus(CCT.normalise_image_values(self.experiment.microscope.get_image())), value_log

            self.logger.info(f"Autofocus shift {int(peak_shift)} from max point")
        except Exception as e:
            self.logger.warning("Autofocus fit failed", exc_info=e)
            return max_position, False

        # sleep(self.large_move_delay) #large move

        return (
            max_position,
            fit_position,
        )  # self.experiment.microscope.get_pfs_offset(False)

    def autofocus_npoint(self, positions):
        values = []
        self.experiment.microscope.move_pfs(positions[0], absolute=True)
        sleep(self.large_move_delay)  # large move
        for offset in positions:
            self.experiment.microscope.move_pfs(offset, absolute=True)
            sleep(self.move_delay)
            values.append(
                self.focus_metric(
                    CCT.normalise_image_values(self.experiment.microscope.get_image())
                )
            )

        return self.autofocus_post_process(positions, values)

    def autofocus_post_process(self, positions, values):
        value_log = list(zip(positions, values))
        max_position, fit_position = self.focus_scores_to_offset(positions, values)
        if fit_position:
            self.experiment.microscope.move_pfs(int(fit_position), absolute=True)
            sleep(self.large_move_delay)
        else:
            self.experiment.microscope.move_pfs(int(max_position), absolute=True)
            is_focused, metric_values = self.determine_abs_focus(
                CCT.normalise_image_values(self.experiment.microscope.get_image())
            )
            if is_focused:
                self.logger.info("Simple autofocus passed abs focus check")
            return is_focused, value_log

        return True, value_log  # self.experiment.microscope.get_pfs_offset(False)

    # Autofocusing in Computer Microscopy: Selecting the Optimal Focus Algorithm
    def unrolled_max_grad_variance(self, image):
        center, radius = self.experiment.contour_tracker.find_cell_center(image)
        if center is None:
            return None
        transformed = self.experiment.contour_tracker.transform_image(
            image, center, 100, 0
        )
        gradients = np.gradient(transformed, axis=0)
        mean_max_grad = np.abs(gradients).max(axis=0).std()
        # sobel = cv2.Sobel(image, cv2.CV_64F, 2, 0)
        # if np.abs(gradients.min(axis=0)).mean() > np.abs(gradients.max(axis=0)).mean():
        #    return -mean_max_grad
        return mean_max_grad

    def unrolled_sobel_metric(self, image):
        center, radius = self.experiment.contour_tracker.find_cell_center(image)
        if center is None:
            center = self.experiment.frame_size // 2
        transformed = self.experiment.contour_tracker.transform_image(
            image, center, 100, 0
        )
        sobel = cv2.Sobel(transformed, cv2.CV_64F, 0, 2)
        return np.linalg.norm(sobel)

    def unrolled_mean_max_grad(self, image):
        """
        Mean (across azimuths) of maximum radial gradients
        """
        center, radius = self.experiment.contour_tracker.find_cell_center(image)
        if center is None:
            return -1000
        transformed = self.experiment.contour_tracker.transform_image(
            image, center, 100, 0
        )
        gradients = np.gradient(transformed, axis=0)
        mean_max_grad = np.abs(gradients).max(axis=0).mean()
        # sobel = cv2.Sobel(image, cv2.CV_64F, 2, 0)
        if np.abs(gradients.min(axis=0)).mean() > np.abs(gradients.max(axis=0)).mean():
            return -mean_max_grad
        return mean_max_grad

    def get_all_metrics(self, image, skip_normalise=False):
        if not skip_normalise:
            image = CCT.normalise_image_values(image)
        mean = image.mean()
        mmg = self.unrolled_mean_max_grad(image) / mean
        mgv = self.unrolled_max_grad_variance(image) / mean
        us = self.unrolled_sobel_metric(image) / mean * 100
        return [mmg, mgv, us]

    def interactive_threshold_setup(self):
        self.logger.info("Starting interacive setup")

        input("Find a nice cell and press enter")

        image = self.experiment.microscope.get_image()
        valid_cells, _ = self.experiment.valid_cells(image, "initial_calibration")
        image_center = self.experiment.microscope.max_image_size() / 2
        center_cell = self.experiment.find_nearest_cell(valid_cells, image_center, 600)
        if center_cell is None:
            self.logger.error("No cell found near center, restarting")
            return self.interactive_threshold_setup()
        self.logger.info(f"Cell found at {center_cell}")
        self.experiment.camera_cell(center_cell)
        response = input("Verify cell correct (y/n)")
        if response != "y":
            self.logger.info("Aborting interactive setup")
            self.experiment.camera_full()

            return False

        start_position = self.experiment.microscope.get_pfs_offset()
        metrics = []
        image = self.experiment.microscope.get_image()
        metrics.append(self.get_all_metrics(image))

        start_pos = start_position
        steps = 4 * self.steps
        step_size = int(0.5 * self.step_size)
        min_point = start_pos - (self.steps - 1) * step_size
        max_point = start_pos + (self.steps - 1) * step_size
        positions = np.linspace(min_point, max_point, steps)

        autofocus_result, autofocus_metric_results = self.autofocus_npoint(positions)

        if not autofocus_result:
            self.logger.error("Autofocus failed in setup, aborting.")
            return False

        response = input("Verify cell in focus (y/n)")
        if response != "y":
            self.logger.info("Aborting interactive setup")
            self.experiment.camera_full()

            return False

        image = self.experiment.microscope.get_image()
        metrics.append(self.get_all_metrics(image))

        autofocus_position = self.experiment.microscope.get_pfs_offset()
        self.shift_from_max = start_position - autofocus_position

        input("Move to edge of focus and press enter")
        image = self.experiment.microscope.get_image()
        metrics.append(self.get_all_metrics(image))
        focus_limits = [self.experiment.microscope.get_pfs_offset()]

        input("Move to other edge of focus and press enter")
        image = self.experiment.microscope.get_image()
        metrics.append(self.get_all_metrics(image))
        focus_limits.append(self.experiment.microscope.get_pfs_offset())

        focus_range_steps = np.abs(focus_limits[1] - focus_limits[0])
        self.logger.info(f"Focus range: {focus_range_steps} PFS steps")
        metrics_min = np.array(metrics).min(axis=0)
        metrics_max = np.array(metrics).max(axis=0)

        self.threshold_mean_max_grad = metrics_min[0]
        self.threshold_unrolled_max_grad_variance = metrics_max[1]
        self.threshold_unrolled_sobel = metrics_min[2]

        self.experiment.analysis_offset = start_position

        self.logger.info(f"Completed autofocus setup. Thresholds:")
        self.logger.info(f"Mean Max Grad {self.threshold_mean_max_grad:.2f}")
        self.logger.info(
            f"Unrolled max grad variance: {self.threshold_unrolled_max_grad_variance}"
        )
        self.logger.info(f"Unrolled sobel: {self.threshold_unrolled_sobel}")
        self.logger.info(f"Autofocus shift from peak: {self.shift_from_max}")
        response = input("Continue with these settings: (y/n)")
        if response != "y":
            self.logger.info("Aborting interactive setup")
            self.experiment.camera_full()

            return False
        # TODO: get cell near center
        # get metrics, run autofocus, get offset, manually change to limits of focus to get thresholds, range
        self.experiment.microscope.restore_camera_defaults()
        return True


class ThresholdTempFileAutoFocusPreprocessor(ThresholdAutoFocusPreprocessor):
    def __init__(
        self,
        experiment: CellFindingExperiment,
    ):
        super().__init__(experiment)
        self.temp_filename = "/dev/shm/autofocus-temp.movie"
        self.move_delay = 0.15
        self.large_move_delay = 0.5
        self.steps = 7
        self.step_size = 120

    def record_temp_file(self, positions):
        # lower = start_position - steps//2*step_size
        # upper = start_position + steps//2*step_size
        # r = list(range(lower, upper, step_size))
        m = self.experiment.microscope

        # Check if the microscope is a Temika microscope
        if not hasattr(m, "_write_script"):
            raise NotImplementedError(
                "record_temp_file is only implemented for Temika microscopes. "
                "Custom microscopes must implement their own autofocus sweep or recording method."
            )

        # Locally import TemikaXML elements to keep core acquisition independent of Temika XML drivers
        try:
            from TemikaXML.SamplePlatform import NOTHING
            from TemikaXML.Camera import Save, Record, PerfectFocus, Sleep, Trigger, Script
        except ImportError as e:
            raise ImportError(
                "The TemikaXML library is required to run autofocus sweeps on Temika microscopes. "
                "Please ensure the microscope drivers are correctly installed."
            ) from e

        original_trigger = m._trigger
        if original_trigger != "SOFTWARE":
            after = [m.get_camera_script_element()]
            m._trigger = "SOFTWARE"
            focus_steps = [m.get_camera_script_element()]
        else:
            focus_steps = []
            after = []

        focus_steps += [
            Save(name=self.temp_filename, append=NOTHING),
            Record(self.experiment.microscope.camera_name, record=True),
        ]
        is_first = True
        for p in positions:
            focus_steps += [
                PerfectFocus(offset=p),
                Sleep(
                    seconds=self.move_delay if not is_first else self.large_move_delay
                ),
                Trigger(self.experiment.microscope.camera_name),
            ]
            is_first = False

        after = [Record(self.experiment.microscope.camera_name, record=False)] + after

        script = Script(*focus_steps, *after)

        script_send = time()
        # script_time = time()
        self.experiment.microscope._write_script(script)
        m._trigger = original_trigger
        script_return = time()

        # start a non-blocking move towards the middle
        m.blocking = False
        m.move_pfs(int(np.mean(positions)), True)
        m.blocking = True
        move_start = time()

        return script_send, script_return, move_start

    def autofocus_npoint(self, positions):
        n_point_enter = time()
        script_send, script_return, move_start = self.record_temp_file(positions)

        movie = get_movie_reader(self.temp_filename + ".movie")
        metric_res = []
        if movie.n_frames != len(positions):
            self.logger.error(
                f"Recorded file frames ({movie.n_frames}) != steps ({self.steps}), this might cause focus failure"
            )
            if movie.n_frames > 100:
                return False, []

        for f in movie.frames():
            metric_v = self.focus_metric(CCT.normalise_image_values(f))
            metric_res.append(metric_v)

        movie_read = time()
        v = self.autofocus_post_process(positions, metric_res)
        comp = time()
        self.logger.info(
            f"Focus timings: setup:{script_send-n_point_enter:.2f}s,script:{script_return-script_send:.2f}s,ms:{move_start-script_return:.2f}s,read:{movie_read-script_return:.2f}s,post:{comp-movie_read:.2f}s"
        )
        return v


# This only defines the run at cells for easier addition to other experiment base classes
# a bit weird in that it doesnt inherit from BaseExperiment, but this is cleaner than diamond inheritance
# use first because of MRO left to right
class BaseFlickeringExperiment:
    def run_at_cell(self, global_id):
        """
        Implement this to record cell,
        will be run once cell preprocessing finishes, if it is successful.

        camera is already configured to record frame_size around cell
        """
        if not hasattr(self, "fps"):
            self.fps = 660

        if not hasattr(self, "duration"):
            self.duration = 20

        if not hasattr(self, "shutter"):
            self.shutter = 0.0008

        # TODO: how much time is this costing us? Is it better to pull the first frame of a movie file instead?
        t = time()
        self.microscope.configure_camera({"shutter": self.shutter}, delay_send=True)
        img = self.microscope.get_image()
        cv2.imwrite(
            self.data_folder + f"/{global_id}-preview.jpeg",
            (CCT.normalise_image_values(img) * 255.0).astype(np.uint8),
        )
        preview_time = time() - t
        self.logger.info(f"Obtaining preview image took {preview_time:.2f} s")

        self.logger.info(f"Recording cell {global_id}")
        # this way these are included in the metadata
        self.microscope.record_video(
            self.data_folder + f"/{global_id}", self.fps, self.duration
        )  # TODO: make recording setup more sensible!
        return True, {
            "fps": self.fps,
            "duration": self.duration,
            "shutter": self.shutter,
        }  # TODO: should we return some more info here?


# testing the new resilience features
if __name__ == "__main__":
    from temika.microscope import TemikaMicroscope
    TemikaMicroscope.start_temika()
    m = TemikaMicroscope(timeout=23)
    while True:
        # m.get_image()
        m.set_illumination(0, 1, 0)
        m.move_stage([100, 100])
        print("waiting")
        # m.get_image()
        sleep(2)
        m.move_stage([-100, -100])
        m.set_illumination(0, 1, 1)
        sleep(1)
