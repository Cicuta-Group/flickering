from flickering.acquisition.autoimager import *
import cv2
import os
from flickering.tracking.correlation_tracker import CorrelationContourTracker as CCT
import warnings
from flickering.acquisition.multifocus import *
from flickering.acquisition.autofocus.RandomForestAutofocus import (
    RandomForestCellValidator,
)


class MultiWellExperiment(
    RandomForestCellValidator, BaseFlickeringExperiment, MultiFocus
):
    #   def on_cell_success(self, cell_info):
    #        if cell_info["cell_id"] > 50 and self.duration > 20:
    #            self.duration = 20
    #            self.logger.info("Switching to regular movie duration")
    #
    #        return super().on_cell_success(cell_info)

    def pre_fov_change(self, spiral):
        if spiral is not None:
            spiral: SpiralMoves = spiral
            image = self.microscope.get_image()
            image = self.contour_tracker.normalise_image_values(image)
            cv2.imwrite(
                self.data_folder
                + f"/fov-{spiral.name}-{spiral.index:03d}-{self.repeat_index}.png",
                (image * 255.0).astype(np.uint8),
            )

    # the inheritance reached its limit
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


if __name__ == "__main__":
    import sys

    try:
        from temika.microscope import TemikaMicroscope
        from TemikaXML.SamplePlatform import Pad, PadRangeParser, Pad_96_Well_Plate
        from TemikaXML.Camera import Temperature
    except ImportError as e:
        print(
            "Error: The 'temika' and/or 'TemikaXML' package is required to run the default script configuration.",
            file=sys.stderr,
        )
        print(
            "Please install them or modify this block to use your own microscope driver and stage coordinates.",
            file=sys.stderr,
        )
        sys.exit(1)

    folder = "./data/2025-11-18_chamber_age/"
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(folder + "/rfv_test/"):
        os.makedirs(folder + "/rfv_test/", exist_ok=True)
    logfile = folder + f"log_{time():.0f}.log"
    print(f"Logfile: {logfile}")
    logging.basicConfig(filename=logfile, encoding="utf-8", level=logging.INFO)

    class Pad_24_Well_Custom(Pad):
        props = {
            "well_count": 24,
            "size_x": 15e3,
            "size_y": 15e3,
            "grid_pitch_x": 18e3,
            "grid_pitch_y": 18e3,
            "feducial_offset_x": -8.1e3,
            "feducial_offset_y": -6.7e3,
            "first_col": 1,
            "last_col": 6,
            "first_row": "A",
            "last_row": "D",
        }
        parser = PadRangeParser(props=props)

        def __init__(self, *args):
            super().__init__(*args)

    class Pad_40_Well_Custom(Pad):
        props = {
            "well_count": 40,
            "size_x": 10e3,
            "size_y": 10e3,
            "grid_pitch_x": 13e3,
            "grid_pitch_y": 13e3,
            "feducial_offset_x": -8.1e3,
            "feducial_offset_y": -6.7e3,
            "first_col": 1,
            "last_col": 8,
            "first_row": "A",
            "last_row": "E",
        }
        parser = PadRangeParser(props=props)

        def __init__(self, *args):
            super().__init__(*args)

    class Pad_15_Well_Custom(Pad):
        props = {
            "well_count": 15,
            "size_x": 15e3,
            "size_y": 15e3,
            "grid_pitch_x": 18e3,
            "grid_pitch_y": 18e3,
            "feducial_offset_x": -8.1e3,
            "feducial_offset_y": -6.7e3,
            "first_col": 1,
            "last_col": 5,
            "first_row": "A",
            "last_row": "C",
        }
        parser = PadRangeParser(props=props)

        def __init__(self, *args):
            super().__init__(*args)

    class Pad_28_Well_Custom(Pad):
        props = {
            "well_count": 28,
            "size_x": 15e3,
            "size_y": 15e3,
            "grid_pitch_x": 18e3,
            "grid_pitch_y": 18e3,
            "feducial_offset_x": -8.1e3,
            "feducial_offset_y": -6.7e3,
            "first_col": 1,
            "last_col": 7,
            "first_row": "A",
            "last_row": "D",
        }
        parser = PadRangeParser(props=props)

        def __init__(self, *args):
            super().__init__(*args)

    class Pad_16_Well_Grace(Pad):
        props = {
            "well_count": 16,
            "size_x": 7e3,
            "size_y": 7e3,
            "grid_pitch_x": 9e3,
            "grid_pitch_y": 9e3,
            "feducial_offset_x": -8.1e3,
            "feducial_offset_y": -6.7e3,
            "first_col": 1,
            "last_col": 8,
            "first_row": "A",
            "last_row": "B",
        }
        parser = PadRangeParser(props=props)

        def __init__(self, *args):
            super().__init__(*args)

        print("Running at 37")
        # TemikaMicroscope.start_temika()

    m = TemikaMicroscope(timeout=300)
    # TODO: only for using the ibidi unstable pfs chambers
    # m.limit_pfs_move_size = 100

    start_position = m.get_stage_position()
    start_z = m._z
    start_pfs = m.get_pfs_offset()
    # THIS IS IMPORTANT, OTHERWISE RECOVERY TURNS ILLUMINATION OFF!
    m.set_illumination(0, 3, 1, "EXTERNAL0")  # default camera trigger is iternal

    # this causes problems for some reason, it sends a trigger but there might be a timing issue with receiving the frame
    # m.camera_defaults["trigger"] = "SOFTWARE"
    m.camera_defaults["fps"] = 50
    m.restore_camera_defaults()

    # channel 0=objective
    m.set_temperatures_settings(
        [
            Temperature(0, 0, "HEATER", 37, True, True, 70, 25000, 37, 0),
            Temperature(1, 1, "HEATER", 37, True, True, 70, 20000, 100, 0),
        ]
    )
    # wells = Pad_96_Well_Plate.range("(D-F)7") + list(reversed(Pad_96_Well_Plate.range("(D-F)8")))+Pad_96_Well_Plate.range("(D-F)9")

    # wells = list(Pad_16_Well_Grace.range_in_traversal_order("(A-B)(1-8)"))#(A-D)(1-7)"))
    # wells = list(Pad_28_Well_Custom.range_in_traversal_order("A1"))#(A-D)(1-7)"))

    # print(wells)
    # exit(0)
    # wells.remove(Pad_28_Well_Custom("A1"))
    # wells.remove(Pad_28_Well_Custom("B2"))
    # wells.remove(Pad_28_Well_Custom("B4"))
    # wells.remove(Pad_28_Well_Custom("D7"))
    # wells.remove(Pad_28_Well_Custom("D6"))
    # wells.remove(Pad_28_Well_Custom("D7"))

    # wells.remove(Pad_28_Well_Custom("A1"))
    # wells.remove(Pad_28_Well_Custom("A2"))
    # wells.remove(Pad_28_Well_Custom("A4"))
    # wells.remove(Pad_28_Well_Custom("B3"))
    # wells.remove(Pad_28_Well_Custom("B7"))
    # wells.remove(Pad_28_Well_Custom("C3"))
    # wells.remove(Pad_28_Well_Custom("C6"))
    # wells.remove(Pad_28_Well_Custom("C7"))
    # wells.remove(Pad_28_Well_Custom("D2"))

    # print(wells)
    # print(len(wells))
    wells = []
    # exit()
    e = MultiWellExperiment(
        m,
        wells,
        500,
        reference_well=None,
        move_by=np.array([180, 180]),
        data_folder=folder,
        well_rotation=0,
    )
    e.interactive_position_init(["Old", "NewOld", "New"])

    # e.check_center_positions(True, 5)
    # exit()
    e.description = "3 custom chambers, one from old adhesive, one 1 week ago adhesive, cut week ago and assembled week ago, one cut and glued fresh (assembled 1hr before). PBS+BSA+5mmol glucose. Plated cold, heated on microscope. A17/11/2025 blood, 32 focus offset, multifocus 11steps@90"
    e.initialise_validator(
        0.4, 0.4
    )  # initial should be higher - we want to reject early
    # e.interactive_position_init(["S","L","Loffset","Soffset"])
    e.check_center_positions(False, wait_time=0.5)

    # e.image_spirals(50)
    # e.check_center_positions(wait_time=1)
    e.max_dwell_time = 120
    e.min_valid_cells = 4
    e.tracking = False
    e.edge_well_time_multiplier = 2  # 2 #disable for 2 wells only
    e.cells_per_well = 1000  # tracking starting with 350 cells, hopefully enough
    # e.thresholds["CELL_CLEAR_AREA_FROM"] = 15
    e.thresholds["CELL_MIN_SEPARATION"] = 25
    e.thresholds["CELL_CLEAR_AREA_2_98_VAR"] = 0.8
    e.thresholds["CELL_CLEAR_AREA_FROM"] = 10
    e.thresholds["CELL_CLEAR_AREA_WIDTH"] = 5
    e.thresholds["CELL_MIN_CONTRAST"] = 0.03
    e.pfs_off_between_wells = False
    e.analysis_shutter = 0.001

    e.repeats_n = 100  # 00
    e.contour_tracker.laplace_th = 0.0015
    # allow more elongated cells
    e.contour_tracker.percentile_th = 0.3
    e.contour_tracker.std_dev_th = 0.2
    e.contour_tracker.max_shift_th = 5

    # e.contour_tracker.debug = True

    # focus = ThresholdTempFileAutoFocusPreprocessor(e)
    # focus = LastTrackingAutofocus(e)

    # work around to disable cell focus validation
    class TempMCA(MultiCellAutofocus):
        def determine_abs_focus(
            self, image, skip_normalise=True, center=None, contour=None, radius=None
        ):
            try:
                metrics = self.get_all_metrics(
                    image,
                    skip_normalise=skip_normalise,
                    center=center,
                    contour=contour,
                    radius=radius,
                )
                # metric_values =  pd.DataFrame(data=[metrics], columns=self.metric_names())

                if np.isnan(metrics[0]):
                    self.logger.info("Invalid focus metrics")
                    metrics.append(np.nan)
                    return False, metrics
                return True, metrics
            except Exception as e:
                self.logger.warning("Focus state determination failed:", exc_info=e)
                return False, [-1000, -1000, -1000, -1000]

    focus = TempMCA(e)
    focus.skip_post_check = True
    focus.ignore_post_result = False
    focus.disable_autofocus = False
    focus.verify_focus = False
    focus.readjust_cell_camera = True  # probably not needed on glass?
    focus.steps = 11
    focus.step_size = 90
    focus.use_temp_file = (
        False  # this lets us use shorter wait times so might be faster
    )

    # possibly too strict?
    # focus.threshold_mean_max_grad = 0.42
    # focus.threshold_unrolled_max_grad_variance = 0.076
    # focus.threshold_unrolled_sobel = 7000
    focus.shift_from_max = 32
    focus.final_threshold = 0.2  # low as the focus score is not super reliable
    e.cell_preprocessor = focus
    focus.move_delay = 0.2  # wobbly?
    focus.large_move_delay = 0.8  # very wobbly
    focus.threshold = 0.3

    contour_tracker = e.contour_tracker
    contour_tracker.hough_param_1 = 50
    contour_tracker.hough_param_2 = 33
    contour_tracker.hough_min_r = 28
    contour_tracker.hough_max_r = 80
    contour_tracker.hough_min_d = 70
    contour_tracker.mask_width = 15
    contour_tracker.ignore_center_rad = 0

    # if not focus.interactive_threshold_setup():
    # print(e.contour_tracker.debug_data["validation_scores"])
    #    print("Focus setup failed")
    #    exit(-1)

    e.duration = 10
    e.timeout = 3600 * 24
    e.fps = 660

    # logging.basicConfig(filename=logfile, encoding="utf-8", level=logging.INFO)

    # cell_log_f = f"{folder}/1714461242.json"

    # with open(cell_log_f, "r") as f:
    #    last_run = json.load(f)

    # cell_log = last_run["cells"]
    try:
        # e.continue_interrupted(cell_log_f)
        with warnings.catch_warnings(action="ignore"):
            e.run()
        logging.info("Experiment done, moving to start position")
        m.move_stage(start_position, True, 2)
    except Exception as exception:
        logging.error("Experiment run failed:", exc_info=exception)
    except KeyboardInterrupt:
        print("ABORTING")
        logging.error("Experiment aborted, returning to start position")
        m.move_stage(start_position, True, 2)
        m.move_z(start_z, True)
        m.restore_camera_defaults()
        sleep(2)
        m.set_pfs(True)
        m.move_pfs(start_pfs, True)
    finally:
        e.write_log()
