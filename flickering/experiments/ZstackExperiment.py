from flickering.acquisition.autoimager import *
from flickering.acquisition.multifocus import *
import cv2
import os
from flickering.tracking.correlation_tracker import CorrelationContourTracker as CCT
from flickering.acquisition.autofocus.RandomForestAutofocus import RandomForestCellValidator
import json
from copy import deepcopy
from time import time
from random import choice

class ZstackExperiment(RandomForestCellValidator, RepeatsMultiWellCellFindingExperiment):
    def run_at_cell(self, global_id):
        """
        Implement this to record cell,
        will be run once cell preprocessing finishes, if it is successful.

        camera is already configured to record frame_size around cell
        """
        if not hasattr(self, "fps"):
            self.fps = 100

        if not hasattr(self, "duration"):
            self.duration = 20

        if not hasattr(self, "shutter"):
            self.shutter = 0.0008

        #this runs from the best focus position
        use_offsets = np.linspace(-2000, 2000, 15)
        if choice(["up", "down"]) == "down":
            self.logger.info("Running offsets in reverse")
            use_offsets = use_offsets[::-1]

        i=0
        fnames = []
        shutters = []
        zs = []
        original_offset = self.microscope.get_pfs_offset()
        original_z = self.microscope._z
        for offset in use_offsets:
            fname = f"{global_id}-{i}_{offset:.0f}"

            #self.microscope.configure_camera({"shutter": shutter})
            #brightness = use_shutters.min()/shutter #TODO check if this works
            #self.microscope.set_illumination(0,2,brightness, "EXTERNAL0")
            self.microscope.move_pfs(int(original_offset+offset), True)
            sleep(0.3)
            zs.append(self.microscope._z) #updated by move_pfs (I think)
            img = self.microscope.get_image()
            img = CCT.normalise_image_values(img)
            res, info, img = self.cell_preprocessor.readjust_camera(img, {}, time())
            cv2.imwrite(self.data_folder + f"/{fname}-preview.jpeg", (img*255.0).astype(np.uint8))
            #1.0 at the minumum shutter time
            if not res:
                self.logger.error("Failed to readjust to cell, skipping")
                #TODO: more tolerant?
                break
            self.logger.info(f"Recording cell {fname}@{offset:.2f}/z={zs[-1]:.1f}um")
            self.microscope.record_video(self.data_folder + fname, self.fps, self.duration)
            fnames.append(fname)
            shutters.append(offset)
            #brightnesss.append(brightness)

        self.microscope.move_pfs(int(original_offset), True)

        #self.microscope.configure_camera({"shutter": original_shutter})
        #self.microscope.set_illumination(0,2,self.illumination, "EXTERNAL0")

        return True, {
            "fps": self.fps,
            "duration": self.duration,
            "shutter": self.shutter,
            "offsets": shutters,
            "zs": zs,
            "file_names": fnames
        }


if __name__ == "__main__":
    import sys
    try:
        from temika.microscope import TemikaMicroscope
        from TemikaXML.SamplePlatform import Pad, PadRangeParser, Pad_96_Well_Plate
    except ImportError as e:
        print("Error: The 'temika' and/or 'TemikaXML' package is required to run the default script configuration.", file=sys.stderr)
        print("Please install them or modify this block to use your own microscope driver and stage coordinates.", file=sys.stderr)
        sys.exit(1)

    folder = "./data/2025-02-05_zstacks1/"
    
    os.mkdir(folder)
    os.mkdir(folder+"/rfv_test/")
    logfile = folder + f"log_{time():.0f}.log"
    print(f"Logfile: {logfile}")
    logging.basicConfig(filename=logfile, encoding="utf-8", level=logging.INFO)
    
    m = TemikaMicroscope(timeout=300)
    m.camera_defaults["trigger"] = "INTERNAL"
    m.camera_defaults["fps"] = 50
    m.restore_camera_defaults()
    start_position = m.get_stage_position()
    start_z = m._z
    start_pfs = m.get_pfs_offset()
    wells = list(Pad_96_Well_Plate.range_in_traversal_order("A1"))
    e = ShutterSpeedExperiment(m, wells, 1000, reference_well = wells[0], move_by=np.array([180,180]), data_folder=folder)
    e.illumination = 1 #for 0.8 shutter
    e.duration = 20
    
    #THIS IS IMPORTANT, OTHERWISE RECOVERY TURNS ILLUMINATION OFF!
    m.set_illumination(0, 2, e.illumination, "EXTERNAL0") #default camera trigger is iternal
    
    #this causes problems for some reason, it sends a trigger but there might be a timing issue with receiving the frame
    #m.camera_defaults["trigger"] = "SOFTWARE"
    m.camera_defaults["fps"] = 50
    m.restore_camera_defaults()
    
    #channel 0=sample,1=objective
    m.set_temperatures_settings([Temperature(0,0,"HEATER",37,True, True,70, 20000,100,0),Temperature(1,1,"HEATER",37,True, True,70, 25000,37,0)])
    #exit(0)
    e.description = "37C, chamber,1:3700 dilution, 1mg/ml BSA + 20mM glucose, 12/1/2025 blood, 32 focus offset, multifocus 11steps@100"
    e.initialise_validator(0.6,0.52)
    e.max_dwell_time = 5*600000
    e.tracking = False
    e.edge_well_time_multiplier = 2#2 #disable for 2 wells only
    e.cells_per_well = 400
    #e.thresholds["CELL_CLEAR_AREA_FROM"] = 15
    e.thresholds["CELL_MIN_SEPARATION"] = 25
    e.pfs_off_between_wells = False
    e.analysis_shutter = 0.0008
    
    e.repeats_n = 1#00
    e.contour_tracker.laplace_th = 0.0015
    #allow more elongated cells
    e.contour_tracker.percentile_th = 0.3
    e.contour_tracker.std_dev_th = 0.2
    e.contour_tracker.max_shift_th = 5
    
    #e.contour_tracker.debug = True
    
    #focus = ThresholdTempFileAutoFocusPreprocessor(e)
    #focus = LastTrackingAutofocus(e)
    focus = RandomForestAutofocus(e)
    focus.steps = 11
    focus.step_size = 100
    focus.skip_post_check = False
    focus.ignore_post_result = False
    focus.disable_autofocus = False #TODO: will enough cells be in focus? Will this have unintended effects? - try for one loop and see?
    focus.verify_focus = True
    focus.readjust_cell_camera = True #probably not needed on glass?
    
    #possibly too strict?
    #focus.threshold_mean_max_grad = 0.42
    #focus.threshold_unrolled_max_grad_variance = 0.076
    #focus.threshold_unrolled_sobel = 7000
    focus.shift_from_max = 32
    #something seems wrong with this, don't know why
    focus.final_threshold = 0.35 #normally lower
    e.cell_preprocessor = focus
    focus.move_delay = 0.7 # wobbly?
    focus.large_move_delay = 1 #very wobbly
    
    contour_tracker = e.contour_tracker
    contour_tracker.hough_param_1 = 50
    contour_tracker.hough_param_2 = 33
    contour_tracker.hough_min_r = 28
    contour_tracker.hough_max_r = 80
    contour_tracker.hough_min_d = 70
    contour_tracker.mask_width = 15
    contour_tracker.ignore_center_rad = 0
    
    #if not focus.interactive_threshold_setup():
        #print(e.contour_tracker.debug_data["validation_scores"])
    #    print("Focus setup failed")
    #    exit(-1)
    
    e.timeout = 3600*20
    
    #e.fps = 664
    #logging.basicConfig(filename=logfile, encoding="utf-8", level=logging.INFO)
    
    #cell_log_f = f"{folder}/1714461242.json"
    
    #with open(cell_log_f, "r") as f:
    #    last_run = json.load(f)
    
    
    #cell_log = last_run["cells"]
    try:
        #e.continue_interrupted(cell_log_f)
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
