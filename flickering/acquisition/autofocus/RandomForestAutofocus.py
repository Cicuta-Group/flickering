from flickering.acquisition.autoimager import *
import os
import pickle
import sklearn
from flickering.acquisition.autoimager import CellFindingExperiment
from flickering.tracking.correlation_tracker import CorrelationContourTracker as CCT
import pandas as pd
import warnings
from scipy.interpolate import LinearNDInterpolator
from time import time
import cv2
from flickering.analysis.fitter import ContourFitter


class RandomForestAutofocus(ThresholdTempFileAutoFocusPreprocessor):
    def __init__(self, experiment: CellFindingExperiment, threshold=0.6):
        super().__init__(experiment)

        self.threshold = threshold
        self.final_threshold = 0.5 #Lower focus threshold after autofocus runs

        self.tracker = CCT()
        self.tracker.refine_correlation = True
        self.tracker.mask_refinement = False

        # TODO: freeze other parameters

        dirname = os.path.dirname(__file__)
        model_file = os.path.join(dirname, "focus_forest.pkl")
        with open(model_file, "rb") as f:
            self.classifier = pickle.load(f)

    def metric_names(self):
        return [
            "unrolled_mean_max_grad",
            "unrolled_max_grad_variance",
            "unrolled_sobel",
            "profile_contrast",
            "profile_width",
            "profile_gradient",
            "profile_fit_width",
            "profile_amplitude",
            "profile_fit_r2",
            "profile_fit_inside_err",
            "profile_fit_outside_err",
        ]

    """
    Uses a pre-trained random forest to decide if in focus. If not run regular autofocus
    """
    def get_all_metrics(self, img, center=None, contour=None, unrolled = None, skip_normalise=False, radius=None):
        # plt.imshow(img, cmap="gray")
        try:
            if not skip_normalise:
                img = CCT.normalise_image_values(img)
            mean = img.mean()
            if contour is None:
                center, contour = self.tracker.get_contour(img, center=center, radius=radius)
            if center is not None:
                # contour_scores = list(self.tracker.get_validation_scores(contour))
                if unrolled is None:
                    unrolled = self.tracker.unroll_around_contour(
                        img, contour, center, 128
                    )

                profile = unrolled.mean(axis=1)

                contrast = (profile.max() - profile.min()) / profile.mean()
                width = np.argmax(profile) - np.argmin(profile)

                def gauss_diff(x, a, b, c, d):
                    return a * (x - c) * np.exp(-b * (x - c) * (x - c)) + d

                profile = profile / profile.mean()
                xvals = np.arange(len(profile)) - len(profile) / 2
                with warnings.catch_warnings(action="ignore"):
                    popt, pcov = curve_fit(
                        gauss_diff,
                        xvals,
                        profile,
                        p0=[(profile.max() - profile.mean())/2.0, 1 / 16, -2, 1],
                        xtol=1e-3
                    )
                    fit_width = np.sqrt(1 / popt[1])
                    yvals = gauss_diff(xvals, *popt)
                    explained_variance = ((yvals - profile) ** 2).sum()
                    variance_tot = np.sum((profile - np.mean(profile)) ** 2)
                    r2 = 1 - explained_variance / variance_tot

                    internal_th = popt[2] - 2 * np.sqrt(1 / popt[1])
                    outside_th = popt[2] + 2 * np.sqrt(1 / popt[1])
                    mean_profile_err_inside = (
                        yvals[xvals < internal_th] - profile[xvals < internal_th]
                    ).sum() / (xvals < internal_th).sum()
                    mean_profile_err_outside = (
                        yvals[xvals > outside_th] - profile[xvals > outside_th]
                    ).sum() / (xvals > outside_th).sum()
                    extra_fscores = [
                        contrast,
                        width,
                        contrast / (width+1e-5),
                        fit_width,
                        popt[0],
                        r2,
                        mean_profile_err_inside,
                        mean_profile_err_outside,
                    ]

                    mmg,mgv = self.grad_metrics(unrolled)
                    mmg = mmg/mean
                    mgv = mgv/mean
                    us = self.unrolled_sobel_metric_unrolled(unrolled)/mean*100

                    focus_scores =  [mmg, mgv, us]
                    # jpeg size could be useful, but the training samples were already jpeged
                    # rv, reencoded = cv2.imencode(".jpeg",img)
                    # focus_scores.append(len(reencoded))
                return focus_scores + extra_fscores
        except Exception as e:
            self.logger.error("Error getting focus metric:", exc_info=e)

        return [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,

        ]

    def determine_abs_focus(self, image, skip_normalise=True, center=None, contour=None, radius=None):
        try:
            metrics = self.get_all_metrics(image, skip_normalise=skip_normalise, center=center, contour=contour,radius=radius)
            metric_values =  pd.DataFrame(data=[metrics], columns=self.metric_names())

            if np.isnan(metrics[0]):
                self.logger.info("Invalid focus metrics")
                metrics.append(np.nan)
                return False, metrics
            prob = self.classifier.predict_proba(metric_values)
            metrics.append(prob[0,1])
            if prob[0,1] < self.threshold:
                self.logger.debug(f"Focus probability {prob[0,1]:.2f}, rejected")
                return False, metrics
            self.logger.debug(f"Focus probability {prob[0,1]:.2f}, accepted")

            return True, metrics
        except Exception as e:
            self.logger.warning("Focus state determination failed:", exc_info=e)
            return False, [-1000,-1000,-1000,-1000] #TODO: a bit hacky

    def determine_abs_focus_final(self, image):
        abs_focus, metrics =  self.determine_abs_focus(image)
        if not abs_focus and metrics[-1] > self.final_threshold:
            self.logger.debug("Focus probability > final threshold, accepting cell")
            return True, metrics

        return abs_focus, metrics

    def focus_metric(self, image):
        """
        Override this to change metric
        """
        #keep the old metric for now
        return super().unrolled_mean_max_grad(image)

    #we want to make use of the already unrolled image
    #so there's a bit of duplication here
    #Autofocusing in Computer Microscopy: Selecting the Optimal Focus Algorithm
    def unrolled_max_grad_variance(self, unrolled):
        gradients = np.gradient(unrolled, axis=0)
        mean_max_grad = np.abs(gradients).max(axis=0).std()
        #sobel = cv2.Sobel(image, cv2.CV_64F, 2, 0)
        #if np.abs(gradients.min(axis=0)).mean() > np.abs(gradients.max(axis=0)).mean():
        #    return -mean_max_grad
        return mean_max_grad

    def unrolled_sobel_metric_unrolled(self, unrolled):
        sobel = cv2.Sobel(unrolled, cv2.CV_64F, 0, 2)
        return np.linalg.norm(sobel)

    def grad_metrics(self,unrolled):
        #these combine unrolled mean max grad and unrolled max grad variance to avoid duplicate gradient calculation
        gradients = np.gradient(unrolled, axis=0)
        max_grad = np.abs(gradients).max(axis=0)
        mean_max_grad = max_grad.mean()
        max_grad_var = max_grad.std()
        #sobel = cv2.Sobel(image, cv2.CV_64F, 2, 0)
        if np.abs(gradients.min(axis=0)).mean() > np.abs(gradients.max(axis=0)).mean():
            mean_max_grad=-mean_max_grad

        return (mean_max_grad,max_grad_var)

    def interactive_threshold_setup(self):
        self.logger.error("Interactive setup not supported for RandomForest Autofocus")
        return False

    def get_configuration_dict(self):
        conf = super().get_configuration_dict()
        del conf["tracker"]
        del conf["classifier"]

        return conf

#TODO: incomplete, is it really needed?
#we might need to extrapolate for points outside the grid
class PositionTrackingAutofocus(RandomForestAutofocus):
    def __init__(self, experiment: CellFindingExperiment, threshold=0.6):
        super().__init__(experiment, threshold)
        self.cell_positions = [] #this will keep [x,y,z,pfs_offset]

    def run(self):
        current_pfs = self.experiment.microscope.get_pfs_offset(True)
        current_stage = self.experiment.microscope.get_stage_position(False)
        current_z = self.experiment.microscope._z
        if len(self.cell_positions) > 1:
            #TODO: interpolated position
            pass
        test_image = CCT.normalise_image_values(self.experiment.microscope.get_image())
        result, info =  super().run()

class LastTrackingAutofocus(RandomForestAutofocus):
    def __init__(self, experiment: CellFindingExperiment, threshold=0.6):
        super().__init__(experiment, threshold)
        self.cell_positions = [] #this will keep [x,y,z,pfs_offset]

    def determine_abs_focus(self, image):
        is_focused, metric_values = super().determine_abs_focus(image)
        if is_focused:
            current_pfs = self.experiment.microscope.get_pfs_offset(True)
            current_stage = self.experiment.microscope.get_stage_position(False)
            current_z = self.experiment.microscope._z
            cell_entry = [current_stage[0],current_stage[1], current_z, current_pfs]
            self.cell_positions.append(cell_entry)

        return is_focused, metric_values

    def run(self):
        af_start = time()
        info = {}
        adjusted = False
        start_pfs = self.experiment.microscope.get_pfs_offset()

        test_image = CCT.normalise_image_values(self.experiment.microscope.get_image())
        if self.readjust_cell_camera:
            adjusted, info, test_image = self.readjust_camera(test_image, info, af_start)
            if not adjusted:
                return False, info

        if len(self.cell_positions) > 0:
            #TODO code duplication, same position tested twice
            is_focused, metric_values = self.determine_abs_focus(test_image)
            if is_focused:
                elapsed = time()-af_start
                self.logger.info(f"Autofocus skip took seconds:{elapsed:.2f}")
                info["autofocus-status"] = "skip"
                info["autofocus-elapsed"] = elapsed
                info["autofocus-skip-metrics"] = metric_values
                return True, info

            last_entry = self.cell_positions[-1]
            zdiff = self.experiment.microscope._z - last_entry[2]
            pfsdiff = start_pfs - last_entry[3]
            if np.abs(pfsdiff) > 50 and np.abs(zdiff) < 500 and np.abs(pfsdiff) < self.steps * self.step_size: #TODO too big?
                self.logger.info(f"Trying autofocus from last successful position shifts: (z={zdiff}, pfs={pfsdiff})")
                self.experiment.microscope.move_pfs(last_entry[3], True)
                sleep(self.move_delay)
                test_image = CCT.normalise_image_values(self.experiment.microscope.get_image())
                if self.readjust_cell_camera:
                    adjusted, info, test_image = self.readjust_camera(test_image, info, af_start)
                    if not adjusted:
                        return False, info

                is_focused, metric_values = self.determine_abs_focus(test_image)
                if is_focused:
                    elapsed = time()-af_start
                    self.logger.info(f"Focus at last position, took seconds:{elapsed:.2f}")
                    info["autofocus-status"] = "skip"
                    info["autofocus-elapsed"] = elapsed
                    info["autofocus-skip-metrics"] = metric_values
                    info["autofocus-skip-mode"] = "last"
                    return True, info
            elif np.abs(pfsdiff) <= 50:
                self.logger.debug(f"Low pfs shift relative to last cell") #TODO debug
            else:
                self.logger.info(f"Large shift from last in focus cell: (z={zdiff}, pfs={pfsdiff})")

        original_rcc = self.readjust_cell_camera

        self.readjust_cell_camera = False
        result, info_original =  super().run()
        self.readjust_cell_camera = original_rcc

        elapsed = time()-af_start
        info_original["autofocus-elapsed"] = elapsed
        self.logger.info(f"LastTrack autofocus took {elapsed:.2f}s")
        return result, info_original

class RandomForestCellValidator(CellFindingExperiment):
    """Requires a RandomForestAutofocus as preprocessor
    Is this a reasonable inheritance approach (it's a weird composition approach)
    Args:
        CellFindingExperiment (_type_): _description_
    """
    def initialise_validator(self, threshold_validation=0.5, initial_threshold = 0.5):
        self.validation_threshold = threshold_validation
        self.initial_threshold = initial_threshold

        dirname = os.path.dirname(__file__)
        model_file = os.path.join(dirname, "validation_forest.pkl")
        with open(model_file, "rb") as f:
            self.validation_classifier = pickle.load(f)
        self.logger.debug("Loaded validation classifier")
        self.validator_processed = 0

    def get_all_metrics(self, image, center=None, contour=None,radius=None):
        preprocessor:RandomForestAutofocus = self.cell_preprocessor
        if contour is None:
            center, contour = preprocessor.tracker.get_contour(image, center=center, radius=radius)
        if center is not None and contour is not None:
            unrolled = preprocessor.tracker.unroll_around_contour(
                            image, contour, center, 128
                        )
        else:
            #this failed
            self.logger.info("Cell contour not tracked")
            #TODO: remove - hacky test only
            #cv2.imwrite(f"{self.data_folder}/rfv_test/validation-"+ ("final" if final else "initial")+"-lost.jpeg", (CCT.normalise_image_values(image)*255.0).astype(np.uint8))
            #np.array(image).tofile(f"{self.data_folder}/rfv_test/validation-"+ ("final" if final else "initial")+"-lost.npy")
            return False

        #cf = ContourFitter()
        #q_range, good_indices, fft_r, R_mean_pixels, fft_mean = cf.amplitudes_raw([True], [contour])

        focus_metrics = list(preprocessor.get_all_metrics(image, center=center, contour=contour, unrolled=unrolled, skip_normalise=True))
        perc_diff_m ,stdev_m, max_laplace, max_shift = preprocessor.tracker.get_validation_scores(contour)
        row = [perc_diff_m ,stdev_m, max_laplace, max_shift]
        row += focus_metrics
        columns = ["percentile_diff", "stdev", "max_laplace", "max_shift"]
        columns += preprocessor.metric_names()
        if center is not None and not np.isnan(row[0]):
            stds = unrolled.std(axis=1).mean()
            stds_lim = unrolled.std(axis=1)[40:80].mean()
            std_inside = unrolled[:62].std()
            std_outside = unrolled[66:].std()
            #trasformed_width = min(np.abs(np.array(center)-np.array(self.frame_size)).max()-1, 60)
            #start_at = max(0,int(contour.mean()-30)
            unrolled_orig = preprocessor.tracker.transform_image(image, center, 60, max(0,int(contour.mean()-30)))
            uncorrected_stds = unrolled_orig.std(axis=1).mean()

            row += [stds, stds_lim, uncorrected_stds,std_inside, std_outside]
        else:
            #this failed
            self.logger.info("Cell not found after focus")
            #TODO: remove - hacky test only
            #cv2.imwrite(f"{self.data_folder}/rfv_test/validation-"+ ("final" if final else "initial")+"-lost.jpeg", (CCT.normalise_image_values(image)*255.0).astype(np.uint8))
            #np.array(image).tofile(f"{self.data_folder}/rfv_test/validation-"+ ("final" if final else "initial")+"-lost.npy")
            return False

        columns += ["corrected_unrolled_stds", "corrected_unrolled_stds_lim", "unrolled_stds","std_inside","std_outside"]

        #row += list(np.abs(fft_r[0][2:9]))

        return row

    def get_metric_columns(self):
        return (
            ["percentile_diff", "stdev", "max_laplace", "max_shift"] +
            self.cell_preprocessor.metric_names() +
            ["corrected_unrolled_stds", "corrected_unrolled_stds_lim", "unrolled_stds","std_inside","std_outside"]#+
            #[f"mode_{i}" for i in range(2,9)]
        )

    def validate_cell(self, image, final=False, center=None, contour=None):
        self.validator_processed += 1
        try:
            metric_values =  self.get_all_metrics(image, center=center, contour=contour)

            if metric_values is False:
                self.logger.info("Invalid focus metrics")
                return False

            metric_values = pd.DataFrame(data=[metric_values], columns=self.get_metric_columns())

            prob = self.validation_classifier.predict_proba(metric_values)
            if (final and (1-prob[0,1]) < self.validation_threshold) or (not final and (1-prob[0,1]) < self.initial_threshold):
                self.logger.info(f"Validation probability {(1-prob[0,1]):.2f}, rejected")
                #TODO: remove - hacky test only
                cv2.imwrite(f"{self.data_folder}/rfv_test/validation-"+ ("final" if final else "initial")+f"-{self.validator_processed}-invalid.jpeg", (CCT.normalise_image_values(image)*255.0).astype(np.uint8))
                np.array(image).tofile(f"{self.data_folder}/rfv_test/validation" + ("final" if final else "initial")+f"-{self.validator_processed}-invalid.npy")
                return False

            self.logger.info(f"Validation probability {(1-prob[0,1]):.2f}, accepted")
            #TODO: remove - hacky test only
            cv2.imwrite(f"{self.data_folder}/rfv_test/validation-"+ ("final" if final else "initial")+f"-{self.validator_processed}-valid.jpeg", (CCT.normalise_image_values(image)*255.0).astype(np.uint8))
            np.array(image).tofile(f"{self.data_folder}/rfv_test/validation-"+ ("final" if final else "initial")+f"-{self.validator_processed}-valid.npy")
            return True
        except Exception as e:
            self.logger.warning(f"Error in cell validation: ", exc_info=e)
            cv2.imwrite(f"{self.data_folder}/rfv_test/validation-"+ ("final" if final else "initial")+f"-{self.validator_processed}-error.jpeg", (CCT.normalise_image_values(image)*255.0).astype(np.uint8))
            np.array(image).tofile(f"{self.data_folder}/rfv_test/validation-"+ ("final" if final else "initial") + f"-{self.validator_processed}-error.npy")
            return False
