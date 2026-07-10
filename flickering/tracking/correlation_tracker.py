# TODO some of the imports might be unneccessary
import numpy as np
from numba import jit, prange
import numba
import scipy
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.signal import correlate, correlation_lags
import itertools
import cv2
from time import time
from flickering.utils.movie_reader import get_movie_reader
import flickering.utils.visualisation  # import visualise_contours
import os
import types
from more_itertools import peekable
from tqdm.auto import tqdm
from scipy.signal import correlate2d
from numpy.lib.stride_tricks import sliding_window_view

# # from contour_analysis.contour_fitter import  # legacy, removed *
from glob import glob

# from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count
import logging
from threading import RLock
import matplotlib.pyplot as plt
from functools import partial
from flickering.utils.debug import *
import gc
from os.path import exists
from typing import List, Union, Tuple, Generator
from collections.abc import Iterable
from flickering.tracking.contour_io import ContourIO
import datetime

numba.set_num_threads(4)


@jit(nopython=True, parallel=True)
def _normalise_image_values_parallel(image):
    h, w = image.shape
    min_v = image.min()
    max_v = image.max()
    diff = max_v - min_v if max_v != min_v else 1.0
    out = np.empty((h, w), dtype=np.float32)
    for i in prange(h):
        for j in range(w):
            out[i, j] = np.float32(image[i, j] - min_v) / np.float32(diff)
    return out


class CorrelationContourTracker:
    """Contour tracker implementation

    Keeps track of configuration

    TODO: optimise critical loop
    TODO: try to avoid multiple polar->xy conversions where possible

    Raises:
        RuntimeError: _description_
        NotImplementedError: _description_
    """

    METHOD_LINEAR = "linear"
    METHOD_CUBIC = "cubic"
    METHOD_LANCZOS = "lanczos"
    METHOD_NEAREST = "nearest"

    MASK_TYPE_GENERATE = -1
    MASK_TYPE_GAUSS = 0
    MASK_TYPE_GAUSS_DIFF = 1
    MASK_TYPE_GAUSS_NORM = 2

    VERSION = 1

    def __init__(self) -> None:
        # TODO configuration
        self.hough_dp = 1
        self.hough_min_r = 40 // 2
        self.hough_max_r = 400 // 2
        self.hough_min_d = 100
        self.hough_param_1 = 55
        self.hough_param_2 = 33
        self.mask_width = (
            30  # select this many pixels around the x direction max to correlate with
        )
        self.correlate_width = 80  # CORRELATE_WIDTH": 150,  # maximum deviation
        self.ignore_center_rad = 15
        self.max_shift = 5
        self.interpolation_method = CorrelationContourTracker.METHOD_LINEAR
        self.refine_interpolation_method = CorrelationContourTracker.METHOD_LINEAR
        self.mask_type = CorrelationContourTracker.MASK_TYPE_GENERATE
        self.rays = 360  # MUST BE divisible by 4
        self.refine_center_max_iterations = 10
        self.refine_center_tolerance = 0.1  # pixels
        self.logger = logging.getLogger("CONTOUR")
        self.epsilon = 1e-9
        # how many centers to use for mean center, set to 0 to use
        # hough on every frame instead of starting with mean
        self.use_centers_mean = 20

        # contour validation
        self.percentile_th = 0.35  # maximum 90th-10th percentile/mean radius
        self.std_dev_th = 0.15  # 0.15  # maximum std dev/mean readius.

        self.laplace_th = (
            0.0012  # this is applied after removing outliers with a gaussian filter
        )
        self.max_shift_th = 2  # maximum pixel difference between neighbour points

        self.threads = max(cpu_count() - 1, 1)
        self.progress_bar = True

        self.debug = False
        self.debug_data = {"masks": [], "validation_scores": []}

        self._refined_mask_failed = False  # flag to detect mask refinement failure, used in fixed mask mode to try obtaining mask from different frame
        self.use_fixed_mask = False
        self._mask = None
        self._mask_offset = 0

        # unroll image around a detected contour to get a better mask
        # currently only implemented with use_fixed_mask (TODO: but also it will be slow otherwise)
        self.mask_refinement = False
        self.mask_refinement_shift_to_max = False

        self._mask_ready = False

        self.refine_correlation = True
        self.parabolic_fit_width = 5  # SHOULD BE ODD

        self.subtract_means = True
        # search this range around given initial radius, None for any. Only use if good initial radius is available
        self.radius_search_range = 10

        # when not using mean center, use the radius from the first contour found
        # this will cause the whole video tracking to fail if the frist contour is incorrect!
        self.use_radius_initial = False

        self.save_mode = "R"
        self.radial_only_interpolation = False

    def scale_config(self, scale):
        """Scale size dependent variables (i.e. multiply everything in pixel units by scale)

        Default scale roughly corresponds to human RBCs with 60x, 1.5x lens, Grasshopper

        Args:
            scale (float): Multiplier, if cells are larger in the images should be >1.
        """
        self.hough_min_r = round(scale * self.hough_min_r)
        self.hough_max_r = round(scale * self.hough_max_r)
        self.hough_min_d = round(scale * self.hough_min_d)
        self.mask_width = round(scale * self.mask_width)
        self.correlate_width = round(scale * self.correlate_width)
        self.ignore_center_rad = round(self.ignore_center_rad * scale)
        self.max_shift = round(self.max_shift * scale)

    def transform_image(
        self, image: np.ndarray, center: np.ndarray, output_width: int, start_at: int
    ):
        start_at = max(start_at, 0)
        if self.interpolation_method == CorrelationContourTracker.METHOD_LINEAR:
            interp_mode = cv2.INTER_LINEAR
        elif self.interpolation_method == CorrelationContourTracker.METHOD_CUBIC:
            interp_mode = cv2.INTER_CUBIC
        elif self.interpolation_method == CorrelationContourTracker.METHOD_LANCZOS:
            interp_mode = cv2.INTER_LANCZOS4
        elif self.interpolation_method == CorrelationContourTracker.METHOD_NEAREST:
            interp_mode = cv2.INTER_NEAREST
        else:
            interp_mode = cv2.INTER_LINEAR

        interp_mode += cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
        # interp_mode = cv2.INTER_LINEAR if self.interpolation_method == CorrelationContourTracker.METHOD_LINEAR
        # start_at is not supported here
        # 73, 360, [130, 132], (73, 9)
        # self.logger.info(f"{image.dtype}, {image.shape}, {output_width+start_at}, {self.rays}, {[int(center[1]), int(center[0])]}, {output_width+start_at, interp_mode}")
        transformed_cv = cv2.warpPolar(
            image,
            (output_width + start_at, self.rays),
            [int(center[1]), int(center[0])],
            output_width + start_at,
            interp_mode,
        )
        transformed_cv = np.roll(
            np.flip(np.transpose(transformed_cv), axis=1), shift=self.rays // 4, axis=1
        )  # TODO: enforce rays%4=0
        return transformed_cv[start_at:, :]

    def unroll_around_contour(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        center: np.ndarray,
        output_width: int,
    ) -> np.ndarray:
        """Unroll image.
        Uses cv2.remap for performance and high-quality interpolation.
        """
        unroll_starts = contour - output_width // 2
        average = image.mean()

        angles = np.linspace(0, 2 * np.pi, self.rays, endpoint=False)
        rs = np.arange(output_width)

        # Coordinate maps: (radial_offset, angle)
        # rs_grid[i, j] = i, angles_grid[i, j] = angles[j]
        rs_grid, angles_grid = np.meshgrid(rs, angles, indexing="ij")

        # High-precision radial-only mode (Linear Radial, Nearest Tangential)
        # maintained for compatibility, but Lanczos is often superior.
        if self.radial_only_interpolation:
            xs = np.arange(image.shape[0])
            ys = np.arange(image.shape[1])
            rs_repeated = rs_grid.ravel() + np.tile(unroll_starts, (output_width,))
            dirs_repeated = np.tile(
                np.array([np.cos(angles), np.sin(angles)]), (1, output_width)
            )

            r_floor = np.floor(rs_repeated)
            r_ceil = r_floor + 1
            delta = rs_repeated - r_floor

            interpolated_nearest = RegularGridInterpolator(
                (xs, ys),
                image,
                method="nearest",
                bounds_error=False,
                fill_value=average,
            )

            center_arr = np.array(center)[:, np.newaxis]
            coords_floor = center_arr + r_floor * dirs_repeated
            coords_ceil = center_arr + r_ceil * dirs_repeated

            val_floor = interpolated_nearest(np.transpose(coords_floor))
            val_ceil = interpolated_nearest(np.transpose(coords_ceil))

            output_flat = (1 - delta) * val_floor + delta * val_ceil
            return np.reshape(output_flat, (output_width, self.rays))

        # Standard mode using cv2.remap
        if self.refine_interpolation_method == self.METHOD_LINEAR:
            interp_mode = cv2.INTER_LINEAR
        elif self.refine_interpolation_method == self.METHOD_CUBIC:
            interp_mode = cv2.INTER_CUBIC
        elif self.refine_interpolation_method == self.METHOD_LANCZOS:
            interp_mode = cv2.INTER_LANCZOS4
        elif self.refine_interpolation_method == self.METHOD_NEAREST:
            interp_mode = cv2.INTER_NEAREST
        else:
            interp_mode = cv2.INTER_LINEAR

        cur_rs = rs_grid + unroll_starts[np.newaxis, :]
        # map_x/y indices: [row, col] -> [r, theta]
        # map_x (X-coords for OpenCV) must take center[1] (X) + sin(theta) component
        # map_y (Y-coords for OpenCV) must take center[0] (Y) + cos(theta) component
        # to match the legacy convention.
        map_x = (center[1] + cur_rs * np.sin(angles_grid)).astype(np.float32)
        map_y = (center[0] + cur_rs * np.cos(angles_grid)).astype(np.float32)

        # map_x/y indices: [row, col] -> [r, theta]
        # In cv2.remap, rows of maps correspond to rows of output.
        # Our output is (output_width, self.rays), so map rows are radial offsets.
        output = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=interp_mode,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(average),
        )

        return output
        # if DISPLAY and DEBUG:
        #    cv2.imshow("transformed_image", output)
        #    cv2.waitKey(WAIT_KEY)
        return output

    def mean_center(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        center = np.array([0.0, 0.0])
        radii = 0
        n = 0
        for frame in frames:
            cell_center, radius = self.find_cell_center(frame, refine=True)
            if cell_center is None:
                self.logger.debug("Failed to find cell center, skipping frame in mean")
                continue
            center += cell_center
            radii += radius
            n += 1

        if n == 0:
            self.logger.error("No centers found in mean center")
            raise RuntimeError("No cell centers found!")

        return center / n, radii / n

    def find_contour_in_unrolled(
        self, unrolled, correlation_mask, ignore_in_unrolled=0, contour_start=None
    ):
        lags = correlation_lags(
            unrolled.shape[0], correlation_mask.shape[0], mode="valid"
        )
        # TODO: is there a faster way to do this?
        # turns out there is (but only slightly), correlate2d
        # correlations = np.apply_along_axis(
        #    correlate, 0, unrolled, correlation_mask, mode="valid"
        # )

        # mask_2d = np.expand_dims(correlation_mask, 1)
        # correlations = correlate2d(unrolled, mask_2d, mode='valid')

        # this seems much faster than correlate2d
        windowed_array = sliding_window_view(
            unrolled, window_shape=(correlation_mask.size,), axis=0
        )
        correlations = np.dot(windowed_array, correlation_mask)

        debug_display("correlations", correlations)
        # print(f"Contour start: {contour_start}")
        # this is a little nasty: we want sub-pixel resolution, so we interpolate in the map from correlation
        # indices to original image indices (radii)
        interpolate_indices = np.array(list(range(correlations.shape[0])))
        interpolated_lags = interp1d(interpolate_indices, lags)
        ignore_in_correlations = max(0, ignore_in_unrolled - self.mask_width // 2)

        if (
            contour_start is not None
        ):  # this determines the region where to start looking for the contour
            r_orig = (
                contour_start - self.mask_width // 2
            )  # should this be + to map to lag
            # alternative options for starting contour position
            # r_from_0 = np.argmax(correlations[:, 0])
            # r_max_mean = np.argmax(correlations.mean(axis=1))
            # r_mean_max = np.mean(np.argmax(correlations, axis=0))
            # r_freq_max = np.argmax(
            #    np.bincount(np.argmax(correlations, axis=0))
            # )  # this relies on a low resolution...
            # r_max = np.mean(np.argmax(correlations, axis=0))

            r = r_orig  # expected lag, not contour position
        else:
            r = np.argmax(correlations[:, 0])  # TODO is this used?

        running_average = 0
        average_last = 5
        max_location = 0
        maxima = []  # somehow list appends are faster

        # The loop is unavoidable if we want to keep the moving start
        # it could be replaced if we force circularity rather than continuity
        # at this stage of the detection
        # Numba makes this MUCH slower for some reason
        for i in range(correlations.shape[1]):
            # This is the performance critical loop, runs~2*(360-3600) times/per frame (mean~5000)
            # for a 20s 500fps video -> ~5e7 times per video, each microsecond counts (as 50s)
            # check the impact of any changes
            if i >= average_last:
                r = int(
                    (running_average / average_last + max_location) / 2.0 + 0.5
                )  # this is faster than round
                # average with extra weight for the last point
                # calling mean is too slow
                running_average -= maxima[-average_last]
            start_r = max(ignore_in_correlations, int(r - self.max_shift))
            end_r = min(int(r + self.max_shift), correlations.shape[0])
            if end_r - start_r < 2 or start_r < 0 or end_r < 0:
                self.logger.warning(
                    f"Contour drifted out of range {start_r}-{end_r}, r={r}, {i}, {ignore_in_correlations}, {correlations.shape} skipping frame"
                )  # TODO: better error message
                return None, None  # is this really terminal?
            max_location = correlations[start_r:end_r, i].argmax() + start_r
            maxima.append(max_location)
            running_average += max_location

        starts = np.array(maxima) - self.parabolic_fit_width // 2
        clipped = starts.clip(0, correlations.shape[0] - self.parabolic_fit_width)
        edge_counter = np.sum(starts != clipped)

        # this is some numpy magic to select the region
        row_indices = np.arange(self.parabolic_fit_width)[:, None] + clipped[None, :]
        col_indices = np.arange(correlations.shape[1])
        max_correlation_areas = correlations[row_indices, col_indices]

        # TODO: make this an option?
        # curve fit apparently isn't vectorised?
        # fit_maxs = []
        # for i in range(rays):
        #    coeffs, cov = curve_fit(lambda x, a, b, c: a*np.exp(-(x-b)**2/c),np.arange(5), max_correlation_areas[:,i])
        #     print(coeffs)
        #     print(coeffs.shape)
        #     fit_maxs.append([1])
        debug_display("correlation_areas", max_correlation_areas)

        coeffs = np.polyfit(
            np.arange(self.parabolic_fit_width), max_correlation_areas, 2
        )
        # TODO: iterate the fit location selection process - select parabolic_fit_width around the newly found max
        #       this was done in previous work, I'm not conviced it's necessary or beneficial - we already reject
        #       excessive shifts
        # logger.debug(f"polyfit output: {coeffs}")
        # find maximum of the parabola
        fit_maxs = -coeffs[1] / (2 * coeffs[0] + self.epsilon)

        # if the fit fails, we use the pixel-level maximum
        fit_maxs[fit_maxs < 0] = 2
        fit_maxs[fit_maxs > self.parabolic_fit_width - 1] = (
            self.parabolic_fit_width // 2
        )
        fit_maxs += clipped

        # the contour is in the center of the mask
        subpixel_offsets = interpolated_lags(fit_maxs) + self.mask_width // 2

        return subpixel_offsets, edge_counter

    def get_contour(
        self,
        image: np.ndarray,
        center: Union[np.ndarray, None] = None,
        iteration=0,
        use_mask=None,
        radius=None,
    ) -> Tuple[Union[np.ndarray, None], np.ndarray]:
        """_summary_

        Args:
            frame (np.array): image

        Returns:
            tuple[np.array, np.array]: Center, Radii
        """
        # display=False,

        image = CorrelationContourTracker.normalise_image_values(image)
        debug_display("contour_detection", image)

        if center is None:
            center, radius_new = self.find_cell_center(
                image, radius=radius, refine=True
            )
            if center is None:
                self.logger.warning("No center found, skipping frame")
                return None, np.array([])
            if radius is None:
                radius = radius_new
        # this assumes no extrapolation
        max_rad = (
            min(np.min(np.array(image.shape) - center), np.min(center)) + 30
        )  # TODO handle extrapolated better

        if radius is None:
            # we want to get the approximate location of the contour before unrolling
            # so that we don't need to unroll the entire image
            # we can trivially look at 4 points
            center_int = np.array(center).round().astype(int)
            # logger.debug(f"center_int = {center_int}")
            try:
                max_u = self.ignore_center_rad + np.argmax(
                    image[center_int[0] + self.ignore_center_rad :, center_int[1]]
                )
                max_d = center_int[0] - np.argmax(
                    image[: center_int[0] - self.ignore_center_rad, center_int[1]]
                )
                max_l = center_int[1] - np.argmax(
                    image[center_int[0], : center_int[1] - self.ignore_center_rad]
                )
                max_r = self.ignore_center_rad + np.argmax(
                    image[int(center[0]), int(center[1]) + self.ignore_center_rad :]
                )
            except:
                self.logger.warning("Detected center is invalid, skipping frame")
                self.logger.debug("Exeption details:", exc_info=True)
                return None, np.array([])

            # TODO: use radius estimate from hough/previous iteration?
            max_positions = np.array([max_u, max_d, max_l, max_r])
            contour_start = max_positions.mean()
            contour_start_median = np.median(max_positions)
            if np.abs(contour_start - contour_start_median) > 4:
                self.logger.debug(
                    "Outlier max position, using median for initial contour position"
                )
                contour_start = contour_start_median

            contour_start = np.round(contour_start).astype(int)
        else:  # start at the radius provided, only relevant for unrolling
            contour_start = np.round(radius).astype(int)

        if (
            contour_start < self.mask_width // 2
            or contour_start + self.mask_width // 2 >= max_rad
        ):
            self.logger.warning(
                f"First contour start ({contour_start}) out of range (min-max: {self.mask_width//2}-{max_rad-self.mask_width // 2}),center: {center} skipping cell"
            )
            if radius is not None:
                self.logger.warning(
                    f"This originated as incorrect radius value {radius}"
                )
            return None, np.array([])

        # logger.debug(f"First contour start ({contour_start})")

        unrolled_start = max(0, contour_start - self.correlate_width // 2)

        ignore_in_unrolled = int(max(0, self.ignore_center_rad - unrolled_start))
        width = int(min(self.correlate_width, max_rad - unrolled_start))
        # if the contour starts too high in r, we want unrolled_start+width=max_rad => width = max_rad-unrolled_start
        # print(max_rad, contour_start, width)
        # print(center,  width, unrolled_start, ignore_in_unrolled)

        unrolled = self.transform_image(image, center, width, unrolled_start)

        mean = unrolled.mean(axis=1)

        debug_display("unrolled", unrolled)

        mask_offset = 0  # wa assume the contour is in the center of the mask but rounding breaks this
        # TODO: if we got the radius, should we still do this?
        if False and radius is not None and self.radius_search_range is not None:
            contour_start_initial = contour_start
            search_start = max(
                ignore_in_unrolled,
                int(radius - unrolled_start - self.radius_search_range),
            )
            search_end = max(
                width, int(radius - unrolled_start + self.radius_search_range)
            )
            contour_start = np.argmax(mean[search_start:search_end]) + search_start

            if np.abs(contour_start_initial - contour_start) > 10 and self.debug:
                self.logger.info(
                    f"Radius {contour_start_initial}->{contour_start}, radius={radius}"
                )
        elif radius is not None:  # trust the radius provided
            contour_start = int(radius - unrolled_start)
            contour_start_initial = contour_start
            # self.logger.debug(f"Radius {contour_start_initial}->{contour_start}, radius={radius}")
        else:
            contour_start_initial = contour_start
            contour_start = np.argmax(mean[ignore_in_unrolled:]) + ignore_in_unrolled

        # sanity check for contour start, can potentially be changed to use the uncorrected start
        if (
            contour_start < self.mask_width // 2
            or contour_start + self.mask_width / 2 >= max_rad
        ):
            self.logger.warning(
                f"Corrected contour start ({contour_start}) ({contour_start_initial}->{unrolled_start}+{ignore_in_unrolled}+{contour_start}) out of range ({self.mask_width // 2} - {max_rad}), using initial"
            )
            contour_start = contour_start_initial
            # return None, np.array([])

        # this is the mask we are correlating against
        new_mask_generated = False
        if use_mask is not None:
            correlation_mask = use_mask
        elif (
            self.use_fixed_mask and self._mask is not None and self._mask_ready
        ):  # TODO: there's probably a cleaner way now we can pass mask...
            correlation_mask = self._mask
            mask_offset = self._mask_offset
        elif self.mask_type == CorrelationContourTracker.MASK_TYPE_GENERATE:
            new_mask_generated = True
            correlation_mask = mean[
                contour_start
                - self.mask_width // 2 : contour_start
                + self.mask_width // 2
            ]

            mask_offset = radius - contour_start - unrolled_start
        else:
            correlation_mask = CorrelationContourTracker.gauss_mask(
                self.mask_width, 1.5, self.mask_type
            )  # TODO: make std adjustable

        if self.mask_type > 0:
            correlation_mask = -correlation_mask

        if (
            self.subtract_means
            and self.mask_type == CorrelationContourTracker.MASK_TYPE_GENERATE
        ):
            # TODO: how do we calculate the means?
            # use the start and end of unrolled?
            # TODO: is this length sensible?
            use_mean_parts = (self.correlate_width - self.mask_width) // 2
            mean_for_subtract = (
                mean[:use_mean_parts].mean() + mean[-use_mean_parts:].mean()
            ) / 2
            unrolled -= mean_for_subtract
            if new_mask_generated:
                correlation_mask -= mean_for_subtract

        debug_display("mask", correlation_mask)

        # TODO: remove this duplication, _mask should now have this, there's also debug_data['masks']
        self.debug_data["mask"] = correlation_mask
        # self.debug_data["mask_offset"] = mask_offset

        # the contour is in the center of the mask
        # contour_start needs to be the initial point on the contour. For excentric cells this is too far from the mean radius
        # so this either requires high max_shift or we need to keep track of the start point rather than the mean
        # here we pick the maximum correlation point at that azimuth, which will fail if there is a better match on that line
        # and the contour will likely not recover
        subpixel_offsets, edge_counter = self.find_contour_in_unrolled(
            unrolled, correlation_mask, ignore_in_unrolled, contour_start
        )

        if subpixel_offsets is None:
            self.logger.warning("Frame processing failed")
            return [], None

        subpixel_offsets += unrolled_start
        subpixel_offsets += mask_offset

        if (self.mask_refinement and not self._mask_ready) or self.refine_correlation:
            new_unrolled_width = self.correlate_width
            unrolled_around_contour = self.unroll_around_contour(
                image, subpixel_offsets, center, new_unrolled_width
            )

        # TODO: make this efficient & easy to configure
        # draw_contour(image, subpixel_offsets, [center[1], center[0]], display=True)
        # avoid unnecessery work if we are reusing the same mask
        # if using fixed mask this is disabled after first frame by _mask_ready
        # self._mask = [correlation_mask]
        if self.mask_refinement and not self._mask_ready:
            # we want to make sure the position of the contour mask doesn't shift
            # currently for some reason with smaller mask widths (<36?) it gets shifted away from the center
            # extra 15 each side
            original_mask = correlation_mask
            correlation_mask = unrolled_around_contour.mean(axis=1)
            if (
                self.subtract_means
                and self.mask_type == CorrelationContourTracker.MASK_TYPE_GENERATE
            ):
                # TODO: how do we calculate the means?
                # use the start and end of unrolled?
                # TODO: is this length sensible?
                # TODO: code duplication
                use_mean_parts = (self.correlate_width - self.mask_width) // 2
                mean_for_subtract = (
                    mean[:use_mean_parts].mean() + mean[-use_mean_parts:].mean()
                ) / 2
                correlation_mask -= mean_for_subtract
                unrolled_around_contour -= mean_for_subtract

            debug_display(
                "CONTOUR_UNROLLED", unrolled_around_contour, "unrolled around contour"
            )

            if self.mask_refinement_shift_to_max:
                new_max_position = np.argmax(correlation_mask)

                if (
                    new_max_position < self.mask_width // 2
                    or new_max_position + self.mask_width // 2
                    > unrolled_around_contour.shape[0]
                ):
                    self.logger.warning(
                        f"Mask could not be shifted to center! (max@{new_max_position}), reusing original mask"
                    )
                    # TODO: handle better?
                    correlation_mask = original_mask
                    self._refined_mask_failed = True
                else:
                    # self._mask.append(correlation_mask)
                    correlation_mask = correlation_mask[
                        new_max_position
                        - self.mask_width // 2 : new_max_position
                        + self.mask_width // 2
                    ]
                    if self._refined_mask_failed:  # TODO: change to info
                        self.logger.warning("Refined mask recovered")
                        self._refined_mask_failed = False

                    mask_offset = (
                        0  # we applied the offset when unrolling around contour
                    )
            else:
                mask_offset = 0
                correlation_mask = correlation_mask[
                    len(correlation_mask) // 2
                    - self.mask_width // 2 : len(correlation_mask) // 2
                    + self.mask_width // 2
                ]

        self._mask = correlation_mask
        self._mask_offset = mask_offset

        if self.refine_correlation:
            # this is probably not needed and might even be counterproductive (more interpolation)
            # we now repeat the fitting but on image unrolled around the contour
            # contour is around new_unrolled_width // 2 in the unrolled
            original_contour_mean_r = subpixel_offsets.mean()
            ignore_in_corrected_unrolled = int(
                max(
                    0,
                    self.ignore_center_rad
                    + new_unrolled_width / 2
                    - original_contour_mean_r,
                )
            )
            subpixel_offsets_corrected, edge_counter_corrected = (
                self.find_contour_in_unrolled(
                    unrolled_around_contour,
                    correlation_mask,
                    ignore_in_corrected_unrolled,
                    new_unrolled_width // 2,
                )
            )
            if subpixel_offsets_corrected is None:
                self.logger.warning(
                    "Failed to find contour in second phase fit, using initial estimate!"
                )
            else:
                subpixel_offsets = (
                    subpixel_offsets_corrected
                    + subpixel_offsets
                    - new_unrolled_width // 2
                )  # TODO: shift by

                edge_counter = edge_counter_corrected

        if self.refine_center_max_iterations > iteration:
            xy_contour = CorrelationContourTracker.convert_contour_xy(
                subpixel_offsets, np.array([0.0, 0.0])
            )
            refined_center = np.array(xy_contour).mean(axis=0)
            shift = np.linalg.norm(refined_center)
            if self.logger.level == logging.DEBUG:  # prevent rendering of fstring
                self.logger.debug(
                    f"Refining centre position, shift {shift} ({center}+{refined_center})"
                )
            if (
                self.refine_center_tolerance > 0
                and shift < self.refine_center_tolerance
            ):
                # we only want the warning on the final pass
                if edge_counter > 0:
                    self.logger.warning(
                        f"Max correlation near edge of region {edge_counter} times"
                    )
                self.logger.debug("Centre position change below threshold, returning")
                return center + refined_center, subpixel_offsets
            new_offsets = xy_contour - refined_center
            new_rad = np.sqrt(new_offsets[:, 0] ** 2 + new_offsets[:, 1] ** 2).mean()
            return self.get_contour(
                image,
                center + refined_center,
                iteration + 1,
                use_mask=correlation_mask if self.mask_refinement else None,
                radius=new_rad,  # TODO: is this too slow? We can reuse the radius as changes are small if yes
            )

        # we only want the warning on the final pass
        if edge_counter > 0:
            self.logger.warning(
                f"Max correlation near edge of region {edge_counter} times"
            )
        # self.logger.warning(f"Center location did not converge.")
        return center, subpixel_offsets

    def process_sequence(
        self,
        frames: Union[List[np.ndarray], Generator[np.ndarray, None, None]],
        center: Union[np.ndarray, None] = None,
        seq_len=None,
        radius=None,
    ) -> ContourIO:
        self._mask_ready = (
            False  # reset mask ready flag to avoid mask reuse in certain configs
        )
        processing_start = time()
        if seq_len:
            n_frames = seq_len
        elif not isinstance(frames, types.GeneratorType):
            n_frames = len(frames)
        else:
            n_frames = None

        first_frame = None
        # TODO clean up this tree
        if center is not None:
            mean_center = center
            mean_radius = radius
        elif self.use_centers_mean:
            if isinstance(frames, types.GeneratorType):
                self.logger.debug("Processing Generator type frames")
                if self.use_centers_mean > 1:
                    raise NotImplementedError(
                        "Using multiple frames for center with a Frame Generator is not supported"
                    )
                if seq_len is None:
                    raise NotImplementedError(
                        "Generator and use_centers_mean requires seq_len"
                    )

                frames = peekable(frames)
                first_frame = frames.peek()
                mean_center, mean_radius = self.mean_center([first_frame])
            else:
                n_frames = len(frames)
                indices = slice(0, n_frames, n_frames // self.use_centers_mean)
                mean_center, mean_radius = self.mean_center(frames[indices])
        else:
            mean_center = None
            mean_radius = None
        self.logger.info(f"Using mean center {mean_center}")

        if self.use_fixed_mask:  # this will prepare the mask
            if isinstance(frames, types.GeneratorType):
                frames = peekable(frames)

            first_contour, _, _ = self.process_frame(
                first_frame if first_frame is not None else frames[0],
                mean_center,
                radius=mean_radius,
            )
            if self.mask_refinement and self._refined_mask_failed:
                self.logger.error(
                    "Failed to identify refined mask in first frame, using frame based center and radius"
                )
                first_contour, _, _ = self.process_frame(
                    first_frame if first_frame is not None else frames[0]
                )

            if self.mask_refinement and self._refined_mask_failed:
                self.logger.error(
                    "Failed to identify refined mask in first frame, retrying with frame 100"
                )
                first_contour, _, _ = self.process_frame(frames[100])

            self._mask_ready = (
                True  # this will prevent mask_refiniment for future frames
            )
            if self.use_radius_initial:
                if mean_radius is None:
                    mean_radius = first_contour.mean()

                self.logger.info(f"Using radius {mean_radius} from first contour")
        # TODO can this handle large datasets (i.e. > memory size?)
        if self.threads > 1:
            with Pool(self.threads) as p:
                self.logger.debug("Starting processing threads")
                if n_frames > 0 and self.progress_bar:
                    results = []
                    for r in tqdm(
                        p.imap(
                            lambda f: self.process_frame(
                                f, mean_center, radius=mean_radius
                            ),
                            frames,
                        ),
                        total=n_frames,
                    ):
                        results.append(r)
                else:
                    results = p.imap(
                        lambda f: self.process_frame(
                            f, mean_center, radius=mean_radius
                        ),
                        frames,
                    )
        else:
            results = []
            if self.progress_bar:
                for f in tqdm(frames, total=n_frames):
                    results.append(
                        self.process_frame(f, mean_center, radius=mean_radius)
                    )
            else:
                for f in frames:
                    results.append(
                        self.process_frame(f, mean_center, radius=mean_radius)
                    )

        total_n = 0
        cio = ContourIO()
        cio.tracker_version = CorrelationContourTracker.VERSION
        cio.mode = self.save_mode
        valid_n = 0
        for result in results:
            cio.add_contour(
                result[0], result[1], None if self.save_mode == "XY" else result[2]
            )
            if result[1]:
                valid_n += 1
            total_n += 1

        # does this stop the processes?
        # if self.threads > 1:
        #    p.close()
        processing_end = time()
        # visualise_contours(
        #    cio.contours, cio.valid_indices, contour_filename + "-preview.png"
        # )
        self.logger.info(
            f"Processed {valid_n}/{total_n} frames in {round(processing_end-processing_start,2)}s, {round(total_n/(processing_end-processing_start),2)} fps"
        )

        return cio

    def find_cell_center(self, image: np.ndarray, refine=False, radius=None):
        """_summary_

        Args:
            image (np.ndarray): _description_
            hough_parameters (dict, optional): _description_. Defaults to HOUGH_PARAMETERS.

        Returns:
            Tuple[center, radius_px]: center is [x_pixel, y_pixel] (TODO: check), radius_px is int
        """
        if image.max() > 1.0:
            image = CorrelationContourTracker.normalise_image_values(image)

        denoised = cv2.fastNlMeansDenoising(
            (image * 255.0).astype(np.uint8), None, 40, 17, 13
        )

        circles = cv2.HoughCircles(
            denoised,
            cv2.HOUGH_GRADIENT,
            self.hough_dp,
            self.hough_min_d,
            param1=self.hough_param_1,
            param2=self.hough_param_2,
            minRadius=(
                self.hough_min_r
                if radius is None
                else int(radius - self.radius_search_range)
            ),
            maxRadius=(
                self.hough_max_r
                if radius is None
                else int(radius + self.radius_search_range)
            ),
        )
        in_fallback = False
        image_center = np.array(image.shape) / 2

        if circles is None:
            self.logger.warning("No circles detected when analysing cell!")
            debug_display("uncircled_image", denoised)
            if not refine:
                return None, None
            in_fallback = True
            best_center = image_center
            best_radius = None
        else:
            best_center = None
            best_radius = None
            min_dist = 1e9
            for c in circles[0, :]:
                # TODO: save radius as useful parameter of image for setting SCALE
                radius_px = int(c[2])
                center = [int(c[1]), int(c[0])]
                # draw the outer circle
                cv2.circle(denoised, (center[1], center[0]), radius_px, 1.0, 2)
                # draw the center of the circle
                cv2.circle(denoised, (center[1], center[0]), 2, 0.0, 3)
                debug_display("CIRCLE", denoised)
                self.logger.debug(f"Circle center = {center}, radius_px = {radius_px}")
                # print(center)
                # TODO: is this correct XY order?
                if (
                    best_center is None
                    or np.linalg.norm(image_center - np.array(center)) < min_dist
                ):
                    min_dist = np.linalg.norm(image_center - center)
                    best_center = center
                    best_radius = radius_px

        self.logger.debug(f"Selected center = {best_center}, radius_px = {best_radius}")

        # this is a bit inconsistent, so we try to track and use the tracked version
        # this is only used in initial frame processing, mostly to ensure the mask is consistent
        if refine:
            # this is a bit of mess, we need no refinement, fixed mask, and finer center tolerance
            orig_mask_ready = self._mask_ready
            orig_refine_threshold = self.refine_center_tolerance
            orig_refine_steps = self.refine_center_max_iterations
            orig_mask = self.mask_type
            orig_mask_refinement = self.mask_refinement
            orig_refine_corr = self.refine_correlation
            self.refine_correlation = True
            self.mask_refinement = False
            self.refine_center_max_iterations = 900
            # more consistency
            self.mask_type = CorrelationContourTracker.MASK_TYPE_GAUSS_DIFF
            self.refine_center_tolerance = 5e-3
            refined_center, contour = self.get_contour(
                image, best_center, radius=best_radius
            )  # do we need best radius?
            if refined_center is None or contour is None:
                self.logger.warning("Failed to refine center, using initial estimate")
                if in_fallback:
                    self.logger.error(
                        "Failed to refine center, no initial estimate available"
                    )
                    return None, None
                return best_center, best_radius
            best_radius = contour.mean()
            self._mask_ready = orig_mask_ready  # reset this flag
            self.refine_center_tolerance = orig_refine_threshold
            self.refine_center_max_iterations = orig_refine_steps
            self.mask_type = orig_mask
            self.refine_correlation = orig_refine_corr
            self.mask_refinement = orig_mask_refinement
            self.logger.debug(
                f"Refined center finding: {best_center}->{refined_center}"
            )
            return (
                refined_center,
                contour.mean(),
            )  # np.round(refined_center).astype(int), int(np.round(contour.mean()))
        return best_center, best_radius

    def get_validation_scores(self, contour_points):
        mean = np.average(contour_points)
        stdev = np.std(contour_points)
        if len(contour_points) < 10:
            return np.nan, stdev / mean, np.nan, np.nan

        perc_diff = np.percentile(contour_points, 95) - np.percentile(contour_points, 5)

        max_shift = np.abs(np.roll(contour_points, 1) - contour_points).max()

        contour_points_filtered = scipy.ndimage.gaussian_filter1d(
            contour_points, 3, mode="wrap"
        )

        # laplace might detect spikes
        laplacians = (
            scipy.ndimage.laplace(np.roll(contour_points_filtered, self.rays // 2))
            / mean
        )
        max_laplace = np.max(np.abs(laplacians[1:-1]))

        return perc_diff / mean, stdev / mean, max_laplace, max_shift

    def validate_contour(self, contour_points: np.ndarray, center=[0, 0]) -> bool:
        """Check if contour is sensible

        Args:
            contour_points (np.array): List of radii
            center (list): UNUSED

        Returns:
            bool: True if the contour for that frame satisfies the validity criteria
        """
        if len(contour_points) == 0:
            return False

        perc_diff_m, stdev_m, max_laplace, max_shift = self.get_validation_scores(
            contour_points
        )

        validity = (
            perc_diff_m < self.percentile_th
            and stdev_m < self.std_dev_th
            and max_laplace < self.laplace_th
            and max_shift < self.max_shift_th
        )

        if self.debug:
            self.debug_data["validation_scores"].append(
                [perc_diff_m, stdev_m, max_laplace, max_shift]
            )

        if not validity:
            self.logger.warning("Contour for this frame not sensible, frame rejected")

        return validity

    def process_frame(
        self, f: np.ndarray, center: Union[np.ndarray, None] = None, radius=None
    ) -> Tuple[np.ndarray, bool]:
        """_summary_

        Args:
            f (np.ndarray): image frame to process
            center (_type_): _description_
            refine_center (int, optional): _description_. Defaults to 10.
            correlate_parameters (dict, optional): _description_. Defaults to CORRELATION_PARAMETERS.
            hough_parameters (dict, optional): _description_. Defaults to HOUGH_PARAMETERS.

        Returns:
            List: [[contour_pixels],validity_bool]
        """
        # logger.debug(f"Frame: {f}")
        refined_center, contour = self.get_contour(f, center, radius=radius)

        valid = True
        if contour is None or len(contour) < 1:
            self.logger.debug("No contour detected, skipping frame")
            valid = False
            if self.save_mode == "XY":
                contour_final = np.reshape(
                    np.repeat(np.nan, (self.rays * 2)), (self.rays, 2)
                )  # None #TODO: is this the best option? How about np.nan repeated?
            elif self.save_mode == "R":
                contour_final = np.repeat(np.nan, self.rays)
                refined_center = np.array([np.nan, np.nan])
        else:
            if self.save_mode == "XY":
                contour_final = CorrelationContourTracker.convert_contour_xy(
                    contour, refined_center
                )
            elif self.save_mode == "R":
                contour_final = contour
            else:
                self.logger.error(f"Unsupported save mode: {self.save_mode}")
                raise NotImplementedError(f"Unsupported save mode: {self.save_mode}")
            # if is_first:#TODO: remove, this prints args for cpp contour tracker
            # print(f"{movie_filename} {round(center[0])} {round(center[1])} {round(contour_xy[0][1])} {round(contour_xy[0][0])}") #is this correct ordering?
            #    is_first = False
            valid = valid and self.validate_contour(contour, refined_center)

        if self.threads < 2 and self.debug:
            self.debug_data["masks"].append(self.debug_data["mask"])
        return (contour_final, valid, refined_center)

    @staticmethod
    def contour_exists(movie_filename, ignore_older_than=None):
        """Check if contour file exists

        Args:
            movie_filename: Movie file

        Returns:
            bool: contour file exists
        """
        contour_filename = movie_filename.replace(".movie", f"_contour")

        if not os.path.isfile(contour_filename + ".npz"):
            return False

        if ignore_older_than is not None:
            mtime = os.path.getmtime(contour_filename + ".npz")
            if mtime < ignore_older_than:
                return False
            return True

        return True

    # TODO: should these be here or separate
    # we could probably reduce code duplication a little more
    def process_movie(
        self,
        movie_filename: str,
        contour_filename: str = None,
        visualise=False,
        prevent_overwrite=False,
        return_cio=False,
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """Process a .movie file

        Args:
            movie_filename (string): filename to proccess
            contour_filename (string, optional): where to save contour file. Defaults to .movie replaced by interpolationMethod_contour.
            visualise (bool, optional): save visualisation png next to the contour file. Defaults to False.

        Returns:
            Tuple[List[np.ndarray], List[bool]]: contours, valid_indices as in ContourIO
        """
        self.logger.info(f"Processing {movie_filename}")

        if contour_filename is None:
            if movie_filename.endswith(".mkv"):
                contour_filename = movie_filename.replace(".mkv", f"_contour")
            else:
                contour_filename = movie_filename.replace(".movie", f"_contour")

        if os.path.isfile(contour_filename + ".npz") and prevent_overwrite:
            if type(prevent_overwrite) == bool:
                self.logger.info(f"Contour exists for {movie_filename}, skipping")
                return None, None
            mtime = os.path.getmtime(contour_filename + ".npz")
            timestring = datetime.datetime.fromtimestamp(mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if mtime > prevent_overwrite:
                self.logger.info(
                    f"Contour exists from {timestring} for {movie_filename}, skipping"
                )
                return None, None
            else:
                self.logger.info(
                    f"Overwriting contour from {timestring} for {movie_filename}"
                )

        # we might be limited by the movie reading
        movie = get_movie_reader(movie_filename, preload_to_memory=False)
        # if hasattr(movie, "size_x"):
        # if movie.size_x > 512 or movie.size_x < 80:
        #    movie.size_x = 256  # TODO: workaround for bug in .movie files
        cio = self.process_sequence(
            movie.frames(), seq_len=movie.n_frames
        )  # TODO: what will happen with large files?
        if self.debug:
            self.debug_data["masks"] = np.array(self.debug_data["masks"])
            self.debug_data["mask"] = np.array(self._mask)
            self.debug_data["validation_scores"] = np.array(
                self.debug_data["validation_scores"]
            )
            cio.add_extras(self.debug_data)

        if contour_filename != "" and contour_filename is not False:
            cio.write(contour_filename)

        if visualise:
            flickering.utils.visualisation.visualise_contours(
                cio.contours, cio.valid_indices, contour_filename + "-preview.png"
            )

        self.logger.info(f"Processed {movie_filename}")
        if return_cio:
            return cio
        return cio.contours, cio.valid_indices

    @staticmethod
    def gauss_mask(width, std, rank=0):
        x = np.linspace(-width / 2, width / 2, width)
        if rank == 0:
            return np.exp(-x * x / (2 * std * std))
        if rank == 1:
            return -x / std * np.exp(-x * x / (2 * std * std))  # this is inverted later
        if rank == 2:
            return (x**2 / std**4 - 1 / std**2) * np.exp(-(x**2) / (2 * std * std))
        raise NotImplementedError("Rank >2 not implemented")

    @staticmethod
    def normalise_image_values(image, use_percentile_tolerance=False):
        if use_percentile_tolerance:
            max_val = image.max()
            min_val = image.min()
            max_val = np.percentile(
                image, 100 - use_percentile_tolerance, interpolation="nearest"
            )  # more noise tolerant
            image[image > max_val] = max_val
            min_val = np.percentile(
                image, use_percentile_tolerance, interpolation="nearest"
            )
            image[image < min_val] = min_val

            if max_val == min_val:
                diff = 1
            else:
                diff = max_val - min_val

            return (image - min_val) / diff

        if image.dtype.byteorder in (">", "<") and image.dtype.byteorder != "=":
            image = image.astype(image.dtype.name)

        return _normalise_image_values_parallel(image)

    @staticmethod
    @jit(nopython=True)
    def convert_contour_xy(contour, center=np.array([0.0, 0.0])):
        """convert radial contour to xy points
        #numba has a massive effect here!

        Args:
            contour (list): _description_
            center (list): _description

        Returns:
            list: a nested sequence of the cartesian pixels of the contour relative to image frame
        """
        thetas = np.linspace(0, 2 * np.pi, contour.shape[0])
        dirs = np.vstack((np.cos(thetas), np.sin(thetas)))
        coordinates = center + np.swapaxes(contour * dirs, 0, 1)

        return coordinates

    def find_cell_center_yusuf(self, image: np.ndarray, refine=False, radius=None):
        """
        Finds the cell center using a combination of Hough Transform and a 4-point gradient check.
        It includes a robust check to handle cases where no circles are detected and can refine the center.
        """
        # center_guess = np.array(image.shape) / 2.0 # (row, col)
        if image.max() > 1.0:
            image = CorrelationContourTracker.normalise_image_values(image)

        denoised = cv2.fastNlMeansDenoising(
            (image * 255.0).astype(np.uint8), None, 40, 17, 13
        )
        gray_uint8 = (denoised * 255).astype(np.uint8)
        image_not = cv2.bitwise_not(gray_uint8)
        blurred = cv2.medianBlur(image_not, 7)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=3,
        )
        Laplacian_image = cv2.Laplacian(adaptive_thresh, cv2.CV_64F, ksize=5)

        abs_laplacian = np.abs(Laplacian_image)
        hough_input_image = np.uint8(abs_laplacian)

        all_circles = cv2.HoughCircles(
            hough_input_image,
            cv2.HOUGH_GRADIENT,
            self.hough_dp,
            self.hough_min_d,
            param1=self.hough_param_1,
            param2=self.hough_param_2,
            minRadius=self.hough_min_r,
            maxRadius=self.hough_max_r,
        )

        image_center = np.array(image.shape) / 2.0
        center_guess = image_center
        best_radius = self.hough_max_r - 10

        if all_circles is None:
            self.logger.warning(
                "Hough Transform failed to find any circles on this frame."
            )
        else:
            # If circles were found, select the best one (closest to the image center)
            min_dist = 1e9
            for c in all_circles[0, :]:
                radius_px = int(c[2])
                center = [int(c[1]), int(c[0])]  # Stored as (y, x)
                if (
                    radius_px < self.hough_max_r
                    and np.linalg.norm(image_center - np.array(center)) < min_dist
                ):
                    min_dist = np.linalg.norm(image_center - center)
                    center_guess = center
                    best_radius = radius_px

        img_h, img_w = image.shape
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        search_radius = best_radius + 15
        ignore_center_rad = best_radius - 8
        center_int = np.round(center_guess).astype(int)
        cv2.circle(mask, (center_int[1], center_int[0]), search_radius, 255, -1)
        masked_image = cv2.bitwise_and(Laplacian_image, Laplacian_image, mask=mask)

        if self.last_known_radius is not None:
            search_dist = np.round(self.last_known_radius + 5).astype(int)
        else:
            search_dist = np.round(self.hough_max_r + 10).astype(int)

        # The Laplacian has two rings and we can get the most accurate tracking by averaging the positions of two of them.
        # UP
        start_u = center_int[0] + ignore_center_rad
        end_u = min(img_h, center_int[0] + search_dist)
        slice_u = masked_image[start_u:end_u, center_int[1]]
        org_slice_u = np.argsort(slice_u)
        if slice_u.size < 2:
            return None, None
        radius_u_1 = ignore_center_rad + np.argmin((slice_u))
        radius_u_2 = ignore_center_rad + org_slice_u[1]
        radius_u = (radius_u_1 + radius_u_2) / 2
        position_u = [center_int[0] + radius_u, center_int[1]]

        # DOWN
        start_d = max(0, center_int[0] - search_dist)
        end_d = center_int[0] - ignore_center_rad
        slice_d = masked_image[start_d:end_d, center_int[1]]
        org_slice_d = np.argsort(np.flip(slice_d))
        if slice_d.size < 2:
            return None, None
        radius_d_1 = (ignore_center_rad) + np.argmin(np.flip(slice_d))
        radius_d_2 = (ignore_center_rad) + org_slice_d[1]
        radius_d = (radius_d_1 + radius_d_2) / 2
        position_d = [center_int[0] - radius_d, center_int[1]]

        # LEFT
        start_l = max(0, center_int[1] - search_dist)
        end_l = center_int[1] - ignore_center_rad
        slice_l = masked_image[center_int[0], start_l:end_l]
        org_slice_l = np.argsort(np.flip(slice_l))
        if slice_l.size < 2:
            return None, None
        radius_l_1 = ignore_center_rad + np.argmin(np.flip(slice_l))
        radius_l_2 = ignore_center_rad + org_slice_l[1]
        radius_l = (radius_l_1 + radius_l_2) / 2
        position_l = [center_int[0], center_int[1] - radius_l]

        # RIGHT
        start_r = center_int[1] + ignore_center_rad
        end_r = min(img_w, center_int[1] + search_dist)
        slice_r = masked_image[center_int[0], start_r:end_r]
        org_slice_r = np.argsort(slice_r)
        if slice_r.size < 2:
            return None, None
        radius_r_1 = ignore_center_rad + np.argmin(slice_r)
        radius_r_2 = ignore_center_rad + org_slice_r[1]
        radius_r = (radius_r_1 + radius_r_2) / 2
        position_r = [center_int[0], center_int[1] + radius_r]

        # Calculate the initial center and radius from these 4 points
        center_y = center_int[0] + (radius_u - radius_d) / 2.0
        center_x = center_int[1] + (radius_r - radius_l) / 2.0
        center_y = np.round(center_y).astype(int)
        center_x = np.round(center_x).astype(int)

        new_center = [center_y, center_x]
        new_radius_r = np.linalg.norm(position_r - np.array(new_center))
        new_radius_l = np.linalg.norm(position_l - np.array(new_center))
        new_radius_d = np.linalg.norm(position_d - np.array(new_center))
        new_radius_u = np.linalg.norm(position_u - np.array(new_center))
        new_radius = np.median([new_radius_u, new_radius_d, new_radius_l, new_radius_r])

        # Safeguard to revert to last known position if the new guess drifts too far
        if self.last_known_center is not None:
            center_drift = np.linalg.norm(np.array(new_center) - self.last_known_center)
            radius_drift = np.abs(new_radius - self.last_known_radius)
            if (
                center_drift > self.max_center_change_th
                or radius_drift > self.max_radius_change_th
            ):
                self.logger.debug(
                    f"Frame: 4-point check drifted too far. Reverting to last known position."
                )
                best_center = self.last_known_center
                best_radius = self.last_known_radius
            else:
                best_center = new_center
                best_radius = new_radius
        else:
            best_center = new_center
            best_radius = new_radius

        self.logger.debug(f"Selected center = {best_center}, radius_px = {best_radius}")

        if refine:
            self.logger.debug("Refining center")
            refined_center, contour = self.get_contour(
                image, best_center, radius=best_radius
            )
            if contour is not None and len(contour) > 0 and not np.isnan(contour).all():
                return refined_center, np.nanmean(contour)
            else:
                self.logger.warning("Refining center failed")
                return best_center, best_radius

        return best_center, best_radius
