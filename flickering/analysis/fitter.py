import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, ifftshift
from scipy.optimize import curve_fit
from scipy.special import sph_harm, loggamma, lpmv, erf, i0e
from scipy.constants import k
import scipy.integrate as integrate
from scipy.stats import linregress
import scipy
import unicodedata
import pandas as pd
from typing import List, Tuple, Union, Dict
from pathlib import Path

# from python_contour_trackers.contour_tracker import (  # legacy, removed
#     load_contour as load_contour_python,
# )
from flickering.utils import constants
import logging

from numba import jit

# import numba_scipy
from cachetools import cached, FIFOCache
from copy import deepcopy
from dict_hash import sha256
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, sosfiltfilt


EPSILON = 1e-24


class ContourFitter:
    VERSION = 4

    def __init__(self):
        """initialise default state as in previous dictionary. Ones that default to None
        should be initialised each time for consistency

        All distances chosen to be um, all times in ms"""
        self.min_mode = 5  # int: minimum mode to consider in fitting the ps (inclusive)
        self.max_mode = (
            20  # int or "auto": maximum mode to consider in fitting the ps (inclusive)
        )

        self.auto_max_mode_range = (17, 26)  # (min, max) range for auto detection
        self.auto_max_mode_threshold = 0.05  # 5% difference -> stop fit when this is exceeded for 5 consecutive points, starting with a shorter fit

        self.mode_sum_range = 30  # int: number of modes to sum over in the theoretical power spectrum used as the fitting function

        # Fallback initial values for when rough_fit fails or produces invalid results
        self.fallback_sigma = 5.0  # Default sigma in units of 1e-7 N/m
        self.fallback_kappa = 200.0  # Default kappa in units of 1e-21 N·m
        self.rough_fit_r2_threshold = (
            0.8  # Minimum R² for rough fit to be considered valid
        )
        self.sigma_limits = (0.1, 100.0)  # Valid range for sigma in units of 1e-7 N/m
        self.kappa_limits = (
            10.0,
            2000.0,
        )  # Valid range for kappa in units of 1e-21 N·m

        self.sub_radius = None  # str: TODO does different stuff if == "static_shape"
        self.sub_radius_rolling_mean_window = 5000  # number of frames to average to obtain static shape in rolling_mean mode
        self.sub_radius_filter = None
        self.nm_per_px = 65.07  # float
        self.delay_between_frames_ms = 1  # float
        self.exposure_time_ms = None  # float: TODO check it is implemented correctly in acquisition time factor expression
        self.exposure_correction_method = "rautu"  # or original
        self.exposure_correction_location = "fit"  # "spectrum" or "fit"
        self.depth_of_focus_um = None  # float
        self.mean_spontaneous_curvature = 0  # float, defined as H_0 in Rautu et al.
        self.decay_time_fit_max = 50  # int: maximum mode over which to fit decay times
        self.vertical_radius_um = None  # float
        self.kB_T = 4.114  # float: 298K = 4.114e-21 J, 310K = 4.280e-21 J. TODO where are factors put back in?
        self.contour_file = None  # str
        self.movie_file = None  # str
        self.radius_uncertainty = 50  # float: in nm
        # self.extra_info = None
        self.thinning_factor = None  # int: to specify only considering every <thinning_factor>th point in a given contour
        self.remove_outliers = True  # bool: whether to remove spiky contours
        self.maximum_outlier_fluctuation = 0.03
        # float: units of radius, only if above true
        self.angular_noise = 0  # float: change to None to use estimated value
        self.angular_noise_estimate = None  # float: estimated in fit if above not given
        self.radial_noise = 0  # float: change to None to use estimated value
        self.radial_noise_estimate = None  # float: estimated in fit if above not given

        self.fitting_method = "rautu"  # str: original, confinement, rautu
        self.fit_weights_method = "mps_error"  # str: mps_error, constant

        self.noise_floor_above = (
            40  # int: modes with n>this are considered the noise floor
        )
        self.noise_floor_below = 50  # int: modes with n<this and > noise_floor_above are considered in noise estimation

        self.autocorrelation_invalid_method = "average"  # TODO: other possible values?
        # how to handle invalid points for time series. average is the only implemented one
        self.radius_correction_nm = (
            0  # -216 #obtained from comparison with fluorscent labeled membrane
        )
        self.decay_time_use_fit = False  # use theoretical fit decay times instead of direct values, if int fit will be used for n>value
        self.autocorrelation_mode = "separate"  # separate|abs:mode for fitting the complex spectrum. separate imag and real, abs
        self.decay_time_brown_correction = False  # apply the projection correction from https://journals.aps.org/pre/pdf/10.1103/PhysRevE.84.021930
        self.decay_time_fit_min = (
            0  # autocorrelation offset from which exponential decay should be fitted
        )
        self.decay_times_model = "exponential"  # "exponential" or "linear_exponential"
        self.decay_time_brown_correction_not_in_filter = True

        # --- Viscosity fit options (fit internal viscosity) ---
        self.fit_viscosity = False  # Enable fitting for internal viscosity
        self.viscosity_eta_out = (
            0.733e-3  # Pa.s, default value for external viscosity (RPMI, 37C)
        )
        self.fit_viscosity_modes = (
            None  # (start, end) mode numbers to use for the fit (overrides tau range)
        )
        self.fit_viscosity_tau_range = (
            None  # (min_tau, max_tau) in ms, alternative to mode range
        )
        self.fit_viscosity_lock_sigma_kappa = True  # If True, sigma and kappa are fixed during fit; if False, they are also fitted
        self.fit_viscosity_method = "rautu"
        # -----------------------------------------------------

        # TODO: how does excluded handle self?
        self.theory_ps = np.vectorize(
            ContourFitter.theory_ps_single, excluded={1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        )

        self.logger = logging.getLogger("FITTER")
        self.fit_limits = None  # [[tension_min, tension_max], [bending_min, bending_max]] only applies to rautu fit
        self.fit_mode = "linear"  # or log TODO report correct uncertainity if log mode

        self.extra_mps_error = 0  # nm^2, 0.3 is reasonable. This adds additional uncertaintiy to the MPS, can be estimated from simulated data

        self.rolling_window_correction = False
        self.rolling_window_correction_location = "spectrum"  # "spectrum" or "fit"
        self.rolling_window_correction_timescaling = 0.248  # T=a*window_size
        self.rolling_window_correction_exponent = 1.273

        self.decay_time_filter_threshold_s = None  # 1 set to 'static_shape' to subtract mean shape. #high pass filter is applied to contour spectra before correltion calcultion

        self.radial_variance_calculation = False
        self.save_acf_frames = 100

        self.propagate_tau_error = True
        self.verify_config()

    def verify_config(self):
        """
        Verify that the current configuration is valid and all methods are implemented.
        Raises ValueError or NotImplementedError if config is invalid.
        """
        self.logger.debug("Verifying configuration...")

        # 1. Method Selection Validation
        valid_fitting_methods = {
            "original",
            "confinement",
            "confinement_new",
            "rautu",
            "none",
            None,
        }
        if self.fitting_method not in valid_fitting_methods:
            raise NotImplementedError(f"Unknown fitting method: {self.fitting_method}")

        if self.fitting_method in ("confinement", "confinement_new"):
            self.logger.warning(
                f"Fitting method '{self.fitting_method}' is untested and may produce unreliable results."
            )

        valid_fit_weights_methods = {"mps_error", "none", "poisson"}
        if self.fit_weights_method not in valid_fit_weights_methods:
            raise NotImplementedError(
                f"Unknown fit weights method: {self.fit_weights_method}"
            )

        valid_fit_modes = {"linear", "log"}
        if self.fit_mode not in valid_fit_modes:
            raise NotImplementedError(f"Invalid fit_mode: {self.fit_mode}")

        if self.exposure_correction_method is not None:
            valid_exposure_methods = {"rautu", "rautu_const", "original"}
            if self.exposure_correction_method not in valid_exposure_methods:
                raise NotImplementedError(
                    f"Unknown exposure correction method: {self.exposure_correction_method}"
                )

        valid_exposure_locations = {"spectrum", "fit"}
        if self.exposure_correction_location not in valid_exposure_locations:
            raise ValueError(
                f"Invalid exposure_correction_location: {self.exposure_correction_location}"
            )

        if self.sub_radius is not None:
            valid_sub_radius = {"static_shape", "rolling_mean", "butter"}
            if self.sub_radius not in valid_sub_radius:
                raise NotImplementedError(
                    f"Unknown sub_radius method: {self.sub_radius}"
                )

        valid_decay_models = {"exponential", "linear_exponential"}
        if self.decay_times_model not in valid_decay_models:
            raise ValueError(f"Unknown decay_times_model: {self.decay_times_model}")

        valid_autocorr_modes = {"separate", "abs"}
        if self.autocorrelation_mode not in valid_autocorr_modes:
            raise ValueError(
                f"Unknown autocorrelation_mode: {self.autocorrelation_mode}"
            )

        # 2. Logical Dependencies & Implementation Gaps
        if (
            self.fitting_method == "original"
            and self.exposure_correction_location == "fit"
        ):
            if self.exposure_correction_method is not None:
                raise NotImplementedError(
                    "Exposure correction with location='fit' is not implemented for fitting_method='original'."
                )

        if (
            self.exposure_correction_method == "rautu"
            and self.exposure_correction_location == "spectrum"
        ):
            raise NotImplementedError(
                "exposure_correction_method='rautu' is only implemented for location='fit'."
            )

        if self.fit_viscosity:
            if self.fit_viscosity_modes is None:
                raise ValueError(
                    "fit_viscosity_modes must be specified when fit_viscosity is True."
                )
            valid_visc_methods = {"rautu", "yoon"}
            if self.fit_viscosity_method not in valid_visc_methods:
                raise NotImplementedError(
                    f"Unknown fit_viscosity_method: {self.fit_viscosity_method}"
                )

        if (
            self.exposure_correction_method is not None
            and self.exposure_time_ms is not None
        ):
            if self.exposure_time_ms < 0:
                raise ValueError("exposure_time_ms must be positive for correction.")

        if self.rolling_window_correction:
            if self.sub_radius not in {"rolling_mean", "butter"}:
                raise ValueError(
                    "rolling_window_correction requires sub_radius='rolling_mean' or 'butter'."
                )
            if (
                self.sub_radius_rolling_mean_window is None
                or self.sub_radius_rolling_mean_window <= 0
            ):
                raise ValueError(
                    "sub_radius_rolling_mean_window must be positive when rolling_window_correction is active."
                )

        # 3. Numeric Constraints
        if isinstance(self.max_mode, int):
            if self.min_mode >= self.max_mode:
                raise ValueError("min_mode must be strictly less than max_mode.")

        if (
            not isinstance(self.auto_max_mode_range, tuple)
            or len(self.auto_max_mode_range) != 2
        ):
            raise ValueError("auto_max_mode_range must be a tuple of (min, max).")
        if self.auto_max_mode_range[0] >= self.auto_max_mode_range[1]:
            raise ValueError("auto_max_mode_range[0] must be less than range[1].")

        if self.noise_floor_above >= self.noise_floor_below:
            raise ValueError("noise_floor_above must be less than noise_floor_below.")

        if self.nm_per_px <= 0:
            raise ValueError("nm_per_px must be positive.")
        if self.delay_between_frames_ms <= 0:
            raise ValueError("delay_between_frames_ms must be positive.")
        if self.kB_T <= 0:
            raise ValueError("kB_T must be positive.")

        if self.sigma_limits[0] >= self.sigma_limits[1]:
            raise ValueError("sigma_limits[0] must be less than sigma_limits[1].")
        if self.kappa_limits[0] >= self.kappa_limits[1]:
            raise ValueError("kappa_limits[0] must be less than kappa_limits[1].")

        self.logger.debug("Configuration verified successfully.")

    def fit_decay_times(self, taus, fit_results, radius_nm):
        """
        Unified method to fit decay times.
        - Fits theoretical_tau to experimental decay times.
        - Can fit for eta_in, and optionally for sigma and kappa.
        - Can return fit parameters or a full array of decay times based on the fit.
        - Uses configuration from fit_viscosity_* parameters.
        """
        self.verify_config()
        # Determine fit range from configuration
        fit_modes = None
        if self.fit_viscosity_modes is None:
            raise NotImplementedError(
                "fit_viscosity_modes is None, cannot fit decay times."
            )

        fit_start, fit_end = self.fit_viscosity_modes
        fit_modes = np.arange(fit_start, fit_end)

        if self.fit_viscosity_tau_range is not None:
            min_tau, max_tau = self.fit_viscosity_tau_range
            taus_mean = np.mean(taus[:, [0, 2]], axis=1)
            fit_modes = np.where((taus_mean >= min_tau) & (taus_mean <= max_tau))[0]
        # else:
        #    fit_modes = np.arange(self.decay_time_fit_start, self.decay_time_fit_end)

        # Exclude modes > 180
        fit_modes = fit_modes[fit_modes <= 180]
        fit_modes = fit_modes[fit_modes >= fit_start]
        fit_modes = fit_modes[fit_modes <= fit_end]

        # Prepare data for fitting
        valid_indices = np.isfinite(taus[fit_modes, :4]).all(axis=1)
        if not np.any(valid_indices):
            self.logger.warning("No valid decay times to fit in fit_decay_times.")
            return {
                "fit_success": False,
                "error": "No valid decay times in fit range.",
            }, taus

        fit_modes = fit_modes[valid_indices]
        tau_fit_values = np.mean(taus[fit_modes][:, [0, 2]], axis=1)  # ms
        tau_fit_errors = np.mean(taus[fit_modes][:, [1, 3]], axis=1)  # ms
        tau_fit_values_s = tau_fit_values / 1000.0
        tau_fit_errors_s = tau_fit_errors / 1000.0
        tau_fit_errors_s[tau_fit_errors_s < 0] = np.inf
        # Define model and parameters based on configuration
        eta_out = self.viscosity_eta_out

        if self.fit_viscosity_lock_sigma_kappa:
            sigma = fit_results.get("sigma", {}).get("value")
            kappa = fit_results.get("kappa", {}).get("value")
            if sigma is None or kappa is None:
                raise ValueError(
                    "Sigma and kappa must be available from power spectrum fit when locked."
                )

            sigma_SI = sigma * 1e-7
            kappa_SI = kappa * 1e-21

            def tau_model(n, eta_in):
                if self.fit_viscosity_method == "rautu":
                    return theoretical_tau(
                        n, eta_in, eta_out, radius_nm * 1e-9, kappa_SI, sigma_SI
                    )
                elif self.fit_viscosity_method == "yoon":
                    return theoretical_tau_yoon(
                        n, eta_in, eta_out, radius_nm * 1e-9, kappa_SI, sigma_SI
                    )
                else:
                    raise NotImplementedError(
                        f"{self.fit_viscosity_method} not implemented"
                    )

            p0 = [2e-2]  # initial guess for eta_in
            bounds = ([1e-4], [10])
        else:

            def tau_model(n, eta_in, sigma, kappa):
                if self.fit_viscosity_method == "rautu":
                    return theoretical_tau(
                        n,
                        eta_in,
                        eta_out,
                        radius_nm * 1e-9,
                        kappa * 1e-21,
                        sigma * 1e-7,
                    )
                elif self.fit_viscosity_method == "yoon":
                    return theoretical_tau_yoon(
                        n,
                        eta_in,
                        eta_out,
                        radius_nm * 1e-9,
                        kappa * 1e-21,
                        sigma * 1e-7,
                    )
                else:
                    raise NotImplementedError(
                        f"{self.fit_viscosity_method} not implemented"
                    )

            sigma0 = fit_results.get("sigma", {}).get("value", 5.0)
            kappa0 = fit_results.get("kappa", {}).get("value", 200.0)
            p0 = [2e-2, sigma0, kappa0]
            bounds = ([1e-3, sigma0 * 0.1, kappa0 * 0.1], [1, sigma0 * 10, kappa0 * 10])

        # Perform the fit
        try:
            popt, pcov = curve_fit(
                tau_model,
                fit_modes,
                tau_fit_values_s,
                sigma=tau_fit_errors_s,
                absolute_sigma=True,
                p0=p0,
                bounds=bounds,
                maxfev=10000,
                xtol=1e-4,
            )
        except Exception as e:
            self.logger.error(f"Decay time fit failed: {e}")
            return {"fit_success": False, "error": str(e)}, taus

        # Calculate R^2
        predicted_taus = tau_model(fit_modes, *popt)
        residuals = tau_fit_values_s - predicted_taus
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((tau_fit_values_s - np.mean(tau_fit_values_s)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Process results
        fit_params = {
            "fit_success": True,
            "fit_modes": fit_modes.tolist(),
            "r2": r2,
            "eta_in": {
                "value": popt[0],
                "error": float(np.sqrt(pcov[0, 0])) if pcov.shape[0] > 0 else None,
            },
            "eta_out": {"value": eta_out},
            "dts": tau_fit_values_s,
            "taus": taus,
        }

        if self.fit_viscosity_lock_sigma_kappa:
            fit_params["sigma"] = {"value": sigma, "unit": 1e-7}
            fit_params["kappa"] = {"value": kappa, "unit": 1e-21}

            # For updating taus
            fit_sigma_SI = sigma_SI
            fit_kappa_SI = kappa_SI
        else:
            fit_params["sigma"] = {
                "value": popt[1],
                "error": float(np.sqrt(pcov[1, 1])) if pcov.shape[1] > 1 else None,
                "unit": 1e-7,
            }
            fit_params["kappa"] = {
                "value": popt[2],
                "error": float(np.sqrt(pcov[2, 2])) if pcov.shape[1] > 2 else None,
                "unit": 1e-21,
            }

            # For updating taus
            fit_sigma_SI = popt[1] * 1e-7
            fit_kappa_SI = popt[2] * 1e-21

        # Update taus array if requested
        if self.decay_time_use_fit:
            all_modes = np.arange(len(taus))
            if self.fit_viscosity_method == "rautu":
                fit_taus_s = theoretical_tau(
                    all_modes,
                    popt[0],
                    eta_out,
                    radius_nm * 1e-9,
                    fit_kappa_SI,
                    fit_sigma_SI,
                )
            elif self.fit_viscosity_method == "yoon":
                fit_taus_s = theoretical_tau_yoon(
                    all_modes,
                    popt[0],
                    eta_out,
                    radius_nm * 1e-9,
                    fit_kappa_SI,
                    fit_sigma_SI,
                )

            # Manual error propagation
            if self.fit_viscosity_method == "rautu":
                partials_func = _calculate_partials_theoretical_tau_analytical
            elif self.fit_viscosity_method == "yoon":
                partials_func = _calculate_partials_theoretical_tau_yoon
            else:
                raise NotImplementedError(
                    f"{self.fit_viscosity_method} not implemented"
                )

            if self.fit_viscosity_lock_sigma_kappa:
                # Only eta_in has uncertainty
                d_tau_d_eta_in, _, _ = partials_func(
                    all_modes,
                    popt[0],
                    eta_out,
                    radius_nm * 1e-9,
                    fit_kappa_SI,
                    fit_sigma_SI,
                )
                var_eta_in = pcov[0, 0]
                var_tau = (d_tau_d_eta_in**2) * var_eta_in
            else:
                # eta_in, sigma, and kappa have uncertainties and covariances
                eta_in_fit, sigma_fit, kappa_fit = popt
                d_tau_d_eta_in, d_tau_d_kappa, d_tau_d_sigma = partials_func(
                    all_modes,
                    eta_in_fit,
                    eta_out,
                    radius_nm * 1e-9,
                    kappa_fit * 1e-21,
                    sigma_fit * 1e-7,
                )

                var_eta_in, var_sigma, var_kappa = np.diag(pcov)
                cov_eta_sigma = pcov[0, 1]
                cov_eta_kappa = pcov[0, 2]
                cov_sigma_kappa = pcov[1, 2]

                var_tau = (
                    (d_tau_d_eta_in**2 * var_eta_in)
                    + (d_tau_d_sigma**2 * var_sigma)
                    + (d_tau_d_kappa**2 * var_kappa)
                    + 2 * (d_tau_d_eta_in * d_tau_d_sigma * cov_eta_sigma)
                    + 2 * (d_tau_d_eta_in * d_tau_d_kappa * cov_eta_kappa)
                    + 2 * (d_tau_d_sigma * d_tau_d_kappa * cov_sigma_kappa)
                )

            fit_taus_err_s = np.sqrt(var_tau)
            fit_taus_ms = fit_taus_s * 1000.0
            fit_taus_err_ms = fit_taus_err_s * 1000.0

            use_fit_from = (
                self.decay_time_use_fit
                if isinstance(self.decay_time_use_fit, (int, float))
                else 0
            )

            taus[use_fit_from:, 0] = fit_taus_ms[use_fit_from:]
            taus[use_fit_from:, 2] = fit_taus_ms[use_fit_from:]
            taus[use_fit_from:, [1, 3]] = np.stack(
                [fit_taus_err_ms[use_fit_from:], fit_taus_err_ms[use_fit_from:]], axis=1
            )

        return fit_params, taus

    def get_config_dict(self):
        config = deepcopy(self.__dict__)
        del config["logger"]
        del config["theory_ps"]
        del config["movie_file"]
        del config["contour_file"]

        config["fit_limits"] = np.array(self.fit_limits).tolist()
        config["version"] = self.VERSION

        return config

    def get_config_hash(self):
        config_dict = self.get_config_dict()

        return sha256(config_dict)

    @staticmethod
    @cached(FIFOCache(1024 * 512))  # in practice this should only have a few values
    def structure_factor(Delta: float, n_mode: int, q_mode: Union[float, int]) -> float:
        """Defined as in (S.2.36) Rautu et al., but without the combinatorial factor in the first integrand term, which is better passed to the final function.
        prefactor has no limit as delta approaches 0. However, it does seem to grow slowly (log divergence?)

        NOTE: renamed from "L" on 22.02.23. Not yet changed variable names as this needs deeper reading through

        Args:
            Delta (float): focal depth of microscope over which fluctuations are imaged [m]
            n_mode (int): mode number corresponding to n in the equation
            q_mode (Union[float, int]): mode number corresponding to q in the equation

        Returns:
            float: the entire structure factor L_n,q without the combinatorial factor defined in S.2.33
        """

        if n_mode < q_mode:
            return 0.0

        # prefactor and mu_0 use i0e, a helpful function for the product of exponential and bessel. The old versions suffer from overflow at delta < 0.0188

        mu_0 = (
            Delta
            * np.sqrt(np.pi / 2.0)
            * erf(1.0 / (Delta * np.sqrt(2)))
            / ((np.pi / 2.0) * i0e((1.0 / (4.0 * (Delta**2)))))
        )  # ok

        # this can be sped up, immediate zero if n_mode+q_mode is odd?
        prefactor = (1.0 + (-1) ** (n_mode + q_mode)) / (
            np.pi * i0e((4.0 * (Delta**2.0)) ** -1)
        )  # ok

        # @jit(nopython=True) makes this a lot slower?
        def integrand(omega, d, n, q):
            integrand_one = lpmv(
                float(q), float(n), float(omega)  # float for numba
            )  # is this actually correct? seems to result in very high values
            integrand_two = ((2 * (d**2) + omega**2) / (d**2)) - (
                mu_0 * (d**2 + omega**2) / (np.sqrt(1.0 - omega**2) * (d**2))
            )
            integrand_three = np.exp(-((omega**2) / (2.0 * (d**2))))

            return integrand_one * integrand_two * integrand_three

        integral, integral_err = integrate.quad(
            lambda x: integrand(x, Delta, n_mode, q_mode), a=0, b=1
        )

        return prefactor * integral

    @staticmethod
    def theory_ps_single(
        q: Union[float, int],
        alpha: np.ndarray,
        beta: np.ndarray,
        taus: Union[np.ndarray, None],
        delta: float = 0,
        q_sum: int = 30,
        rw_correction_factors: Union[np.ndarray, None] = None,
        exposure_correction_method: str = "rautu",
        exposure_correction_location: str = "fit",
        max_mode: Union[int, None] = None,
    ) -> float:
        """This is the fitting function that takes into account the depth of focus and acquisition time.
        TODO: check acquisition time correction [eqn(22) of Pécréaux vs S.2.38 of Rautu]

        exposure_correction_location != 'fit' should now call with taus = None

        Args:
            q (Union[float, int]): the mode number (x-coordinate) of the power spectrum
            alpha (np.ndarray): _description_
            beta (np.ndarray): _description_
            taus (np.ndarray): Inverted decay times scaled by aquisition time
            delta (float, optional): _description_. Defaults to 0.
            q_sum (int, optional): number of mdoes to sum over in the theoretical power spectrum used as the fitting function. Defaults to 30.

        Returns:
            float: _description_
        """
        if taus is None:
            acquisition_time_factor = np.ones(int(q_sum + q + 2))
        else:
            # TODO this preprocessing should be elsewhere
            # harmonic mean?
            # g = 1/np.mean(taus[:, ::2], axis=1)
            g = taus  # averaging now done outside, befier 1/tau step
            if exposure_correction_method == "original":
                # Original correction (Faucon et al. 1989)
                # C(t) = 2*t^2 * (1/t + exp(-1/t) - 1) where t = tau / T_exp
                # Here g is passed as (tau / T_exp)^-1 = T_exp / tau = 1/t
                # So t = 1/g
                # C(g) = 2*(1/g)^2 * (g + exp(-g) - 1)
                #      = 2/g^2 * (g + exp(-g) - 1)

                # Avoid division by zero if g is 0 (infinite tau) -> correction is 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    acquisition_time_factor = 2 * (g + np.exp(-g) - 1) / (g**2)
                    acquisition_time_factor[g == 0] = 1.0
                    acquisition_time_factor[~np.isfinite(acquisition_time_factor)] = 1.0
            else:
                # Rautu correction
                acquisition_time_factor = (
                    (1 - np.exp(-g)) / g
                ) ** 2  # This differs from pecroux  # Rautu et al S.2.38
            # TODO: compare to Pecreaux

        if rw_correction_factors is None:
            rw_correction_factors = np.ones(int(q_sum + q + 2))

        # The log of the combinatorial factor in S.2.33. NOTE the 0.5 was added on 21.03.23 as don't think the sqrt was catered for
        # 2 * added as this is log of the prefactor in structure factor^2
        def ln_comb(n, m) -> float:
            return (
                # 2
                # * 0.5
                # (
                loggamma(n - m + 1)
                - loggamma(n + m + 1)
                + np.log(2 * n + 1)
                - np.log(4 * np.pi)
                # )
            )

        if (
            exposure_correction_method == "rautu_const"
            or exposure_correction_method == "original"
        ) and exposure_correction_location == "fit":
            true_atfs = acquisition_time_factor
            acquisition_time_factor = np.ones_like(acquisition_time_factor)

        if delta == 0:
            sum_val = 0
            modes_to_sum = range(int(q), int(q + q_sum + 1))
            for i, n in enumerate(modes_to_sum):
                atf = (
                    acquisition_time_factor[n]
                    if n < len(acquisition_time_factor)
                    and exposure_correction_method == "rautu"
                    and exposure_correction_location == "fit"
                    else 1
                )
                rw_cf = (
                    rw_correction_factors[n] if n < len(rw_correction_factors) else 1
                )
                term = (
                    atf
                    * 1.0
                    / rw_cf
                    * (np.real(sph_harm(q, n, 0, np.pi / 2.0)) ** 2)
                    / (
                        (float(n) - 1)
                        * (float(n) + 2)
                        * ((beta**2) + float(n) * (float(n) + 1))
                    )
                )
                sum_val += term
            r = 4 * np.pi * alpha * (beta**3) * sum_val

        else:
            sum_val = 0
            modes_to_sum = range(int(q), int(q + q_sum + 1))
            for i, n in enumerate(modes_to_sum):
                atf = (
                    acquisition_time_factor[n]
                    if n < len(acquisition_time_factor)
                    else 1
                )
                rw_cf = (
                    rw_correction_factors[n] if n < len(rw_correction_factors) else 1
                )
                term = (
                    atf
                    * 1.0
                    / rw_cf
                    * np.exp(ln_comb(n, q))
                    * (ContourFitter.structure_factor(delta, n, q) ** 2)
                    / (
                        (float(n) - 1)
                        * (float(n) + 2)
                        * ((beta**2) + float(n) * (float(n) + 1))
                    )
                )
                sum_val += term
            r = 4 * np.pi * alpha * (beta**3) * sum_val

        if (
            exposure_correction_method == "rautu_const"
            or exposure_correction_method == "original"
        ) and exposure_correction_location == "fit":
            return r * true_atfs[int(q)]  # apply fixed correction at mode q

        return r

    @staticmethod
    def contour_length(contours: np.ndarray) -> float:
        """TODO: where used?

        Args:
            contours (np.ndarray): _description_

        Returns:
            float: _description_
        """
        return np.sum(np.linalg.norm(np.diff(contours[0], axis=0), axis=1))

    def autocorrelation_functions(
        self, R_fft: np.ndarray, indices: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Calculates autocorrelation functions for each mode.
        Optimized to reduce memory for long videos by limiting the number of lags calculated and stored.
        """

        if (
            self.decay_time_filter_threshold_s is None
            and self.decay_times_model == "exponential"
        ):
            self.logger.warning(
                "Using autocorrelation without static shape removal is not recommended and can produce invalid results"
            )

        # We only need enough lags for fitting, plotting and saving
        plot_xlim = 50.0  # ms
        frames_needed_for_plot = int(np.ceil(plot_xlim / self.delay_between_frames_ms))
        max_lags_needed = max(
            self.save_acf_frames, self.decay_time_fit_max + 1, frames_needed_for_plot
        )
        n_frames = R_fft.shape[0]
        max_lags = min(max_lags_needed, n_frames // 2)

        def autocorrelation(x, lags):
            # Normalization and padding for FFT-based correlation
            std_val = np.std(x)
            if std_val < EPSILON:
                return np.zeros(lags)

            x_norm = (x - np.average(x)) / std_val
            (n,) = x_norm.shape
            # Pad with zeros to avoid cyclic correlation artifacts
            xp = ifftshift(x_norm)
            xp = np.r_[xp[: n // 2], np.zeros_like(xp), xp[n // 2 :]]

            f = fft(xp)
            p = np.absolute(f) ** 2
            pi = ifft(p)

            # Divide by number of overlapping points for unbiased estimator
            denom = n - np.arange(lags)
            return np.real(pi)[:lags] / denom

        if self.autocorrelation_invalid_method == "average" and not np.all(indices):
            average = np.nanmean(R_fft[indices], axis=0)
            processed_r_fft = R_fft.copy()
            processed_r_fft[~indices] = average
        else:
            processed_r_fft = R_fft

        max_m = (
            self.max_mode if self.max_mode != "auto" else self.auto_max_mode_range[1]
        )
        modes_count = max_m + self.mode_sum_range

        real = []
        real_err = []
        imag = []
        imag_err = []

        for uu in processed_r_fft.T[:modes_count]:
            r = autocorrelation(np.real(uu), max_lags)
            real.append(r)
            real_err.append(1.0 / np.sqrt(n_frames - np.arange(max_lags)))

            i = autocorrelation(np.imag(uu), max_lags)
            imag.append(i)
            imag_err.append(1.0 / np.sqrt(n_frames - np.arange(max_lags)))

        return np.concatenate(
            (np.stack((real, real_err), axis=2), np.stack((imag, imag_err), axis=2)),
            axis=2,
        )

    def decay_times(
        self, R_fft: np.ndarray, indices: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Finds the decay time for each mode. Since u is complex, finds the decay of real [:,0] and imaginary parts [:,2] separately. These should be same.
        [:,1] and [:,3] give the standard error.

        Args:
            q (np.ndarray): _description_
            R_fft (np.ndarray): _description_
            fit_param (Dict): _description_

        Returns:
            np.ndarray: _description_
        """
        self.verify_config()

        if self.decay_time_filter_threshold_s is not None:
            if self.decay_time_filter_threshold_s == "static_shape":
                # subtract mean shape from each mode
                R_fft = R_fft.copy()
                R_fft[indices] -= np.mean(R_fft[indices], axis=0)
            else:
                # apply a high pass filter to R_fft to clean up the signal
                # TODO: split from
                cutoff = 1 / self.decay_time_filter_threshold_s  # frequency in Hz
                nyq = 0.5 * 1000 / self.delay_between_frames_ms
                normal_cutoff = cutoff / nyq
                sos = butter(4, normal_cutoff, btype="high", analog=False, output="sos")
                R_fft = R_fft.copy()
                # Subtract DC component
                R_fft[indices] -= np.mean(R_fft[indices], axis=0)

                # Use a longer pad length to avoid spikes and reflections
                # For low cutoffs, transients can be very long. Use at least one period if possible.
                fps = 1000 / self.delay_between_frames_ms
                padlen = int(fps / cutoff) if cutoff > 0 else 150
                padlen = max(padlen, 3 * 5 * 10)  # At least 10x default

                R_fft[indices] = np.apply_along_axis(
                    lambda x: sosfiltfilt(
                        sos, x, padlen=min(len(x) - 1, padlen), padtype="odd"
                    ),
                    axis=0,
                    arr=R_fft[indices],
                )
        z = self.autocorrelation_functions(R_fft, indices)

        def exponential_decay(x, C_0, tau, B):
            return C_0 * np.exp(-x / tau) + B

        fit_max = self.decay_time_fit_max
        fit_min = self.decay_time_fit_min  # try different ones
        fits = []
        full_fits = []
        if self.autocorrelation_mode == "abs":
            abs_val = np.sqrt(z[:, :, 0] * z[:, :, 0] + z[:, :, 2] * z[:, :, 2])
            abs_err = (
                np.sqrt(
                    np.power(z[:, :, 0] * z[:, :, 1], 2) ** 2
                    + np.power(z[:, :, 2] * z[:, :, 3], 2)
                )
                / abs_val
            )
            for vals, errs in zip(abs_val, abs_err):
                x = np.arange(len(vals[:fit_max]))[fit_min:]
                fit_abs, fit_cov_abs = curve_fit(
                    exponential_decay,
                    x,
                    vals[fit_min:fit_max],
                    sigma=errs[fit_min:fit_max],
                    maxfev=10000,
                    # p0=(vals[fit_min], 50,vals[fit_max:].mean()),
                    # bounds=((0, 0, -1),(1000, 1000, 1))
                )
                err_abs = np.sqrt(np.diag(fit_cov_abs))
                fits.append((fit_abs[1], err_abs[1], fit_abs[1], err_abs[1]))
                full_fits.append({"abs": {"params": fit_abs, "cov": fit_cov_abs}})
        #            plt.plot(vals)
        #            plt.figure()
        else:
            for zz in z:
                x = np.arange(len(zz[:fit_max]))[fit_min:]
                fit_real, fit_cov_real = curve_fit(
                    exponential_decay,
                    x,
                    zz[fit_min:fit_max, 0],
                    sigma=zz[fit_min:fit_max, 1],
                    maxfev=10000,
                    p0=[zz[0, 0] - zz[fit_max, 0], 1, zz[fit_max, 0]],
                )

                err_real = np.sqrt(np.diag(fit_cov_real))
                fit_imag, fit_cov_imag = curve_fit(
                    exponential_decay,
                    x,
                    zz[fit_min:fit_max, 2],
                    sigma=zz[fit_min:fit_max, 3],
                    p0=fit_real,
                    maxfev=10000,
                )
                err_imag = np.sqrt(np.diag(fit_cov_imag))
                fits.append((fit_real[1], err_real[1], fit_imag[1], err_imag[1]))
                full_fits.append(
                    {
                        "real": {"params": fit_real, "cov": fit_cov_real},
                        "imag": {"params": fit_imag, "cov": fit_cov_imag},
                    }
                )

        return_fits = np.asarray(fits, dtype=float)
        return_fits *= self.delay_between_frames_ms

        return return_fits, z, full_fits

    def decay_times_with_linear(
        self,
        R_fft: np.ndarray,
        indices: Union[np.ndarray, None] = None,
        generate_plots=False,
    ) -> np.ndarray:
        self.verify_config()
        """
        New version of decay_times that fits a linear term in the exponential decay.
        """
        if self.decay_time_filter_threshold_s is not None:
            if self.decay_time_filter_threshold_s == "static_shape":
                # subtract mean shape from each mode
                R_fft = R_fft.copy()
                R_fft[indices] -= np.mean(R_fft[indices], axis=0)
            else:
                raise NotImplementedError(
                    "Filtering not implemented as it probably doesn't make sense with linear term"
                )
        z = self.autocorrelation_functions(R_fft, indices)

        def single_full_fit_model(tau_lags, alpha, C_th, slope_lin, D=0):
            return (
                C_th * np.exp(-alpha * tau_lags) + slope_lin * tau_lags + (1 - C_th + D)
            )

        fit_max = self.decay_time_fit_max
        fit_min = self.decay_time_fit_min
        fits = []
        full_fits = []

        # if generate_plots:
        #    os.makedirs("decay_fits", exist_ok=True)

        if self.autocorrelation_mode == "abs":
            for zz in z:
                x = np.arange(len(zz[:fit_max]))[fit_min:]
                try:
                    fit_abs, fit_cov_abs = curve_fit(
                        single_full_fit_model,
                        x,
                        zz[fit_min:fit_max, 0],
                        sigma=zz[fit_min:fit_max, 1],
                        maxfev=10000,
                        p0=[1 / 50, zz[0, 0], 0, 0],
                        bounds=([0, 0, -np.inf, -1], [np.inf, 1.01, np.inf, 1]),
                    )
                    err_abs = np.sqrt(np.diag(fit_cov_abs))
                    fits.append(
                        (
                            1 / fit_abs[0],
                            err_abs[0] / (fit_abs[0] ** 2),
                            1 / fit_abs[0],
                            err_abs[0] / (fit_abs[0] ** 2),
                            fit_abs[1],
                            fit_abs[1],
                            fit_abs[2],
                            fit_abs[2],
                            np.nan,
                            np.nan,
                        )
                    )
                except RuntimeError:
                    fits.append(
                        (
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
                        )
                    )
        else:
            for i, zz in enumerate(z):
                x = np.arange(len(zz[:fit_max]))[fit_min:]
                try:
                    fit_real, fit_cov_real = curve_fit(
                        single_full_fit_model,
                        x,
                        zz[fit_min:fit_max, 0],
                        sigma=zz[fit_min:fit_max, 1],
                        maxfev=10000,
                        p0=[1 / 50, zz[0, 0], 0, 0],
                        bounds=([0, 0, -np.inf, -1e-5], [np.inf, 1.01, np.inf, 1e-5]),
                    )
                    err_real = np.sqrt(np.diag(fit_cov_real))
                except RuntimeError:
                    fit_real = [np.nan, np.nan, np.nan, np.nan]
                    err_real = [np.nan, np.nan, np.nan, np.nan]

                try:
                    fit_imag, fit_cov_imag = curve_fit(
                        single_full_fit_model,
                        x,
                        zz[fit_min:fit_max, 2],
                        sigma=zz[fit_min:fit_max, 3],
                        p0=[1 / 50, zz[0, 2], 0, 0],
                        bounds=([0, 0, -np.inf, -1e-5], [np.inf, 1.01, np.inf, 1e-5]),
                        maxfev=10000,
                    )
                    err_imag = np.sqrt(np.diag(fit_cov_imag))
                except RuntimeError:
                    fit_imag = [np.nan, np.nan, np.nan, np.nan]
                    err_imag = [np.nan, np.nan, np.nan, np.nan]

                if generate_plots:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    # Extend x-axis for plotting to show linear trend
                    x_plot = np.linspace(x[0], x[-1] * 1.2, 200)

                    # Real part
                    decay_time_real = (
                        1 / fit_real[0] * self.delay_between_frames_ms
                        if not np.isnan(fit_real[0])
                        else np.nan
                    )
                    fit_label_real = (
                        f"Real Fit ($\\tau$={decay_time_real:.2f} ms)"
                        if not np.isnan(decay_time_real)
                        else "Real Fit (failed)"
                    )
                    ax1.errorbar(
                        x,
                        zz[fit_min:fit_max, 0],
                        yerr=zz[fit_min:fit_max, 1],
                        fmt="o",
                        label="Real Data",
                        markersize=3,
                        zorder=1,
                    )
                    ax1.plot(
                        x_plot,
                        single_full_fit_model(x_plot, *fit_real),
                        label=fit_label_real,
                        zorder=2,
                    )
                    ax1.plot(
                        x_plot,
                        fit_real[2] * x_plot + fit_real[3] + (1 - fit_real[1]),
                        label="Linear Component",
                        linestyle="--",
                        zorder=2,
                    )
                    ax1.set_title(f"Mode {i} Real Part")
                    ax1.legend()

                    # Imaginary part
                    decay_time_imag = (
                        1 / fit_imag[0] * self.delay_between_frames_ms
                        if not np.isnan(fit_imag[0])
                        else np.nan
                    )
                    fit_label_imag = (
                        f"Imaginary Fit ($\\tau$={decay_time_imag:.2f} ms)"
                        if not np.isnan(decay_time_imag)
                        else "Imaginary Fit (failed)"
                    )
                    ax2.errorbar(
                        x,
                        zz[fit_min:fit_max, 2],
                        yerr=zz[fit_min:fit_max, 3],
                        fmt="o",
                        label="Imaginary Data",
                        markersize=3,
                        zorder=1,
                    )
                    ax2.plot(
                        x_plot,
                        single_full_fit_model(x_plot, *fit_imag),
                        label=fit_label_imag,
                        zorder=2,
                    )
                    ax2.plot(
                        x_plot,
                        fit_imag[2] * x_plot + fit_imag[3] + (1 - fit_imag[1]),
                        label="Linear Component",
                        linestyle="--",
                        zorder=2,
                    )
                    ax2.set_title(f"Mode {i} Imaginary Part")
                    ax2.legend()

                    # os.makedirs("decay_fits", exist_ok=True)
                    # plt.savefig(f"decay_fits/mode_{i}.png")
                    # plt.close(fig)

                fits.append(
                    (
                        1 / fit_real[0],
                        err_real[0] / (fit_real[0] ** 2),
                        1 / fit_imag[0],
                        err_imag[0] / (fit_imag[0] ** 2),
                        fit_real[1],
                        fit_imag[1],
                        fit_real[2],
                        fit_imag[2],
                        fit_real[3],
                        fit_imag[3],
                    )
                )
                full_fits.append(
                    {
                        "real": {
                            "params": fit_real,
                            "cov": fit_cov_real if "fit_cov_real" in locals() else None,
                        },
                        "imag": {
                            "params": fit_imag,
                            "cov": fit_cov_imag if "fit_cov_imag" in locals() else None,
                        },
                    }
                )

        return_fits = np.asarray(fits, dtype=float)
        return_fits[:, :4] *= self.delay_between_frames_ms

        return return_fits, z, full_fits

    def calculate_variances(self, contours, good_indices):
        """Calculate variances with different subtract options.
        Requires radial form of contours

        Args:
            contours (_type_): _description_
        """
        if len(contours.shape) > 2:
            raise NotImplementedError("Only radial contours supported")

        cutoff = 1000 / (
            self.delay_between_frames_ms * self.sub_radius_rolling_mean_window
        )  # frequency in Hz
        nyq = 0.5 * 1000 / self.delay_between_frames_ms
        normal_cutoff = cutoff / nyq
        sos = butter(4, normal_cutoff, btype="high", analog=False, output="sos")
        # Subtract DC component
        temp_contours = contours[good_indices] - contours[good_indices].mean(axis=0)

        # Use a longer pad length to avoid spikes and reflections
        fps = 1000 / self.delay_between_frames_ms
        padlen = int(fps / cutoff) if cutoff > 0 else 150
        padlen = max(padlen, 3 * 5 * 10)

        contours_butter = np.apply_along_axis(
            lambda x: sosfiltfilt(
                sos, x, padlen=min(len(x) - 1, padlen), padtype="odd"
            ),
            axis=0,
            arr=temp_contours,
        )
        butter_variance = np.sqrt((contours_butter**2).mean()) * self.nm_per_px
        static_shape_variance = (
            np.sqrt(
                (
                    (contours[good_indices] - contours[good_indices].mean(axis=0)) ** 2
                ).mean()
            )
            * self.nm_per_px
        )

        return static_shape_variance, butter_variance

    def process_amplitudes(self, q_range, good_indices, fft_r, R_mean_pixels, fft_mean):
        # TODO: move more processing here
        if self.sub_radius == "static_shape" or (
            self.sub_radius == "rolling_mean"
            and self.sub_radius_rolling_mean_window >= fft_r.shape[0]
        ):
            fft_r[:, 1:] -= fft_mean[1:]
        elif (
            self.sub_radius == "rolling_mean"
        ):  # TODO: this handles the edges weirdly in some cases
            rolling_means = uniform_filter1d(
                fft_r[good_indices, 1:],
                self.sub_radius_rolling_mean_window,
                axis=0,
                mode="reflect",
            )
            fft_r[good_indices, 1:] -= rolling_means
        elif self.sub_radius == "butter":
            cutoff = 1000 / (
                self.delay_between_frames_ms * self.sub_radius_rolling_mean_window
            )  # frequency in Hz
            nyq = 0.5 * 1000 / self.delay_between_frames_ms
            normal_cutoff = cutoff / nyq
            sos = butter(4, normal_cutoff, btype="high", analog=False, output="sos")

            # Subtract DC component
            fft_r[good_indices] -= fft_r[good_indices].mean(axis=0)

            # Use a longer pad length to avoid spikes and reflections
            fps = 1000 / self.delay_between_frames_ms
            padlen = int(fps / cutoff) if cutoff > 0 else 150
            padlen = max(padlen, 3 * 5 * 10)

            fft_r[good_indices] = np.apply_along_axis(
                lambda x: sosfiltfilt(
                    sos, x, padlen=min(len(x) - 1, padlen), padtype="odd"
                ),
                axis=0,
                arr=fft_r[good_indices],
            )
        elif self.sub_radius == "custom":
            fft_r[good_indices, :] = self.sub_radius_filter(fft_r[good_indices])
        return (q_range, good_indices, fft_r, R_mean_pixels, fft_mean)

    def amplitudes(self, indices: np.ndarray, contours: np.ndarray):
        q_range, good_indices, fft_r, R_mean_pixels, fft_mean = self.amplitudes_raw(
            indices, contours
        )
        return self.process_amplitudes(
            q_range, good_indices, fft_r, R_mean_pixels, fft_mean
        )

    def amplitudes_raw(
        self, indices: np.ndarray, contours: np.ndarray
    ) -> Tuple[range, np.ndarray, np.ndarray, float]:
        """Calculates radial FFT amplitudes from contours.
        Optimized to pre-allocate memory and process contours one-by-one.
        """
        indices = np.array(indices)

        def thin_contour(
            indices_thin: np.ndarray, fft_r: np.ndarray, thinning_factor: int
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Reduces contour size by only considering one in every <thinning_factor> points

            Args:
                indices_thin (np.ndarray): _description_
                fft_r (np.ndarray): _description_
                thinning_factor (int): Number of points to skip each time when parsing through contour of a given frame

            Returns:
                Tuple[np.ndarray, np.ndarray]: thinned indices; radial fft
            """
            indices_thin_return = indices_thin[::thinning_factor]
            fft_r = fft_r[::thinning_factor]
            return indices_thin_return, fft_r

        def remove_spikes(
            indices: np.ndarray,
            fft_r: np.ndarray,
            R_mean: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Removes bad contours according to the maximum displacement amplitude given in the fit parameters

            Args:
                indices_to_remove (np.ndarray): _description_
                fft_r (np.ndarray): _description_
                R_mean (float): _description_
                fit_param_spikes (Dict): _description_

            Returns:
                Tuple[np.ndarray, np.ndarray]: _description_
            """
            bad_mask = np.any(
                np.abs(fft_r[:, 1:] / R_mean) > self.maximum_outlier_fluctuation,
                axis=1,
            )
            indices[bad_mask] = False
            return indices, fft_r

        def get_mean_radius(fft_r_arg: np.ndarray) -> float:
            return np.real(fft_r_arg[:, 0].mean())

        # Pre-allocate fft_r to save memory and avoid multiple list copies
        n_frames = len(contours)
        fft_r = None
        min_fft_l = None

        # First pass to find first valid contour and determine length
        for i in range(n_frames):
            c = contours[i]
            if c is not None and len(c) >= 32:
                if c.ndim == 2:
                    r_tmp, _ = self.contour_to_radii_angles(c)
                else:
                    r_tmp = c
                min_fft_l = len(r_tmp)
                break

        if min_fft_l is None:
            self.logger.error("No valid contours found!")
            return range(0), indices, np.array([]), 0.0, np.array([])

        fft_r = np.zeros((n_frames, min_fft_l), dtype=np.complex128)

        # Process contours one by one to fill fft_r
        angles_for_noise = []
        for index, contour in enumerate(contours):
            # this line is evil
            # it takes 96% of runtime when included
            # logging.debug(f"contour in fit_mps.amplitudes = {contour}")
            # if not indices[index]: #this is new, makes sense to me
            #    logging.debug("Contour skipped based on indices")
            #    continue
            if (
                contour is None or len(contour) < 32
            ):  # Lowered from 100 to allow investigation
                if indices[index]:
                    self.logger.warning(
                        f"Valid contour length: {len(contour)}, marking as invalid"
                    )
                    indices[index] = False
                continue

            if contour.ndim == 2:
                radius_px, angle = self.contour_to_radii_angles(contour)
            else:
                radius_px = contour
                angle = np.linspace(0, 2 * np.pi, len(contour), endpoint=False)

            # Compute FFT and store directly in pre-allocated array
            # Note: radius_px might be longer than min_fft_l if contours vary in length
            f = fft(radius_px) / float(len(radius_px))
            fft_r[index] = f[:min_fft_l]

            if self.angular_noise is None and indices[index]:
                angles_for_noise.append(angle)

        if self.angular_noise is None and angles_for_noise:
            # Estimate angular noise from collected angles
            self.angular_noise_estimate = np.mean(
                [
                    np.sqrt(0.5 * np.var(np.diff(a * len(a) / (2 * np.pi)) - 1))
                    for a in angles_for_noise
                ]
            )

        R_mean_pixels = get_mean_radius(fft_r[indices])
        fft_mean = fft_r[indices].mean(axis=0)

        if self.remove_outliers:
            indices, fft_r = remove_spikes(indices, fft_r, R_mean_pixels)
            R_mean_pixels = get_mean_radius(fft_r[indices])
            fft_mean = fft_r[indices].mean(axis=0)

        if self.thinning_factor:
            indices, fft_r = thin_contour(indices, fft_r, self.thinning_factor)
            R_mean_pixels = get_mean_radius(fft_r[indices])
            fft_mean = fft_r[indices].mean(axis=0)

        return (
            range(min_fft_l),
            indices,
            fft_r,
            R_mean_pixels * self.nm_per_px,
            fft_mean,
        )

    def contour_to_radii_angles(
        self, contour: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts contour given as xy points to radii and angles.
        Split from fit_mps/amplitudes as it's useful elsewhere.
        TODO: internal types within output arrays


        Args:
            contour (np.ndarray): _description_
            fit_param (Dict): _description_

        Returns:
            Tuple[np.ndarray[...], np.ndarray[...]]: _description_
        """
        arctan2_arg = np.array(((contour - contour.mean(axis=0)).T)[::-1]).astype(float)
        # logging.debug(f"Passing to arctan2: {arctan2_arg}")  # TODO: turn to debug
        angle = np.arctan2(*arctan2_arg)
        angle += np.pi
        radius = np.sqrt(np.sum((contour - contour.mean(axis=0)) ** 2, axis=1))
        # logging.debug(f"FOR TYPING: contour_to_radii_angles.radius = {radius}")
        # radius *= fit_param_arg["nm_per_px"]  # Convert to nm -- TOO EARLY
        radius += (
            self.radius_correction_nm / self.nm_per_px
        )  # convert to pixels as above comment
        radius = radius[np.argsort(angle)]  # sort the radius list
        angle = angle[np.argsort(angle)]

        return radius, angle

    def fit_limits_alpha_beta(self, R):
        """Maps fit_limits from tension bending in 1e-7, 1e-21 units to alpha beta limits

        alpha depends on sqrt(kappa/sigma^3)
        beta depends on sqrt(sigma/kappa)

        returns formatted for curve_fit [[alpha_min, beta_min], [alpha_max, beta_max]]
        Args:
            R (float): Radius in nm
        """
        if self.fit_limits is None:
            return (-np.inf, np.inf)
        # [[tension_min, tension_max], [bending_min, bending_max]]
        self.fit_limits = np.array(self.fit_limits)
        min_max = self.tension_bending_to_alpha_beta(
            self.fit_limits[0, 1], self.fit_limits[1, 0], R
        )
        max_min = self.tension_bending_to_alpha_beta(
            self.fit_limits[0, 0], self.fit_limits[1, 1], R
        )
        return [[min_max[0], max_min[1]], [max_min[0], min_max[1]]]

    def tension_bending_to_alpha_beta(self, tension, bending, R):
        kT = self.kB_T
        b = np.sqrt(tension * R * R / (bending * 1e4))
        a = 1e4 * kT / (4 * np.pi * tension * b)
        return (
            a,
            b,
        )  # (1e2*kT*R)/np.sqrt(tension*bending)/(4*np.pi), np.sqrt(tension*R*R/(bending*1e4))

    def tension_bending(
        self, optimal_parameters: np.ndarray, pcov: np.ndarray, R, kT_temp
    ) -> np.ndarray:
        """Takes optimal_parameters, pcov output from scipy.optimize.curve_fit and ... TODO

        Args:
            optimal_parameters (array): Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *optimal_parameters) - ydata is minimized
            pcov (2D array): The estimated covariance of optimal_parameters. The diagonals provide the variance of the parameter estimate.
            R (_type_): cell radius in nm TODO
            kT_temp (_type_): k*T in 10^(-21) TODO

        Returns:
            np.ndarray: _description_
        """
        a, b = optimal_parameters
        va, vb = np.diag(pcov)
        vab = pcov[0, 1]

        # a in nm^2, b is dimensionless
        tension = (1e4) * kT_temp / (4 * np.pi * a * b)
        tensionv = ((kT_temp / (4 * np.pi)) ** 2) * (
            va * ((1 / ((a**2) * b)) ** 2)
            + vb * ((1 / ((b**2) * a)) ** 2)
            + 2 * vab / ((a * b) ** 3)
        )
        tension_err = np.sqrt(tensionv) * (1e4)

        # R in nm, kT temp in 1e-21 J, 1e-18 * 1e-21 if a in nm^2
        # si nm^2 * 1e-21 J/nm^2
        # -> in kT units!!!!!!!
        # returns in kT units
        bending = (R**2) * kT_temp / (4 * np.pi * a * (b**3))
        bendingv = (((R**2) * kT_temp / (4 * np.pi)) ** 2) * (
            va * ((1 / ((a**2) * (b**3))) ** 2)
            + vb * ((3 / ((a**1) * (b**4))) ** 2)
            + 2 * 3 * vab / (((a) ** 3) * ((b) ** 7))
        )  # was missing factor of 3 from d/db
        bendingv += (
            4 * (self.radius_uncertainty / R * bending) ** 2
        )  # probably negligible

        bending_err = np.sqrt(bendingv)  # TODO: rewrite to SI like a sane person

        return np.asarray([[tension, tension_err], [bending, bending_err]], dtype=float)

    def mean_powerspectrum(self, x_tilde: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x_tilde (np.ndarray): _description_
            fit_param_mean_ps (Dict): _description_

        Returns:
            np.ndarray: _description_
        """
        angular_noise = self.angular_noise
        radial_noise = self.radial_noise
        error_min = self.noise_floor_above
        M, N = x_tilde.shape

        Y = np.abs(x_tilde) ** 2
        Y_m = Y.mean(axis=0)

        # plt.figure("Full spectrum")
        # plt.plot(Y_m, label = "Full, uncorrected")

        if radial_noise is None:
            # TODO: why sqrt and *N?
            radial_noise = np.sqrt(
                np.mean(Y_m[error_min : self.noise_floor_below]) * float(N)
            )
            self.radial_noise_estimate = radial_noise
            # plt.axhline(np.mean(Y_m[error_min:N//2]), linestyle = '--', c = 'k', label = "Estimated noise floor")

        if angular_noise is None:
            angular_noise = self.angular_noise_estimate

        if angular_noise > 0:
            c = 2 * np.pi / float(N)
            correction_matrix = np.diagflat(
                [np.exp(-((c * angular_noise * q) ** 2)) for q in range(N)]
            ) + np.asarray(
                [
                    [
                        (1 - np.exp(-((c * angular_noise * q) ** 2))) / float(N)
                        for q in range(N)
                    ]
                    for m in range(N)
                ]
            )
            inverse_matrix = np.linalg.inv(correction_matrix)
            Y_m = np.dot(inverse_matrix, Y_m)

        Y_m[Y_m < 0] = 0

        cov_m = np.ones((N // 2, N // 2)) * 2 * (radial_noise**4) / float(
            N**2
        ) + Y_m[: N // 2] * np.eye(N // 2) * 2 * (radial_noise**2) / float(N**1)
        cov_m = cov_m / float(M)
        error_m = np.sqrt(Y.var(axis=0)[: N // 2] / float(M) + np.diag(cov_m))

        Y_m = Y_m[: N // 2]
        corrected_ps = np.column_stack((Y_m - (radial_noise**2) / float(N), error_m))

        return corrected_ps * self.nm_per_px * self.nm_per_px  # px^2->nm^2

    def _get_rolling_window_correction_factor(self, taus):
        taus_mean = np.array(taus)[:, [0, 2]].mean(axis=1)
        taus_err = _mean_tau_error(taus)

        if self.sub_radius == "rolling_mean":
            T_const = (
                0.336
                * (self.sub_radius_rolling_mean_window * self.delay_between_frames_ms)
                ** 0.954
            )
            exponent = self.rolling_window_correction_exponent
            correction = 1 + (taus_mean / T_const) ** exponent
            d_correction_d_tau = (
                exponent / T_const * (taus_mean / T_const) ** (exponent - 1)
            )
            correction_err = np.abs(d_correction_d_tau * taus_err)
        elif self.sub_radius == "butter":
            if self.rolling_window_correction == "old":
                T_const = 0.1374 * (
                    self.sub_radius_rolling_mean_window * self.delay_between_frames_ms
                )
                exponent = 1.22
                correction = 1 + (taus_mean / T_const) ** exponent
                d_correction_d_tau = (
                    exponent / T_const * (taus_mean / T_const) ** (exponent - 1)
                )
                correction_err = np.abs(d_correction_d_tau * taus_err)
            else:  # analytical/filtfilt
                T_const = (
                    self.sub_radius_rolling_mean_window * self.delay_between_frames_ms
                )
                ratio = taus_mean / T_const

                # Vectorized correction calculation
                butterworth_filtfilt_correction_vec = np.vectorize(
                    butterworth_filtfilt_correction
                )
                correction = butterworth_filtfilt_correction_vec(ratio)

                # Error propagation (only if needed)
                if (
                    self.rolling_window_correction_location == "spectrum"
                    and self.propagate_tau_error
                ):
                    # Estimate derivative numerically: d_corr/d_ratio
                    epsilon = 1e-4
                    ratio_plus = ratio + epsilon
                    ratio_minus = ratio - epsilon
                    corr_plus = butterworth_filtfilt_correction_vec(ratio_plus)
                    corr_minus = butterworth_filtfilt_correction_vec(ratio_minus)
                    d_correction_d_ratio = (corr_plus - corr_minus) / (2 * epsilon)

                    d_ratio_d_tau = 1 / T_const
                    d_correction_d_tau = d_correction_d_ratio * d_ratio_d_tau
                    correction_err = np.abs(d_correction_d_tau * taus_err)
                else:
                    correction_err = np.zeros_like(correction)

        else:
            raise NotImplementedError(
                f"Correction not implemented for {self.sub_radius}"
            )

        correction[taus_mean < 0] = 1
        correction_err[taus_mean < 0] = 0
        correction[np.isnan(correction)] = 1
        correction_err[np.isnan(correction_err)] = 0
        return correction, correction_err

    def apply_rolling_window_correction_analytical(self, mps, mps_err, taus):
        correction, correction_err = self._get_rolling_window_correction_factor(taus)

        pad_len = len(mps) - len(correction)
        if pad_len > 0:
            correction = np.pad(correction, (0, pad_len), "constant", constant_values=1)
        else:
            correction = correction[: len(mps)]

        if self.propagate_tau_error:
            pad_len_err = len(mps) - len(correction_err)
            if pad_len_err > 0:
                correction_err = np.pad(
                    correction_err,
                    (0, pad_len_err),
                    "constant",
                    constant_values=0,
                )
            else:
                correction_err = correction_err[: len(mps)]

            # Propagate errors: new_mps_err = sqrt((correction*mps_err)^2 + (mps*correction_err)^2)
            new_mps_err = np.sqrt(
                (correction * mps_err) ** 2 + (mps * correction_err) ** 2
            )
        else:
            new_mps_err = mps_err * correction

        return mps * correction, new_mps_err, correction

    def apply_rolling_window_correction(self, mps, mps_err, taus):
        correction, correction_err = self._get_rolling_window_correction_factor(taus)

        correction = np.pad(
            correction, (0, len(mps) - len(correction)), "constant", constant_values=1
        )
        correction_err = np.pad(
            correction_err,
            (0, len(mps) - len(correction_err)),
            "constant",
            constant_values=0,
        )

        # Propagate errors
        new_mps_err = np.sqrt((correction * mps_err) ** 2 + (mps * correction_err) ** 2)

        return mps * correction, new_mps_err, correction

    def apply_original_exposure_correction(self, mps, mps_err, taus):
        # Extract mean tau and its error
        taus_mean = np.array(taus)[:, [0, 2]].mean(axis=1)
        taus_err = _mean_tau_error(taus)

        # Pad to match mps length if necessary
        if len(taus_mean) < len(mps):
            pad_width = len(mps) - len(taus_mean)
            taus_mean = np.pad(
                taus_mean, (0, pad_width), "constant", constant_values=np.nan
            )
            taus_err = np.pad(taus_err, (0, pad_width), "constant", constant_values=0)

        # Define the ratio t = tau / T_exp
        ratios = (taus_mean / self.exposure_time_ms)[: len(mps)]
        taus_err_sliced = taus_err[: len(mps)]

        # Avoid division by zero or invalid values
        valid_mask = (ratios > 0) & np.isfinite(ratios)

        correction = np.ones_like(mps)
        new_mps = mps.copy()
        new_mps_err = mps_err.copy()

        if np.any(valid_mask):
            r = ratios[valid_mask]
            tau_err_valid = taus_err_sliced[valid_mask]

            # Correction factor C(t) from Faucon et al. 1989, Eq. (52)
            # C(t) = 2*t^2 * (1/t + exp(-1/t) - 1)
            C = 2 * r**2 * (1 / r + np.exp(-1 / r) - 1)

            # The power spectrum is divided by C, so the effective correction is 1/C
            inv_C = 1 / C

            # Derivative of C with respect to t: dC/dt = 2*t - 2*exp(-1/t)
            # dc/dt = dC/dr * dr/dt = dC/dr * (1/T_exp)
            dC_dr = 2 * np.exp(-1 / r) * (2 * r + 1) - 4 * r + 2
            dC_dt = dC_dr * (1 / self.exposure_time_ms)

            # Apply correction to MPS
            new_mps[valid_mask] = mps[valid_mask] * inv_C

            # Error propagation for Y = M/C
            # Var(Y) = (1/C)^2*Var(M) + (M^2/C^4) * (dC/dt * dt)^2

            var_M = mps_err[valid_mask] ** 2
            var_t = (tau_err_valid / self.exposure_time_ms) ** 2

            var_Y = (inv_C**2 * var_M) + (
                (mps[valid_mask] ** 2 / C**4) * (dC_dt**2) * var_t
            )

            new_mps_err[valid_mask] = np.sqrt(var_Y)
            correction[valid_mask] = inv_C

        if self.propagate_tau_error:
            return new_mps, new_mps_err, correction

        return new_mps, mps * correction, correction

    def apply_rautu_const_exposure_correction(self, mps, mps_err, taus):
        # Extract mean tau and its error
        taus_mean = np.array(taus)[:, [0, 2]].mean(axis=1)
        taus_err = _mean_tau_error(taus)

        # Pad to match mps length if necessary
        if len(taus_mean) < len(mps):
            pad_width = len(mps) - len(taus_mean)
            taus_mean = np.pad(
                taus_mean, (0, pad_width), "constant", constant_values=np.nan
            )
            taus_err = np.pad(taus_err, (0, pad_width), "constant", constant_values=0)

        g = (taus_mean / self.exposure_time_ms) ** -1
        valid_mask = (g > 0) & np.isfinite(g)

        correction = np.ones_like(mps)
        new_mps = mps.copy()
        new_mps_err = mps_err.copy()

        if np.any(valid_mask):
            g_valid = g[valid_mask]
            tau_err_valid = taus_err[valid_mask]
            taus_mean_valid = taus_mean[valid_mask]

            # Correction factor is ((1 - exp(-g)) / g)^2
            atf = ((1 - np.exp(-g_valid)) / g_valid) ** 2
            inv_atf = 1 / atf

            # Apply correction
            new_mps[valid_mask] = mps[valid_mask] * inv_atf

            # Error propagation
            # Y = M / A(g(t))
            # Var(Y) = (1/A)^2 Var(M) + (M/A^2 * dA/dg * dg/dt)^2 Var(t)
            d_atf_d_g = (
                2
                * (1 - np.exp(-g_valid))
                / g_valid
                * (np.exp(-g_valid) / g_valid - (1 - np.exp(-g_valid)) / g_valid**2)
            )
            d_g_d_tau = -self.exposure_time_ms / (taus_mean_valid**2)

            var_M = mps_err[valid_mask] ** 2
            var_tau = tau_err_valid**2

            var_Y = (inv_atf**2 * var_M) + (
                (mps[valid_mask] / atf**2) ** 2 * (d_atf_d_g * d_g_d_tau) ** 2 * var_tau
            )

            new_mps_err[valid_mask] = np.sqrt(var_Y)
            correction[valid_mask] = inv_atf

        if self.propagate_tau_error:
            return new_mps, new_mps_err, correction

        return new_mps, mps * correction, correction

    def process_contours(
        self,
        contours: np.ndarray,
        indices: np.ndarray,
        plot=False,
        print_res=False,
        plot_file=None,
        plot_decay_fits=False,
    ) -> Dict:
        self.verify_config()
        q, indices, R_fft, radius_nm, fft_mean = self.amplitudes_raw(indices, contours)

        acf = None
        acf_fits = None
        # Initial calculation of taus
        if (
            self.exposure_time_ms is not None
            or self.rolling_window_correction
            or self.fit_viscosity
            or self.decay_time_use_fit
        ):
            if self.decay_times_model == "exponential":
                taus, acf, acf_fits = self.decay_times(R_fft, indices)
            elif self.decay_times_model == "linear_exponential":
                taus, acf, acf_fits = self.decay_times_with_linear(
                    R_fft, indices, generate_plots=plot_decay_fits
                )
            else:
                raise ValueError(f"Unknown decay_times_model: {self.decay_times_model}")
        else:
            taus = None

        q, indices, R_fft, radius_nm, fft_mean = self.process_amplitudes(
            q, indices, R_fft, radius_nm, fft_mean
        )

        # Get the initial, uncorrected power spectrum
        mps, mps_err = self.mean_powerspectrum(R_fft[indices]).T
        if self.extra_mps_error:
            mps_err += self.extra_mps_error

        original_mps = mps.copy()
        original_mps_err = mps_err.copy()

        # Apply initial corrections before the first fit, using unfitted taus
        if (
            self.rolling_window_correction
            and self.rolling_window_correction_location == "spectrum"
        ):
            if self.rolling_window_correction == "old":
                mps, mps_err, rw_correction = self.apply_rolling_window_correction(
                    mps, mps_err, taus
                )
            else:
                mps, mps_err, rw_correction = (
                    self.apply_rolling_window_correction_analytical(mps, mps_err, taus)
                )

        if (
            self.exposure_time_ms is not None
            and self.exposure_time_ms > 0
            and taus is not None
        ):
            if self.exposure_correction_location == "spectrum":
                if self.exposure_correction_method == "original":
                    mps, mps_err, exposure_correction = (
                        self.apply_original_exposure_correction(mps, mps_err, taus)
                    )
                elif self.exposure_correction_method == "rautu_const":
                    mps, mps_err, exposure_correction = (
                        self.apply_rautu_const_exposure_correction(mps, mps_err, taus)
                    )
            elif self.exposure_correction_location == "fit":
                if self.exposure_correction_method not in (
                    "rautu",
                    "rautu_const",
                    "original",
                ):
                    raise ValueError(
                        f"Incompatible exposure correction method '{self.exposure_correction_method}' with fit location."
                    )

        fit_results = {}

        # Iterative fitting loop
        max_iterations = 3
        exposure_correction = None
        original_taus = None
        for i in range(max_iterations):
            if taus is not None:
                original_taus_loop = taus.copy()
            else:
                original_taus_loop = None

            # TODO: work out if this should be done before or after the rolling correction/original shutter
            # Final post-loop processing
            if self.decay_time_brown_correction and taus is not None:
                taus[:, :4] = tau_correction_brown(taus[:, :4])

            # Fit the current version of the power spectrum
            if self.fitting_method is not None:
                fit_results = self.fit_mps(
                    mps, mps_err, radius_nm, taus[:, :4] if taus is not None else None
                )
            else:
                # If no fitting method, just store the current mps and break
                fit_results = {"mps": mps.tolist(), "mps_err": mps_err.tolist()}
                break

            # If we are not refining taus, no need to iterate
            if not (self.decay_time_use_fit) and not self.fit_viscosity:
                break

            # --- Refine decay times and re-apply corrections for the next iteration ---
            if taus is None:
                raise ValueError(
                    "Decay times must be calculated for viscosity fitting or using fitted decay times."
                )

            # Ensure sigma/kappa are available for viscosity fit
            if "sigma" not in fit_results or "kappa" not in fit_results:
                self.logger.warning(
                    "Sigma/Kappa not found in fit results, performing rough fit for viscosity calculation."
                )
                sigma_fit, _, kappa_fit, _, _, _, _ = self.get_rough_fit(
                    mps, mps_err, radius_nm
                )
                fit_results["sigma"] = {"value": sigma_fit}
                fit_results["kappa"] = {"value": kappa_fit}

            # TODO: this should either be taus or original_taus_loop, depending on where brown correction goes
            viscosity_fit, taus = self.fit_decay_times(taus, fit_results, radius_nm)
            fit_results["viscosity_fit"] = viscosity_fit

            # Store original taus from before the first iteration's fit
            if i == 0:
                original_taus = original_taus_loop

            if (
                self.rolling_window_correction
                and self.rolling_window_correction_location == "spectrum"
            ):
                fit_results["rolling_window_correction"] = rw_correction
            if (
                self.exposure_correction_location == "spectrum"
                and taus is not None
                and self.exposure_time_ms is not None
                and self.exposure_time_ms > 0
            ):
                fit_results["exposure_correction"] = exposure_correction

            # For the next iteration, reset mps and re-apply corrections with the NEW taus
            if i < max_iterations - 1:
                mps = original_mps.copy()
                mps_err = original_mps_err.copy()

                if (
                    self.rolling_window_correction
                    and self.rolling_window_correction_location == "spectrum"
                ):
                    if (
                        self.decay_time_brown_correction
                        and self.decay_time_brown_correction_not_in_filter
                    ):
                        taus[:, :4] = tau_correction_brown_reverse(taus[:, :4])
                    if self.rolling_window_correction == "old":
                        mps, mps_err, rw_correction = (
                            self.apply_rolling_window_correction(mps, mps_err, taus)
                        )
                    else:
                        mps, mps_err, rw_correction = (
                            self.apply_rolling_window_correction_analytical(
                                mps, mps_err, taus
                            )
                        )
                    fit_results["rolling_window_correction"] = rw_correction

                if (
                    self.exposure_time_ms is not None
                    and self.exposure_time_ms > 0
                    and taus is not None
                ):
                    if self.exposure_correction_location == "spectrum":
                        if self.exposure_correction_method == "original":
                            mps, mps_err, exposure_correction = (
                                self.apply_original_exposure_correction(
                                    mps, mps_err, taus
                                )
                            )
                        elif self.exposure_correction_method == "rautu_const":
                            mps, mps_err, exposure_correction = (
                                self.apply_rautu_const_exposure_correction(
                                    mps, mps_err, taus
                                )
                            )
        fit_results["original_decay_times"] = {
            "value": original_taus.tolist() if original_taus is not None else None
        }
        fit_results["decay_times"] = {
            "value": taus.tolist() if taus is not None else None
        }
        # Only keep modes up to max(max_mode, highest max mode in the auto search range)
        max_m = self.max_mode if isinstance(self.max_mode, int) else 0
        auto_max = (
            self.auto_max_mode_range[1] if self.auto_max_mode_range is not None else 0
        )
        keep_modes = max(max_m, auto_max)

        if acf is not None:
            # Stage 1 trim: trim immediately to max(save_acf_frames, frames needed for plot)
            plot_xlim = 50.0  # ms
            frames_needed_for_plot = int(
                np.ceil(plot_xlim / self.delay_between_frames_ms)
            )
            save_len = max(self.save_acf_frames, frames_needed_for_plot)
            acf_trimmed = acf[: keep_modes + 1, :save_len, :]
            fit_results["autocorrelation_function"] = acf_trimmed.tolist()
        else:
            fit_results["autocorrelation_function"] = None

        if acf_fits is not None:
            fit_results["autocorrelation_fits"] = acf_fits[: keep_modes + 1]
        else:
            fit_results["autocorrelation_fits"] = None
        fit_results["static_shape"] = fft_mean.tolist()
        fit_results |= {"mps": mps.tolist(), "mps_err": mps_err.tolist()}
        if plot:
            self.plot_spectrum(
                mps,
                mps_err,
                fit_results,
                fit_results.get("alpha_beta"),
                taus,
                self.delta(radius_nm),
                plot_file,
            )
        if print_res:
            self.print_info(fit_results, self.delta(radius_nm))

        if self.radial_variance_calculation:
            variance, butter_variance = self.calculate_variances(
                np.array(contours), np.array(indices)
            )
            fit_results["contour_variance"] = variance
            fit_results["contour_variance_butter"] = butter_variance

        return fit_results

    def delta(self, radius=None):
        if self.depth_of_focus_um is None or self.depth_of_focus_um == 0:
            return 0

        if self.vertical_radius_um is None and radius is not None:
            return self.depth_of_focus_um / (
                radius / 1000.0
            )  # need to both have same units

        if self.vertical_radius_um is not None:
            return self.depth_of_focus_um / self.vertical_radius_um

        return 0

    def set_temperature(self, temperature_K):
        self.kB_T = 1e21 * temperature_K * k

    def gamma_fit_func(self, n, radius, sigma, kappa, gamma):
        # from Yoon 2009
        # SI units
        l = 2 * np.pi * radius
        q = n / radius
        # gamma = 0
        # TODO: switch to SI everywhere, this hurts
        return (
            self.kB_T
            * 1e-21
            / l
            * np.sqrt(kappa / (2 * (sigma * sigma - 4 * kappa * gamma)))
            * (
                1
                / np.sqrt(
                    2 * kappa * q * q
                    + sigma
                    - np.sqrt(sigma * sigma - 4 * kappa * gamma)
                )
                - 1
                / np.sqrt(
                    2 * kappa * q * q
                    + sigma
                    + np.sqrt(sigma * sigma - 4 * kappa * gamma)
                )
            )
        )

    def gamma_fit_func_reparametrised(self, n, radius, a, b, c):
        # from Yoon 2009
        # SI units
        l = 2 * np.pi * radius
        q = n / radius
        # gamma = 0
        # TODO: switch to SI everywhere, this hurts
        return (
            self.kB_T
            * 1e-21
            / l
            / a
            / 2
            * (1 / np.sqrt(q * q + b) - 1 / np.sqrt(q * q + c))
        )

    def gamma_fit_to_physical(self, a, b, c):
        sigma_over_k = b + c
        a_over_k = c - b
        kappa = a / a_over_k
        sigma = sigma_over_k * kappa
        gamma = -(a * a - sigma * sigma) / (4 * kappa)

        return sigma, kappa, gamma

    def get_rough_fit(self, mps, mps_err, radius, max_mode_override=None):
        max_mode = max_mode_override if max_mode_override is not None else self.max_mode
        if max_mode == "auto":
            # This shouldn't happen if called correctly from fit_mps, but just in case
            max_mode = self.auto_max_mode_range[1]

        q_range_fit = np.array(list(range(self.min_mode, max_mode + 1)))
        alpha_beta_fit, alpha_beta_fit_cov = curve_fit(
            original_fit_func,
            q_range_fit,
            mps[q_range_fit[0] : q_range_fit[-1] + 1],
            # p0 = (5, 400.0), #almost no bending
            sigma=mps_err[q_range_fit[0] : q_range_fit[-1] + 1],
            bounds=self.fit_limits_alpha_beta(
                radius
            ),  # [[0, 0], [np.inf, np.inf]], #we want to prevent 0 tension
            xtol=1e-6,  # only want a rough value to start
            maxfev=30000,
            # xtol = 1e-
        )
        physical_fit = self.tension_bending(
            alpha_beta_fit, alpha_beta_fit_cov, radius, self.kB_T
        )

        sigma_fit, sigma_fit_err = physical_fit[0]
        kappa_fit, kappa_fit_err = physical_fit[1]

        return (
            sigma_fit,
            sigma_fit_err,
            kappa_fit,
            kappa_fit_err,
            alpha_beta_fit,
            alpha_beta_fit_cov,
            q_range_fit,
        )

    def _get_fit_func(self, method, taus, delta, max_mode, radius=None):
        """
        Helper to create the fit function based on the method and parameters.
        """
        if method == "original":
            return original_fit_func
        elif method in ("rautu"):
            rw_correction_factors = None
            if (
                self.rolling_window_correction
                and self.rolling_window_correction_location == "fit"
                and taus is not None
            ):
                rw_correction_factors, _ = self._get_rolling_window_correction_factor(
                    taus
                )

            def fit_fn(q, alpha, beta):
                return self.theory_ps(
                    q,
                    alpha,
                    beta,
                    (
                        (np.mean(taus[:, [0, 2]], axis=1) / self.exposure_time_ms) ** -1
                        if taus is not None
                        and self.exposure_time_ms is not None
                        and self.exposure_correction_method in ("rautu", "original")
                        and self.exposure_time_ms > 0
                        and self.exposure_correction_location == "fit"
                        else None
                    ),
                    delta,
                    self.mode_sum_range,
                    rw_correction_factors,
                    self.exposure_correction_method,
                    self.exposure_correction_location,
                    max_mode=max_mode,
                )

            return fit_fn
        elif method == "confinement":
            raise NotImplementedError("confinement fit currently not supported")

        return original_fit_func

    def detect_max_mode_from_results(
        self, mps, mps_err, radius, fit_results, search_range, taus
    ):
        search_min, search_max = search_range

        # Reconstruct fit_fn using helper
        # We need taus and delta
        if (
            taus is None
            and fit_results.get("decay_times")
            and fit_results["decay_times"]["value"] is not None
        ):
            taus = np.array(fit_results["decay_times"]["value"])

        # If taus is None, maybe we can't do rautu?
        # But fit_results should have what we need.

        delta = self.delta(radius)

        # Use search_max as max_mode for prediction to cover the full range
        fit_fn = self._get_fit_func(
            self.fitting_method, taus, delta, search_max, radius
        )

        # Extract parameters
        # For rautu/original, we need alpha, beta
        if "alpha_beta" in fit_results:
            alpha, beta = fit_results["alpha_beta"]
            params = [alpha, beta]
        else:
            # Fallback or error
            self.logger.warning("No alpha_beta in fit_results for auto-detection")
            return search_max

        # Calculate residuals on the full search range
        q_range_search = np.arange(self.min_mode, search_max + 1)
        predicted = fit_fn(q_range_search, *params)

        # Log-log residuals
        mps_diff = mps[q_range_search] - predicted
        mps_diff_ratio = mps_diff / mps[q_range_search]
        residuals = mps_diff_ratio
        # residuals = log_mps - log_pred

        # Simple heuristic: Find where residual > threshold for N consecutive points
        consecutive = 5

        # plt.figure()
        # print(delta)
        # print(radius)
        # plt.plot(q_range_search, mps[q_range_search])
        # plt.plot(q_range_search, predicted)
        # plt.loglog()
        # plt.show()
        detected_mode = search_max

        # Start searching from search_min
        start_idx = max(0, search_min - self.min_mode)

        for i in range(start_idx, len(q_range_search) - consecutive):
            if np.all(residuals[i : i + consecutive] > self.auto_max_mode_threshold):
                # Found start of deviation
                # The cut-off should be just before this
                detected_mode = q_range_search[i]
                break

        # Enforce range
        detected_mode -= 1  # we want to be a bit conservative here
        detected_mode = max(search_min, min(detected_mode, search_max))

        self.logger.debug(f"Auto-detected max_mode: {detected_mode}")
        return detected_mode

    def fit_mps(
        self,
        mps: np.ndarray,
        mps_err: np.ndarray,
        radius: float,
        taus: Union[np.ndarray, None] = None,
        max_mode_override: Union[int, None] = None,
    ) -> Dict:
        """
        Radius in nm

        Returns:
            Dict: _description_
        """
        self.verify_config()

        delta = self.delta(radius)
        fit_r2 = None
        # Handle auto max_mode
        current_max_mode = self.max_mode
        if max_mode_override is not None:
            current_max_mode = max_mode_override

        if current_max_mode == "auto":
            # Recursive logic
            search_min, search_max = self.auto_max_mode_range
            if len(mps) <= search_max:
                search_max = len(mps) - 1

            # 1. Baseline Fit
            baseline_max = min(search_min + 1, search_max)

            # Ensure baseline_max is valid
            if self.min_mode >= baseline_max:
                baseline_max = search_max  # Fallback

            baseline_results = self.fit_mps(
                mps, mps_err, radius, taus, max_mode_override=baseline_max
            )

            # 2. Detect Max Mode
            detected_mode = self.detect_max_mode_from_results(
                mps, mps_err, radius, baseline_results, (search_min, search_max), taus
            )

            # 3. Final Fit
            final_results = self.fit_mps(
                mps, mps_err, radius, taus, max_mode_override=detected_mode
            )
            final_results["detected_max_mode"] = detected_mode
            return final_results

        # Helper to get range
        def get_fit_range():
            return np.arange(self.min_mode, current_max_mode + 1)

        def get_r2(fn, x, y, pcov, params):
            predicted = fn(x, *params)
            residuals = y - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            residual_variance = np.sum(residuals**2) / (len(y) - len(params))

            # Compute the Hessian matrix
            hessian = np.linalg.inv(pcov) / residual_variance

            return 1 - ss_res / ss_tot, predicted, hessian

        def corrected_fit_func(q, alpha, beta):
            # Use helper
            fit_fn = self._get_fit_func(
                self.fitting_method, taus, delta, current_max_mode, radius
            )
            return fit_fn(q, alpha, beta)

        (
            sigma_fit,
            sigma_fit_err,
            kappa_fit,
            kappa_fit_err,
            alpha_beta_fit,
            alpha_beta_fit_cov,
            q_range_fit,
        ) = self.get_rough_fit(mps, mps_err, radius, max_mode_override=current_max_mode)

        rough_fit_r2, predicted_uncorrected, hessian = get_r2(
            original_fit_func,
            q_range_fit,
            mps[q_range_fit[0] : q_range_fit[-1] + 1],
            alpha_beta_fit_cov,
            alpha_beta_fit,
        )
        predicted = predicted_uncorrected
        alpha_beta_fit_err = np.sqrt(np.diag(alpha_beta_fit_cov))
        physical_fit = self.tension_bending(
            alpha_beta_fit, alpha_beta_fit_cov, radius, self.kB_T
        )

        sigma_fit, sigma_fit_err = physical_fit[0]
        kappa_fit, kappa_fit_err = physical_fit[1]

        # Validate rough fit results
        use_fallback = False
        if (
            rough_fit_r2 < self.rough_fit_r2_threshold
            and self.fitting_method != "original"
        ):
            self.logger.debug(
                f"Rough fit R² ({rough_fit_r2:.3f}) below threshold ({self.rough_fit_r2_threshold}), using fallback initial values"
            )
            use_fallback = True
        elif not (self.sigma_limits[0] <= sigma_fit <= self.sigma_limits[1]):
            self.logger.debug(
                f"Rough fit sigma ({sigma_fit:.2f}) outside valid range {self.sigma_limits}, using fallback initial values"
            )
            use_fallback = True
        elif not (self.kappa_limits[0] <= kappa_fit <= self.kappa_limits[1]):
            self.logger.debug(
                f"Rough fit kappa ({kappa_fit:.2f}) outside valid range {self.kappa_limits}, using fallback initial values"
            )
            use_fallback = True

        # important - we use rough fit for original method - do not use constants
        # this took a day to figure out...
        if self.fitting_method == "original":
            use_fallback = False

        uncorrected = {
            "kappa": {"value": kappa_fit, "error": kappa_fit_err, "unit": 1e-21},
            "sigma": {"value": sigma_fit, "error": sigma_fit_err, "unit": 1e-7},
        }

        if use_fallback:
            # Use fallback values and convert to alpha, beta
            sigma_fit = self.fallback_sigma
            kappa_fit = self.fallback_kappa
            alpha_fallback, beta_fallback = self.tension_bending_to_alpha_beta(
                sigma_fit, kappa_fit, radius
            )
            alpha_beta_fit = np.array([alpha_fallback, beta_fallback])
            self.logger.debug(
                f"Using fallback: sigma={sigma_fit:.2f}, kappa={kappa_fit:.2f}, alpha={alpha_fallback:.6f}, beta={beta_fallback:.6f}"
            )

        gamma_res = None

        fit_weights_absolute = True
        if self.fit_weights_method == "mps_error":
            fit_sigmas = mps_err[q_range_fit[0] : q_range_fit[-1] + 1]
            fit_sigmas_si = 1e-18 * mps_err[q_range_fit[0] : q_range_fit[-1] + 1]
        elif self.fit_weights_method == "none":
            fit_sigmas = None
            fit_sigmas_si = None
        elif self.fit_weights_method == "poisson":
            fit_sigmas = np.sqrt(mps[q_range_fit[0] : q_range_fit[-1] + 1])
            fit_sigmas_si = 1e-18 * fit_sigmas
            fit_weights_absolute = False
        else:
            raise NotImplementedError(
                f"Fit weights method {self.fit_weights_method} is not implemented"
            )

        if self.fitting_method == "rautu":
            # print(taus[:,[0,2]].mean(axis=1)[0:50])
            if self.fit_mode == "linear":
                alpha_beta_fit, alpha_beta_fit_cov, infodict, mesg, ier = curve_fit(
                    corrected_fit_func,
                    q_range_fit,
                    mps[q_range_fit[0] : q_range_fit[-1] + 1],
                    alpha_beta_fit,  # start with the basic fit results for easier fitting
                    sigma=fit_sigmas,
                    bounds=self.fit_limits_alpha_beta(radius),
                    # xtol = 1e-5, #should be close enough?
                    maxfev=10000,
                    absolute_sigma=fit_weights_absolute,
                    full_output=True,
                )
            elif self.fit_mode == "log":
                alpha_beta_fit, alpha_beta_fit_cov, infodict, mesg, ier = curve_fit(
                    lambda q, a, b: np.log(
                        np.clip(corrected_fit_func(q, a, b), 1e-50, np.inf)
                    ),
                    q_range_fit,
                    np.log(mps[q_range_fit[0] : q_range_fit[-1] + 1]),
                    alpha_beta_fit,  # start with the basic fit results for easier fitting
                    sigma=None if fit_sigmas is None else np.log(fit_sigmas),
                    bounds=self.fit_limits_alpha_beta(radius),
                    # xtol = 1e-5, #should be close enough?
                    maxfev=10000,
                    absolute_sigma=True,
                    full_output=True,
                )
            else:
                raise NotImplementedError("Invalid fit_mode, use linear or log")
            # print((infodict["fvec"]*infodict["fvec"]).sum())
            fit_r2, predicted, hessian = get_r2(
                corrected_fit_func,
                q_range_fit,
                mps[q_range_fit[0] : q_range_fit[-1] + 1],
                alpha_beta_fit_cov,
                alpha_beta_fit,
            )

            # alpha_beta_fit_err = np.sqrt(np.diag(alpha_beta_fit_cov))
            # print(alpha_beta_fit)
            physical_fit = self.tension_bending(
                alpha_beta_fit, alpha_beta_fit_cov, radius, self.kB_T
            )
            sigma_fit, sigma_fit_err = physical_fit[0]
            kappa_fit, kappa_fit_err = physical_fit[1]
        elif self.fitting_method == "confinement":
            mps_si = np.array(mps) * 1e-9 * 1e-9
            # print([sigma_fit*1e-7, kappa_fit*1e-21, 0])
            confinement_fit, confinement_fit_cov = curve_fit(
                lambda n, sigma, kappa, gamma: self.gamma_fit_func(
                    n, radius * 1e-9, sigma, kappa, gamma
                ),
                q_range_fit,
                mps_si[q_range_fit[0] : q_range_fit[-1] + 1],
                # [10e-7, 200e-21, 50000], #random guess
                [
                    sigma_fit * 1e-7,
                    kappa_fit * 1e-21,
                    50000,
                ],  # start with the basic fit results for easier fitting
                sigma=fit_sigmas_si,
                # bounds=[[0.05e-7, 0.1e-21, 0], [100e-7, 1000e-21, np.inf]],
                # xtol = 1e-6, #should be close enough?
                maxfev=10000,
                # method="lm"
            )
            fit_r2, predicted, hessian = get_r2(
                lambda n, sigma, kappa, gamma: self.gamma_fit_func(
                    n, radius * 1e-9, sigma, kappa, gamma
                ),
                q_range_fit,
                mps_si[q_range_fit[0] : q_range_fit[-1] + 1],
                confinement_fit_cov,
                confinement_fit,
            )

            uncorrected = {
                "kappa": {"value": kappa_fit, "error": kappa_fit_err, "unit": 1e-21},
                "sigma": {"value": sigma_fit, "error": sigma_fit_err, "unit": 1e-7},
            }

            errs = np.sqrt(confinement_fit_cov.diagonal())
            sigma_fit = confinement_fit[0] * 1e7
            sigma_fit_err = errs[0] * 1e7

            kappa_fit = confinement_fit[1] * 1e21
            kappa_fit_err = errs[1] * 1e21

            gamma_res = {"value": confinement_fit[2], "error": errs[2], "unit": 1}
        elif self.fitting_method == "confinement_new":
            mps_si = np.array(mps) * 1e-9 * 1e-9
            # print([sigma_fit*1e-7, kappa_fit*1e-21, 0])
            sigma_si = sigma_fit * 1e-7
            kappa_si = kappa_fit * 1e-21
            gamma_si = 1000
            a = np.sqrt(sigma_si * sigma_si - 4 * kappa_si * gamma_si)
            b = (sigma_si - a) / (2 * kappa_si)
            c = (sigma_si + a) / (2 * kappa_si)
            confinement_fit, confinement_fit_cov = curve_fit(
                lambda n, a, b, c: self.gamma_fit_func_reparametrised(
                    n, radius * 1e-9, a, b, c
                ),
                q_range_fit,
                mps_si[q_range_fit[0] : q_range_fit[-1] + 1],
                [a, b, c],
                # [sigma_fit*1e-7, kappa_fit*1e-21, 0], #start with the basic fit results for easier fitting
                sigma=fit_sigmas_si,
                # bounds=[[0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]],
                # xtol = 1e-6, #should be close enough?
                maxfev=10000,
                # method="lm"
            )
            fit_r2, predicted, hessian = get_r2(
                lambda n, a, b, c: self.gamma_fit_func_reparametrised(
                    n, radius * 1e-9, a, b, c
                ),
                q_range_fit,
                mps_si[q_range_fit[0] : q_range_fit[-1] + 1],
                confinement_fit_cov,
                confinement_fit,
            )

            uncorrected = {
                "kappa": {"value": kappa_fit, "error": kappa_fit_err, "unit": 1e-21},
                "sigma": {"value": sigma_fit, "error": sigma_fit_err, "unit": 1e-7},
            }

            sigma_fit, kappa_fit, gamma_fit = self.gamma_fit_to_physical(
                *confinement_fit
            )
            # errs = np.sqrt(confinement_fit_cov.diagonal())
            sigma_fit *= 1e7
            sigma_fit_err = 0  # errs[0] * 1e7

            kappa_fit *= 1e21
            kappa_fit_err = 0  # errs[1] * 1e21

            alpha_beta_fit = confinement_fit
            gamma_res = {"value": gamma_fit, "error": 0, "unit": 1}
        elif self.fitting_method == "original":
            fit_r2 = rough_fit_r2

        if True:
            det = np.linalg.det(hessian)
            if det < 0:
                self.logger.warning(f"Hessian determinant negative!")
                fit_p = np.nan
            else:
                fit_p = 4 * np.pi / np.sqrt(det)  # S.3.10 of Rautu
        else:
            fit_p = np.nan

        # we want to get the number of "line crossings"
        point_above = predicted[0] < mps[self.min_mode]
        crossings = 0
        possible_crossings = 0
        for i in range(self.min_mode + 1, current_max_mode + 1):
            if (mps[i] < predicted[i - self.min_mode] and point_above) or (
                mps[i] > predicted[i - self.min_mode] and not point_above
            ):
                point_above = not point_above
                crossings += 1
            possible_crossings += 1

        # TODO: probably should only do the results here, left for now
        return {
            "kappa": {
                "value": kappa_fit,
                "error": kappa_fit_err,
                "unit": 1e-21,
            },
            "sigma": {
                "value": sigma_fit,
                "error": sigma_fit_err,
                "unit": 1e-7,
            },
            "uncorrected": uncorrected,
            "radius": {"value": radius},
            #'decay_times':{'value':taus.tolist() if taus is not None else None},
            # "r_fft": R_fft.tolist(),
            "mps": mps.tolist(),
            "mps_err": mps_err.tolist(),
            "mps_predicted_uncorrected": predicted_uncorrected.tolist(),
            "mps_predicted": (
                predicted_uncorrected.tolist()
                if self.fitting_method == "original"
                else predicted.tolist()
            ),
            "alpha_beta": alpha_beta_fit,
            "gamma": gamma_res,
            "r2": fit_r2,
            "fit_p": fit_p,
            "fit_crosses": crossings,
            "fit_cross_rate": crossings / possible_crossings,
            "detected_max_mode": current_max_mode if self.max_mode == "auto" else None,
        }

    def hessian_min(self, mps, mps_err, fit_fn, qs, fit_res):
        # TODO: does curve_fit not give us this?
        # calculate the Hessian matrix for Chi^2 of the fit
        # fit_fn is fitting function i.e. q,alpha,beta -> expected mps
        # S.3.1 to S.3.10 of Rautu
        mps = np.array(mps)
        mps_err = np.array(mps_err)

        def chi_sq(params):
            # params here is an (m,...) matrix for m fit parameters
            # we need to return (...) matrix
            # I don't think normal fit_fn vectorisation can handle that
            return np.sum(
                ((fit_fn(qs, params[0], params[1]) - mps[qs]) / mps_err[qs]) ** 2
            )

        def wrapped_func(x):
            x = np.asarray(x)
            output_shape = x.shape[1:]  # Shape of the output array
            x = x.reshape(x.shape[0], -1)  # Reshape to (m, product of other dims)
            result = np.empty(x.shape[1])  # Initialize output array

            for i in range(x.shape[1]):
                result[i] = chi_sq(x[:, i])  # Call func for each 'column'

            return result.reshape(output_shape)  # Reshape to original output shape

        return scipy.differentiate.hessian(wrapped_func, fit_res)

    def plot_spectrum(
        self,
        mps,
        mps_err,
        fit_results=None,
        alpha_beta=None,
        taus=None,
        delta=0,
        filename=None,
    ):
        # fig, ax = plt.subplots()
        fig, ax = (plt.gcf(), plt.gca()) if plt.get_fignums() else plt.subplots()

        # Determine max mode for plotting
        plot_max_mode = self.max_mode
        if plot_max_mode == "auto":
            if (
                fit_results
                and "detected_max_mode" in fit_results
                and fit_results["detected_max_mode"] is not None
            ):
                plot_max_mode = fit_results["detected_max_mode"]
            else:
                plot_max_mode = self.auto_max_mode_range[1]  # Fallback

        # Show a bit more than the fit range to see the tail
        # Ensure we show at least up to mode 25 as requested, or plot_max_mode + 10
        display_max = max(plot_max_mode + 10, 24)
        if display_max >= len(mps):
            print(f"Display max too large: {display_max}")

        q_range_fit = range(self.min_mode, display_max + 1)
        q_range_display = q_range_fit  # range(self.min_mode, display_max + 1)

        ax.errorbar(
            q_range_display,
            mps[q_range_display],
            yerr=mps_err[q_range_display],
            marker="o",
            linestyle="",
            capsize=3,
            # label="Data",
        )

        # Mark the cut-off
        if self.max_mode == "auto":
            ax.axvline(
                x=plot_max_mode,
                color="r",
                linestyle="--",
                label=f"Cut-off ({plot_max_mode})",
            )

        if fit_results:
            label = (
                r"$\sigma$ = {:.1f} $\pm$ {:.1f}".format(
                    fit_results["sigma"]["value"], fit_results["sigma"]["error"]
                )
                + r"$\times 10^{-7}$ N/m"
            )
            label = label + "\n"
            label = (
                label
                + r"$\kappa$ = {:.1f} $\pm$ {:.1f}".format(
                    fit_results["kappa"]["value"], fit_results["kappa"]["error"]
                )
                + r"$\times 10^{-21}$ N$\cdot$m"
            )
            if self.fitting_method == "original":
                fit_fn = original_fit_func
            elif self.fitting_method == "rautu":
                if (
                    taus is None
                    and self.exposure_time_ms is not None
                    and self.exposure_time_ms != 0
                ):
                    taus = np.array(fit_results["decay_times"]["value"])

                rw_correction_factors = None
                if (
                    self.rolling_window_correction
                    and self.rolling_window_correction_location == "fit"
                ):
                    rw_correction_factors, _ = (
                        self._get_rolling_window_correction_factor(taus)
                    )

                fit_fn = lambda q, alpha, beta: self.theory_ps(
                    q,
                    alpha,
                    beta,
                    (
                        (taus[:, [0, 2]].mean(axis=1) / self.exposure_time_ms) ** -1
                        if taus is not None
                        and self.exposure_time_ms is not None
                        and self.exposure_correction_method
                        in ("rautu", "rautu_const", "original")
                        and self.exposure_time_ms > 0
                        and self.exposure_correction_location == "fit"
                        else None
                    ),
                    delta,
                    self.mode_sum_range,  # Use mode_sum_range for q_sum
                    rw_correction_factors,
                    self.exposure_correction_method,
                    self.exposure_correction_location,
                    max_mode=plot_max_mode,  # Pass resolved max_mode
                )
            elif self.fitting_method == "confinement":
                label += (
                    "\n"
                    + r"$\gamma = "
                    + f"{fit_results['gamma']['value']/1000:.0f}"
                    + r"\pm "
                    + f"{fit_results['gamma']['error']/1000:.0f}$ kJ/m^4"
                )
                fit_fn = lambda q, alpha, beta: 1e18 * self.gamma_fit_func(
                    q,
                    fit_results["radius"]["value"] * 1e-9,
                    fit_results["sigma"]["value"] * fit_results["sigma"]["unit"],
                    fit_results["kappa"]["value"] * fit_results["kappa"]["unit"],
                    fit_results["gamma"]["value"] * fit_results["gamma"]["unit"],
                )
            elif self.fitting_method == "confinement_new":
                label += (
                    "\n"
                    + r"$\gamma = "
                    + f"{fit_results['gamma']['value']/1000:.0f}"
                    + r"\pm "
                    + f"{fit_results['gamma']['error']/1000:.0f}$ kJ/m^4"
                )
                fit_fn = lambda q, a, b, c: 1e18 * self.gamma_fit_func_reparametrised(
                    q, fit_results["radius"]["value"] * 1e-9, a, b, c
                )
            else:
                raise NotImplementedError(
                    f"Unknown fitting method {self.fitting_method}"
                )
            label += "\n" + r"$r^2 = " + f"{fit_results['r2']:.2f}$"

            ax.plot(
                q_range_fit,
                [fit_fn(qq, *alpha_beta) for qq in q_range_fit],
                lw=4,
                label=label,
            )

        ax.set_xlabel("Mode number")
        ax.set_ylabel(r"Power (nm$^2$)")
        ax.legend()

        ax.set_xscale("log")
        ax.set_yscale("log")

        if filename:
            fig.savefig(filename)
        return fig, ax

    def print_info(self, fit_results, delta):
        print("Fit parameters: \n--------------\n")
        print(unicodedata.lookup("GREEK CAPITAL LETTER DELTA") + f" = {delta}")
        print(f"Mean spontaneous curvature: H = {self.mean_spontaneous_curvature}")
        if self.exposure_time_ms is not None:
            print(
                "Exposure time: "
                + unicodedata.lookup("GREEK SMALL LETTER TAU")
                + " = {:.1f} ms".format(self.exposure_time_ms)
            )
            # print(
            #    "Sum of internal and external viscosities: "
            #    + unicodedata.lookup("GREEK SMALL LETTER ETA")
            #    + " = {:.1f} mPa*s".format(fit_param["tau_p"]["eta"] * 1e6)
            # )
        else:
            print(
                "Exposure time: "
                + unicodedata.lookup("GREEK SMALL LETTER TAU")
                + " = 0 ms (instantaneous)"
            )

        print("\nFit results: \n-----------\n")
        print(
            f"Bending modulus: {unicodedata.lookup('GREEK SMALL LETTER KAPPA')} = {fit_results['kappa']['value'] / 10.0} \u00b1 {fit_results['kappa']['error'] / 10.0} x 10^-20 J"
        )
        print(
            f"\t = {fit_results['kappa']['value'] / self.kB_T} \u00b1 {fit_results['kappa']['error'] / self.kB_T} kT"
        )
        print(
            f"Tension: {unicodedata.lookup('GREEK SMALL LETTER SIGMA')} = {fit_results['sigma']['value']} \u00b1 {fit_results['sigma']['error']} x 10^-7 N/m \n\n"
        )


def _calculate_partials_theoretical_tau_analytical(
    n, eta_in, eta_out, radius, kappa, sigma
):  #
    """
    Calculates partial derivatives of theoretical_tau with respect to eta_in, kappa, and sigma analytically.
    This is for manual error propagation.
    """
    R = radius
    R2 = R * R
    R3 = R * R * R

    # Common terms from theoretical_tau
    f1_n = (n + 2.0) * (2.0 * n - 1.0) / (n + 1.0)
    f2_n = eta_out * (n - 1.0) * (2.0 * n + 3.0) / n
    f3_n = (n - 1.0) * (n + 2.0)
    f4_n = n * (n + 1.0)

    numerator_base = R3 * (eta_in * f1_n + f2_n)
    denominator_base = kappa * f3_n * f4_n + sigma * f3_n * R2

    # Partial derivative with respect to eta_in
    d_tau_d_eta_in = (R3 * f1_n) / denominator_base

    # Partial derivative with respect to kappa
    d_tau_d_kappa = -numerator_base * (f3_n * f4_n) / (denominator_base**2)

    # Partial derivative with respect to sigma
    d_tau_d_sigma = -numerator_base * (f3_n * R2) / (denominator_base**2)

    return d_tau_d_eta_in, d_tau_d_kappa, d_tau_d_sigma


def _calculate_partials_theoretical_tau_yoon(n, eta_in, eta_out, radius, kappa, sigma):
    """
    Calculates partial derivatives of theoretical_tau_yoon with respect to eta_in, kappa, and sigma.
    """
    q = n / radius

    # Common terms
    # tau = N / D
    # N = 2 * (eta_in + eta_out) * q
    # D = sigma * q^2 + kappa * q^4

    N = 2 * (eta_in + eta_out) * q
    D = sigma * q**2 + kappa * q**4

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = N / D

        # d_tau_d_eta_in = 2*q / D
        d_tau_d_eta_in = 2 * q / D

        # d_tau_d_kappa = -tau * q^4 / D
        d_tau_d_kappa = -tau * q**4 / D

        # d_tau_d_sigma = -tau * q^2 / D
        d_tau_d_sigma = -tau * q**2 / D

    return d_tau_d_eta_in, d_tau_d_kappa, d_tau_d_sigma


def _mean_tau_error(taus):
    """Calculates the error on the mean of the real and imaginary parts of tau."""
    err_real = taus[:, 1]
    err_imag = taus[:, 3]
    # Variance of mean is (var_real + var_imag) / 4
    # Error of mean is sqrt(var_mean)
    return np.sqrt(err_real**2 + err_imag**2) / 2.0


def tau_correction_brown_reverse(tau):
    """
    https://journals.aps.org/pre/pdf/10.1103/PhysRevE.84.021930
    """
    return np.power(tau, 0.88) * 0.57  # converts tau_0 to tau'


def tau_correction_brown(tau):
    """
    https://journals.aps.org/pre/pdf/10.1103/PhysRevE.84.021930
    """
    return np.power(tau / 0.57, 1 / 0.88)  # converts tau' to tau_0


def theoretical_tau(
    n: Union[int, np.ndarray],
    eta_in: float,
    eta_out: float,
    radius: float,
    kappa: float,
    sigma: float,
) -> Union[float, np.ndarray]:
    """Calculates the theoretical decay time [s], tau_n, for mode n from Rautu et al, Eq. 2
    NOTE that this expression assumes zero spontaneous curvature
    TODO: workout where called]
    TODO: this doesn't seem to work at all

    Args:
        n (int): mode number for which the decay time should be calculated
        eta_in (float): viscosity of fluid inside cell [Pa s]
        eta_out (float): viscosity of fluid outside cell [Pa s]
        radius (float): radius of the cell [m]
        kappa (float): bending modulus of the membrane [J]
        sigma (float): membrane tension [J / m**2]

    Returns:
        float: _description_
    """
    sigma_bar = sigma * (radius**2) / kappa  # - 2 * H_0 * R + 2 * H_0**2 * R**2
    return (
        ((radius**3) / kappa)
        * (
            (eta_in * (n + 2.0) * (2.0 * n - 1.0) / (n + 1.0))
            + (eta_out * (n - 1.0) * (2.0 * n + 3.0) / (1.0 * n))
        )
        / ((n - 1.0) * (n + 2.0) * (n * (n + 1.0) + sigma_bar))
    )


def theoretical_tau_yoon(
    n: Union[int, np.ndarray],
    eta_in: float,
    eta_out: float,
    radius: float,
    kappa: float,
    sigma: float,
) -> Union[float, np.ndarray]:
    """Calculates the theoretical decay time [s], tau_n, for mode n from Yoon et al, Eq. 2
    NOTE that this expression assumes zero spontaneous curvature
    TODO: workout where called]
    TODO: this doesn't seem to work at all

    Args:
        n (int): mode number for which the decay time should be calculated
        eta_in (float): viscosity of fluid inside cell [Pa s]
        eta_out (float): viscosity of fluid outside cell [Pa s]
        radius (float): radius of the cell [m]
        kappa (float): bending modulus of the membrane [J]
        sigma (float): membrane tension [J / m**2]

    Returns:
        float: _description_
    """
    q = n / radius
    gamma = 0
    eta_membrane = 0  # neglible per Yoon et al
    return (
        2
        * (eta_membrane / radius**2 + eta_in + eta_out)
        * q
        / (2 * gamma + q**2 * sigma + kappa * q**4)
    )


# original PS fit
def ps_func(x):
    return (x**-1) - np.sqrt(1 + (x**2)) ** -1


# TODO: exact differences between these two
def original_fit_func(x, a, b):
    return a * ps_func(x / b)


@jit(nopython=True)
def butterworth_psd_transfer(u, val):
    h_mag_sq = (u**8) / (1 + u**8)
    return 1.0 / (val**2 + u**2) * h_mag_sq**2  # Squared again!


def butterworth_filtfilt_correction(ratio_tau_T):
    """
    Calculates K = Total_Power / Passed_Power
    ratio_tau_T = tau / T_cutoff = tau * f_cutoff
    """

    # Normalized parameter: alpha/omega_c
    # omega_c = 2*pi / T
    # alpha = 1 / tau
    # param = alpha/omega_c = T / (2*pi*tau) = 1 / (2*pi*ratio)
    val = 1.0 / (2 * np.pi * ratio_tau_T)

    # Integrate
    # 1. Total Power (0 to infinity)
    # Analytical integral of 1/(a^2+x^2) is (1/a)*arctan(x/a) -> pi/(2a)
    total_power = np.pi / (2 * val)

    # 2. Passed Power (Numerical)
    # Using integrate.quad instead of quad to ensure reference usage works if import was aliased
    passed_power, _ = integrate.quad(
        butterworth_psd_transfer, 0, np.inf, args=(val,), limit=1000
    )

    return total_power / passed_power
