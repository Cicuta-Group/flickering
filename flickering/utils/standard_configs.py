from flickering.tracking.correlation_tracker import CorrelationContourTracker as CCT
from flickering.analysis.fitter import ContourFitter
from multiprocessing import cpu_count


def default_fitter():
    cf = ContourFitter()
    cf.set_temperature(37 + 273)
    cf.fitting_method = "rautu"
    # delta = 0.1
    cf.depth_of_focus_um = 0.5
    cf.vertical_radius_um = 4
    cf.angular_noise = 0
    cf.radial_noise = 0
    cf.delay_between_frames_ms = 1000 / 660
    cf.exposure_time_ms = 0.8  # TODO: read from metada
    # print(cf.exposure_time_ms)
    # print(cf.delay_between_frames_ms)
    cf.decay_time_brown_correction = False
    cf.decay_times_model = "linear_exponential"
    cf.decay_time_use_fit = True
    cf.decay_time_fit_max = 20
    cf.remove_outliers = False
    cf.radius_correction_nm = 0
    cf.min_mode = 6
    cf.max_mode = "auto"
    cf.nm_per_px = 98.6
    cf.sub_radius = "butter"
    cf.fit_limits = None
    cf.sub_radius_rolling_mean_window = 66
    cf.rolling_window_correction = True
    cf.decay_time_filter_threshold_s = "static_shape"

    cf.fit_viscosity = True
    # --- Viscosity fit options (fit internal viscosity) ---
    cf.viscosity_eta_out = (
        0.733e-3  # Pa.s, default value for external viscosity (RPMI, 37C)
    )
    cf.fit_viscosity_tau_range = (
        1.5,
        10,
    )  # (min_tau, max_tau) in ms, alternative to mode range
    cf.fit_viscosity_lock_sigma_kappa = True
    cf.fit_viscosity_modes = (6, 18)
    cf.fit_viscosity_method = "yoon"
    cf.exposure_correction_method = "original"
    cf.exposure_correction_location = "spectrum"

    # --- Additional parameters explicitly defined to match default configuration ---
    cf.auto_max_mode_range = (17, 26)
    cf.auto_max_mode_threshold = 0.05
    cf.mode_sum_range = 30
    cf.fallback_sigma = 5.0
    cf.fallback_kappa = 200.0
    cf.rough_fit_r2_threshold = 0.8
    cf.sigma_limits = (0.1, 100.0)
    cf.kappa_limits = (10.0, 2000.0)
    cf.sub_radius_filter = None
    cf.mean_spontaneous_curvature = 0
    cf.radius_uncertainty = 50
    cf.thinning_factor = None
    cf.maximum_outlier_fluctuation = 0.03
    cf.angular_noise_estimate = None
    cf.radial_noise_estimate = None
    cf.fit_weights_method = "mps_error"
    cf.noise_floor_above = 40
    cf.noise_floor_below = 50
    cf.autocorrelation_invalid_method = "average"
    cf.autocorrelation_mode = "separate"
    cf.decay_time_fit_min = 0
    cf.decay_time_brown_correction_not_in_filter = True
    cf.fit_mode = "linear"
    cf.extra_mps_error = 0
    cf.rolling_window_correction_location = "spectrum"
    cf.rolling_window_correction_timescaling = 0.248
    cf.rolling_window_correction_exponent = 1.273
    cf.radial_variance_calculation = False
    cf.save_acf_frames = 100
    cf.propagate_tau_error = True

    return cf


def default_tracker():
    cct = CCT()
    cct.mask_type = CCT.MASK_TYPE_GENERATE
    cct.use_centers_mean = 1
    cct.threads = max(cpu_count() - 1, 1)
    cct.debug = False
    cct.ignore_center_rad = 10
    cct.max_shift = 10
    cct.interpolation_method = CCT.METHOD_LINEAR
    cct.refine_interpolation_method = CCT.METHOD_CUBIC
    cct.use_fixed_mask = True  # True
    cct.mask_refinement = True  # helps
    cct.refine_correlation = True  # unclear which is better
    cct.refine_center_max_iterations = 50
    cct.mask_width = 30  # TODO? - how does this work for cells which have a depression in the middle which extends to this region?
    cct.correlate_width = 60
    cct.subtract_means = True
    cct.parabolic_fit_width = 5
    cct.laplace_th = 0.0014
    cct.refine_center_tolerance = 0.05  # lowered from 0.1

    # these work better without the 1.5x zoom
    cct.hough_min_r = 40 // 2
    cct.hough_param_2 = 32
    cct.save_mode = "R"

    return cct


def vesicle_tracker():
    cct = CCT()
    cct.hough_min_r = 25
    cct.hough_param_2 = 16
    cct.hough_dp = 1
    cct.hough_max_r = 120
    cct.hough_param_1 = 30

    cct.mask_type = CCT.MASK_TYPE_GENERATE
    cct.use_centers_mean = None
    cct.use_radius_initial = True
    cct.debug = True
    cct.ignore_center_rad = 20
    # cct.debug = True
    # cct.ignore_center_rad = 0
    cct.max_shift = 2
    cct.interpolation_method = CCT.METHOD_LINEAR
    cct.use_fixed_mask = True  # True
    cct.mask_refinement = True  # True
    cct.refine_correlation = True  # True
    cct.refine_center_max_iterations = 10
    cct.refine_center_tolerance = 0.01
    cct.mask_width = 25  # TODO?
    cct.correlate_width = 120
    cct.subtract_means = True
    cct.parabolic_fit_width = 5

    cct.save_mode = "R"
    return cct
