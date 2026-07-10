# TODO: clean up imports
# TODO: the position calculation is unnecessarily repeated -> we should convert once and then keep converted values
# TODO: make cpu variation?
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from numba import njit
from multiprocessing.pool import ThreadPool  # as Pool
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp2d, interp1d, RegularGridInterpolator
from scipy.signal import correlate, correlation_lags
import cv2
import matplotlib.pyplot as plt
from time import time
# from contour_analysis.contour_fitter import  # legacy, removed *

# from contour_analyzer.read_cpp_contour import *
from glob import glob
from multiprocessing import Pool
import re
import json
import matplotlib.pyplot as plt
import logging
# from autoimager.correlation_contour import  # legacy module removed *
from os.path import isfile
from copy import deepcopy
import logging
from datetime import datetime
from flickering.analysis.fitter import ContourFitter as CF
from flickering.tracking.contour_io import ContourIO
from tqdm.auto import tqdm

use_cuda = True
try:
    import cupy as cp
except ImportError:
    import numpy as cp

    use_cuda = False


def set_device(device):
    global cp, use_cuda
    if device == "cpu":
        import numpy as np

        cp = np
        use_cuda = False
    else:
        import cupy as cyp

        cp = cyp
        use_cuda = True


from numba import njit, prange


@njit(parallel=True)
def positions_from_modes_parallel(surviving_modes, x_positions):
    P = x_positions.shape[0]
    M = surviving_modes.shape[0]
    results = np.zeros(P)
    for i in prange(P):
        s = 0.0
        x = x_positions[i]
        for j in range(M):
            # surviving_modes columns: [0: index, 1: qs, 2: amplitudes, 3: phase, 4: step]
            s += surviving_modes[j, 2] * np.sin(
                surviving_modes[j, 3] + x * surviving_modes[j, 1]
            )
        results[i] = s
    return results


@njit
def positions_from_modes(surviving_modes, expanded_xpos):
    #    mode_contributions = loop_modes[:,2]*np.sin(loop_modes[:,3] + expanded_xpos * loop_modes[:,1])
    #    positions = mode_contributions.sum(axis=1)
    mode_contributions = surviving_modes[:, 2] * np.sin(
        surviving_modes[:, 3] + expanded_xpos * surviving_modes[:, 1]
    )
    return mode_contributions.sum(axis=1)


# TODO: remove extra copies
def cupy_positions_from_modes_with_copy(surviving_modes, expanded_xpos):
    #    mode_contributions = loop_modes[:,2]*np.sin(loop_modes[:,3] + expanded_xpos * loop_modes[:,1])
    #    positions = mode_contributions.sum(axis=1)
    surviving_modes = cp.asarray(surviving_modes)
    expanded_xpos = cp.asarray(expanded_xpos)
    mode_contributions = surviving_modes[:, 2] * cp.sin(
        surviving_modes[:, 3] + expanded_xpos * surviving_modes[:, 1]
    )
    return (
        cp.asnumpy(mode_contributions.sum(axis=1))
        if use_cuda
        else mode_contributions.sum(axis=1)
    )


def cupy_positions_from_modes_no_copy(surviving_modes, expanded_xpos):
    #    mode_contributions = loop_modes[:,2]*np.sin(loop_modes[:,3] + expanded_xpos * loop_modes[:,1])
    #    positions = mode_contributions.sum(axis=1)
    mode_contributions = surviving_modes[:, 2] * cp.sin(
        surviving_modes[:, 3] + expanded_xpos * surviving_modes[:, 1]
    )
    return mode_contributions.sum(axis=1)


class StringSim:
    def __init__(self):
        self.points = 360
        self.time_steps = 200000  # video time steps = frames
        self.dt = 5e-5  # this is the simulation dt, error ~ dt/tau due to the discretisation of mode generation here. mode 30 has tau ~1ms
        self.dt_video = 1e-3
        # sigma = 10000
        self.sigma = 5e-7
        self.kappa = 200e-21
        self.R = 4e-6
        self.kT = 4e-21
        self.mode_discard = None  # automatic, 1e-11 is very conservative
        self.max_mode = 50
        self.mean_dts = np.repeat(0.1, self.max_mode + 1)
        self.position_calc_threads = 1
        self.max_position_calc_lag = 256  # not guaranteed
        self.prune_survivors_every = 2
        self.max_survivors = 500000
        self.mode_discard_cap = None  # 1e-13 #never go below this
        self.shutter_time = 0
        self.warmup_steps = 0
        self.bench_mode = False

    def prep_sim(self):
        self.L = 2 * np.pi * self.R
        self.x_positions = np.linspace(0, self.L, self.points)
        # mode numbers
        self.mode_nums = np.arange(self.max_mode) + 1
        self.current_modes = np.zeros((1, 5))
        self.step_results = []
        self.expanded_xpos = cp.asarray(np.expand_dims(self.x_positions, axis=1))
        self.mode_counts = []

        self.video_mod = int(self.dt_video / self.dt)
        # print(f"Recording state every {self.video_mod} iterations")
        self.surviving_modes = cp.zeros((1, 5))
        self.loop_modes = cp.zeros((self.video_mod * len(self.mode_nums), 5))

        # self.position_pool = ThreadPool(self.position_calc_threads)
        self.position_results = []
        self.mean_dts = cp.asarray(self.mean_dts)
        if self.mode_discard is None:
            qs = self.mode_nums * 2 * np.pi / self.L
            q = qs[-1]
            amplitudes_prefactors = (
                np.sqrt(self.power_spectrum(qs)) * self.spectrum_scaling()
            )
            want_included = (
                self.power_spectrum(qs) / amplitudes_prefactors**2
            )  # number of modes
            self.mode_discard = (
                np.sqrt(self.power_spectrum(qs)) / want_included
            ).min() / 10  # this is probably to small
            if self.mode_discard_cap is not None:
                self.mode_discard = max(self.mode_discard_cap, self.mode_discard)
            # print(np.sqrt(self.power_spectrum(qs))/want_included)
            print(f"Mode discard auto set to: {self.mode_discard}")
            # we want to be safely (10x?) below the power spectrum contribution from a single generated mode

    def spectrum_scaling(self):
        return (
            np.sqrt(2 / self.decay_times(self.mode_nums, True)) * np.sqrt(self.dt) * 2
        )  # (sum to mean conversion)

    def power_spectrum(self, q):
        # return kT/q/q
        return (
            self.kT
            / self.L
            / (2 * self.sigma)
            * (1 / q - 1 / np.sqrt(self.sigma / self.kappa + q * q))
        )  # from evans 2008, pecroux
        # return kT/q + 14900*kT/(q*q*q)
        # return kT/(self.sigma*q) + kT/(self.kappa*q*q*q*q)

    def decay_times(self, mode_indices, force_cpu=False):
        # return 1e-4
        # return np.repeat(1e-7, len(mode_indices))
        # print(mode_indices)
        if force_cpu:
            return (
                cp.asnumpy(self.mean_dts)[mode_indices.astype(int)]
                if use_cuda
                else self.mean_dts[mode_indices.astype(int)]
            )
        return self.mean_dts[mode_indices.astype(int)]

    def simulate(self):
        full_results = cp.zeros((self.time_steps, self.points))
        self.prep_sim()
        total_steps = self.warmup_steps + self.time_steps
        self.pbar = tqdm(total=self.video_mod * total_steps, disable=self.bench_mode)

        if self.bench_mode and self.warmup_steps > 0:
            print("BENCH_WARMUP_START", flush=True)

        # precompute some things
        qs = self.mode_nums * 2 * np.pi / self.L
        amplitudes_prefactors = (
            np.sqrt(self.power_spectrum(qs)) * self.spectrum_scaling()
        )

        amplitudes_prefactors = cp.asarray(amplitudes_prefactors)
        qs = cp.asarray(qs)

        warmup_done_printed = False

        for step in range(self.video_mod * total_steps):
            # add excitations with amplitudes scaled to match expected spectrum
            # modes average contribution to power spectrum is modulated by it's decay time
            # mean contribution = |A|^2*exp(-n*(time_to_next_frame/tau))*sum(exp(-n*dt_video/tau)) = |A|^2 integrate exp(-2t/tau) = |A|^2 tau/2 = power_spectrum(q) => A \propto sqrt(2/tau)
            # second term gives exp(dt_video/tau)/(exp(dt_video/tau)-1)
            amplitudes = amplitudes_prefactors * cp.random.normal(size=len(qs))
            phase = np.pi * cp.random.rand(len(qs))

            # new_modes = np.swapaxes(np.vstack((mode_nums, qs, amplitudes, phase)), 0,1)
            self.loop_modes[
                (step % self.video_mod)
                * len(self.mode_nums) : (step % self.video_mod + 1)
                * len(self.mode_nums)
            ] = cp.swapaxes(
                cp.vstack(
                    (
                        self.mode_nums,
                        qs,
                        amplitudes,
                        phase,
                        cp.full(len(self.mode_nums), step),
                    )
                ),
                0,
                1,
            )

            # current_modes = np.concatenate((current_modes[np.abs(current_modes[:,2]) > mode_discard],new_modes))
            if (step + 1) % self.video_mod == 0:
                current_frame = (step + 1) // self.video_mod
                if (
                    self.bench_mode
                    and not warmup_done_printed
                    and current_frame > self.warmup_steps
                ):
                    print("BENCH_MEASURE_START", flush=True)
                    self.bench_measure_start = time()
                    warmup_done_printed = True

                # age modes
                self.loop_modes[:, 2] *= cp.exp(
                    -(step - self.loop_modes[:, 4])
                    * self.dt
                    / self.decay_times(self.loop_modes[:, 0])
                )
                # these have been decayed to the last frame state already, decay by time between frames
                # print(type(self.surviving_modes[:,0]))
                self.surviving_modes[:, 2] *= cp.exp(
                    -self.dt_video / self.decay_times(self.surviving_modes[:, 0])
                )
                self.pbar.update(self.video_mod)
                # mode_contributions = self.loop_modes[:,2]*np.sin(self.loop_modes[:,3] + self.expanded_xpos * self.loop_modes[:,1])
                # positions = mode_contributions.sum(axis=1)
                # positions = positions_from_modes(self.loop_modes, self.surviving_modes, self.exp

                # filtered_survivors = self.surviving_modes[abs_survivors_indices]
                # cp.delete(self.surviving_modes,
                filtered_loops = self.loop_modes[
                    cp.abs(self.loop_modes[:, 2]) > self.mode_discard
                ]
                self.surviving_modes = cp.concatenate(
                    (self.surviving_modes, filtered_loops)
                )

                if self.shutter_time > 0:
                    t0 = cp.maximum(
                        self.dt * (self.surviving_modes[:, 4] - step)
                        + self.shutter_time,
                        0,
                    )
                    taus = self.decay_times(self.surviving_modes[:, 0])
                    prefactors = (
                        (cp.exp((self.shutter_time - t0) / taus) - 1)
                        * taus
                        / self.shutter_time
                    )  # TODO: do we need to scale by tau?
                    # if step//self.video_mod == 10:
                    # return t0,prefactors
                    # break
                    corrected_modes = self.surviving_modes.copy()
                    corrected_modes[:, 2] *= prefactors
                    if current_frame > self.warmup_steps:
                        if use_cuda:
                            full_results[current_frame - self.warmup_steps - 1, :] = (
                                cupy_positions_from_modes_no_copy(
                                    corrected_modes, self.expanded_xpos
                                )
                            )
                        else:
                            full_results[current_frame - self.warmup_steps - 1, :] = (
                                positions_from_modes_parallel(
                                    corrected_modes, self.x_positions
                                )
                            )
                else:
                    if current_frame > self.warmup_steps:
                        if use_cuda:
                            full_results[current_frame - self.warmup_steps - 1, :] = (
                                cupy_positions_from_modes_no_copy(
                                    self.surviving_modes, self.expanded_xpos
                                )
                            )
                        else:
                            full_results[current_frame - self.warmup_steps - 1, :] = (
                                positions_from_modes_parallel(
                                    self.surviving_modes, self.x_positions
                                )
                            )
                # mode_contributions = self.surviving_modes[:,2]*np.sin(self.surviving_modes[:,3] + self.expanded_xpos * self.surviving_modes[:,1])
                # if self.position_calc_threads > 0:
                #    self.position_results.append(self.position_pool.apply_async(cupy_positions_from_modes_with_copy, (self.surviving_modes, self.expanded_xpos)))
                #
                #    if len(self.position_results) > self.max_position_calc_lag:
                #        if not self.position_results[-self.max_position_calc_lag].ready():
                #            print("Waiting for position calculation to catch up")
                #            self.position_results[-self.max_position_calc_lag].wait(0.5)
                # else:
                #    self.step_results.append(cupy_positions_from_modes_with_copy(self.surviving_modes, self.expanded_xpos))

                self.mode_counts.append(len(self.surviving_modes))
            if step % (self.video_mod * self.prune_survivors_every) == 0:
                self.prune_survivors()
        self.step_results = cp.asnumpy(full_results) if use_cuda else full_results
        if self.bench_mode:
            duration = time() - self.bench_measure_start
            print(f"BENCH_MEASURE_END {duration:.6f}", flush=True)
        full_results = None
        if use_cuda:
            cp.get_default_memory_pool().free_all_blocks()
        self.pbar.close()
        #        if self.position_calc_threads > 0:
        #            for r in self.position_results:
        #                self.step_results.append(r.get())
        # self.position_pool.close()

        return self.step_results, self.mode_counts

    def prune_survivors(self):
        abs_survivors_indices = cp.abs(self.surviving_modes[:, 2]) < self.mode_discard
        self.surviving_modes = cp.delete(
            self.surviving_modes, abs_survivors_indices, axis=0
        )


class StringSimFixedDecays(StringSim):
    def decay_times(self, qs, force_cpu=False):
        dcs = cp.repeat(cp.array(self.decay_time_fixed), len(qs))
        if force_cpu:
            return cp.asnumpy(dcs) if use_cuda else dcs
        return dcs


class FlatPsTest(StringSim):
    def power_spectrum(self, q):
        # return kT/q/q
        # using 1e-16 to stay within the usual scaling
        return np.repeat(1e-17, len(q))


class KickedSim(FlatPsTest):
    def __init__(self):
        super().__init__()
        self.kick_time = 1e-1
        self.kick_ampl = 1.6e-9  # nm in spectrum amplitude (not really)

    def spectrum_scaling_kick(self):
        return (
            np.sqrt(2 / (2 * self.kick_time + self.decay_times(self.mode_nums, True)))
            * 2
        )  # (sum to mean conversion)

    def prep_sim(self):
        super().prep_sim()
        self.kick_steps = self.kick_time / self.dt
        self.surviving_modes = cp.zeros((1, 5))
        self.kick_surviving_modes = cp.zeros((1, 5))
        self.kick_loop_modes = cp.zeros((self.video_mod * len(self.mode_nums), 5))
        relevant_area = 2 * np.pi * self.R * 2e-6
        self.kicks_per_second = (
            0.1 * 500e12 * relevant_area
        )  # 500 kickers per/um (Rodriguez-Garcia, 2015)

    def simulate(self):
        full_results = cp.zeros((self.time_steps, self.points))
        self.prep_sim()
        total_steps = self.warmup_steps + self.time_steps
        self.pbar = tqdm(total=self.video_mod * total_steps)

        if self.bench_mode and self.warmup_steps > 0:
            print("BENCH_WARMUP_START", flush=True)

        # precompute some things
        qs = self.mode_nums * 2 * np.pi / self.L
        amplitudes_prefactors = (
            np.sqrt(self.power_spectrum(qs)) * self.spectrum_scaling()
        )
        amplitudes_prefactors = cp.asarray(amplitudes_prefactors)

        kick_freq = self.kicks_per_second * self.dt / len(self.mode_nums)
        kick_amplitudes_prefactors = cp.asarray(
            self.kick_ampl * self.spectrum_scaling_kick()
        )
        print(kick_amplitudes_prefactors, flush=True)
        qs = cp.asarray(qs)
        warmup_done_printed = False
        for step in range(self.video_mod * total_steps):
            # add excitations with amplitudes scaled to match expected spectrum
            # modes average contribution to power spectrum is modulated by it's decay time
            # mean contribution = |A|^2*exp(-n*(time_to_next_frame/tau))*sum(exp(-n*dt_video/tau)) = |A|^2 integrate exp(-2t/tau) = |A|^2 tau/2 = power_spectrum(q) => A \propto sqrt(2/tau)
            # second term gives exp(dt_video/tau)/(exp(dt_video/tau)-1)
            amplitudes = amplitudes_prefactors * cp.random.normal(size=len(qs))
            phase = np.pi * cp.random.rand(len(qs))

            # new_modes = np.swapaxes(np.vstack((mode_nums, qs, amplitudes, phase)), 0,1)
            self.loop_modes[
                (step % self.video_mod)
                * len(self.mode_nums) : (step % self.video_mod + 1)
                * len(self.mode_nums)
            ] = cp.swapaxes(
                cp.vstack(
                    (
                        self.mode_nums,
                        qs,
                        amplitudes,
                        phase,
                        cp.full(len(self.mode_nums), step),
                    )
                ),
                0,
                1,
            )

            # kick modes
            amplitudes_kick = kick_amplitudes_prefactors * (
                cp.random.randint(0, 2, size=len(qs)) * 2 - 1
            )
            phase_kick = np.pi * cp.random.rand(len(qs))
            self.kick_loop_modes[
                (step % self.video_mod)
                * len(self.mode_nums) : (step % self.video_mod + 1)
                * len(self.mode_nums)
            ] = cp.swapaxes(
                cp.vstack(
                    (
                        self.mode_nums,
                        qs,
                        amplitudes_kick,
                        phase_kick,
                        cp.full(len(self.mode_nums), step),
                    )
                ),
                0,
                1,
            )

            select_kicks = (
                cp.random.uniform(size=(len(self.mode_nums) * (self.video_mod)))
                > kick_freq
            )
            self.kick_loop_modes[select_kicks, 2] = 0

            # current_modes = np.concatenate((current_modes[np.abs(current_modes[:,2]) > mode_discard],new_modes))
            if step % self.video_mod == 0:
                current_frame = step // self.video_mod
                if (
                    self.bench_mode
                    and not warmup_done_printed
                    and current_frame >= self.warmup_steps
                ):
                    print("BENCH_MEASURE_START", flush=True)
                    self.bench_measure_start = time()
                    warmup_done_printed = True

                # age modes
                self.loop_modes[:, 2] *= cp.exp(
                    -(step - self.loop_modes[:, 4])
                    * self.dt
                    / self.decay_times(self.loop_modes[:, 0])
                )

                self.kick_loop_modes[:, 2] *= cp.exp(
                    -cp.heaviside(step - self.loop_modes[:, 4] - self.kick_steps, 0)
                    * (step - self.loop_modes[:, 4] - self.kick_steps)
                    * self.dt
                    / self.decay_times(self.loop_modes[:, 0])
                )

                # these have been decayed to the last frame state already, decay by time between frames
                # print(type(self.surviving_modes[:,0]))
                self.surviving_modes[:, 2] *= cp.exp(
                    -self.dt_video / self.decay_times(self.surviving_modes[:, 0])
                )

                # this is a little tricky, these have been decayed to last frame state but the new modes should not be decayed, old ones
                self.kick_surviving_modes[:, 2] *= cp.exp(
                    -(  # if decaying, decay start time (start+kicktime)-decay up to last timepoint
                        cp.heaviside(
                            step - self.kick_surviving_modes[:, 4] - self.kick_steps, 0
                        )
                        * (step - self.kick_surviving_modes[:, 4] - self.kick_steps)
                        - cp.heaviside(
                            step
                            - self.video_mod
                            - self.kick_surviving_modes[:, 4]
                            - self.kick_steps,
                            0,
                        )
                        * (
                            step
                            - self.video_mod
                            - self.kick_surviving_modes[:, 4]
                            - self.kick_steps
                        )
                    )
                    * self.dt
                    / self.decay_times(self.kick_surviving_modes[:, 0])
                )

                self.pbar.update(self.video_mod)
                # mode_contributions = self.loop_modes[:,2]*np.sin(self.loop_modes[:,3] + self.expanded_xpos * self.loop_modes[:,1])
                # positions = mode_contributions.sum(axis=1)
                # positions = positions_from_modes(self.loop_modes, self.surviving_modes, self.exp

                # filtered_survivors = self.surviving_modes[abs_survivors_indices]
                # cp.delete(self.surviving_modes,
                filtered_loops = self.loop_modes[
                    cp.abs(self.loop_modes[:, 2]) > self.mode_discard
                ]
                filtered_kicks = self.kick_loop_modes[
                    cp.abs(self.kick_loop_modes[:, 2]) > self.mode_discard
                ]

                self.surviving_modes = cp.concatenate(
                    (self.surviving_modes, filtered_loops)
                )
                self.kick_surviving_modes = cp.concatenate(
                    (self.kick_surviving_modes, filtered_kicks)
                )

                if current_frame >= self.warmup_steps:
                    if use_cuda:
                        full_results[current_frame - self.warmup_steps, :] = (
                            cupy_positions_from_modes_no_copy(
                                cp.concatenate(
                                    (self.surviving_modes, self.kick_surviving_modes)
                                ),
                                self.expanded_xpos,
                            )
                        )
                    else:
                        full_results[current_frame - self.warmup_steps, :] = (
                            positions_from_modes_parallel(
                                cp.concatenate(
                                    (self.surviving_modes, self.kick_surviving_modes)
                                ),
                                self.x_positions,
                            )
                        )
                # mode_contributions = self.surviving_modes[:,2]*np.sin(self.surviving_modes[:,3] + self.expanded_xpos * self.surviving_modes[:,1])
                # if self.position_calc_threads > 0:
                #    self.position_results.append(self.position_pool.apply_async(cupy_positions_from_modes_with_copy, (self.surviving_modes, self.expanded_xpos)))
                #
                #    if len(self.position_results) > self.max_position_calc_lag:
                #        if not self.position_results[-self.max_position_calc_lag].ready():
                #            print("Waiting for position calculation to catch up")
                #            self.position_results[-self.max_position_calc_lag].wait(0.5)
                # else:
                #    self.step_results.append(cupy_positions_from_modes_with_copy(self.surviving_modes, self.expanded_xpos))

                self.mode_counts.append(
                    len(self.surviving_modes) + len(self.kick_surviving_modes)
                )
            if step % (self.video_mod * self.prune_survivors_every) == 0:
                abs_survivors_indices = (
                    cp.abs(self.surviving_modes[:, 2]) < self.mode_discard
                )
                abs_survivors_indices_kick = (
                    cp.abs(self.kick_surviving_modes[:, 2]) < self.mode_discard
                )

                self.surviving_modes = cp.delete(
                    self.surviving_modes, abs_survivors_indices, axis=0
                )
                self.kick_surviving_modes = cp.delete(
                    self.kick_surviving_modes, abs_survivors_indices_kick, axis=0
                )

        self.step_results = cp.asnumpy(full_results) if use_cuda else full_results
        if self.bench_mode:
            duration = time() - self.bench_measure_start
            print(f"BENCH_MEASURE_END {duration:.6f}", flush=True)
        full_results = None
        if use_cuda:
            cp.get_default_memory_pool().free_all_blocks()
        self.pbar.close()
        #        if self.position_calc_threads > 0:
        #            for r in self.position_results:
        #                self.step_results.append(r.get())
        # self.position_pool.close()

        return self.step_results, self.mode_counts
