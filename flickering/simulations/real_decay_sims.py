from flickering.simulations.simulator import (
    StringSim,
    StringSimFixedDecays,
    FlatPsTest,
    KickedSim,
)
import numpy as np
import cupy as cp
from tqdm.auto import tqdm
import argparse
import os
from flickering.analysis.fitter import theoretical_tau


def run_simulation(
    output_file,
    sigma=5e-7,
    kappa=200e-21,
    dt=4.90e-5,
    dt_video=1 / 660,
    R=4e-6,
    steps=200000,
    seed=1,
    ss=None,
    shutter_time=0,
):
    if ss is None:
        ss = StringSim()
    ss.shutter_time = shutter_time
    ss.sigma = sigma
    ss.kappa = kappa
    ss.R = R
    # ss.position_calc_threads = 1
    ss.prune_survivors_every = 1
    # ss.mode_discard_cap = 1e-13

    ss.dt = dt  # this is the simulation dt, error ~ dt/tau due to the discretisation of mode generation here. mode 30 has tau ~1ms
    ss.dt_video = dt_video  # based on real video
    ss.time_steps = steps  # 20s
    # ss.max_mode = 30
    ss.mean_dts = theoretical_tau(
        np.arange(ss.max_mode + 1) + 1, 0.02, 0.001, R, kappa, sigma
    )
    print(ss.mean_dts * 1000)
    ss.prep_sim()
    step_results, mode_counts = ss.simulate()
    step_results = np.array(step_results) + ss.R

    with open(output_file, "wb") as f:
        np.savez(
            f,
            step_results=step_results,
            mode_counts=mode_counts,
            sigma=sigma,
            kappa=kappa,
            dt=dt,
            dt_video=dt_video,
            kt=ss.kT,
            decay_times=cp.asnumpy(ss.mean_dts),
            seed=seed,
            mode_discard=ss.mode_discard,
            shutter_time=ss.shutter_time,
        )


# 5h
# for i in tqdm(np.logspace(-1,1, 10)):
#    seed = round(i*100)
#    cp.random.seed(seed)
#    scale_dts = mean_dts_23c*i
#    run_simulation(f"/home/filip/flickering/flickering_experiments/sim_results/s5_k200_dt50-kt4-80s-dc{i:.2f}-0.npz", sigma=5e-7, seed=seed, steps=660*80, mean_dts=scale_dts)

# 5.8h
# finished, prob successful
# for i in tqdm(range(30)):
#    cp.random.seed(i)
#    run_simulation(f"/home/filip/flickering/flickering_experiments/sim_results/s5_k200_dt50-kt4-80s-{i}.npz", seed=i, steps=660*80)


# ~6.5h
# for i in tqdm(range(6,10)):
#    cp.random.seed(i)
#    run_simulation(f"/home/filip/flickering/flickering_experiments/sim_results/s5_k200_dt5-kt4-20s-{i}.npz", dt=4.90e-6, seed=i, steps=660*20)

# estimated 5h
# for i in tqdm(range(1,10)):
#    cp.random.seed(i)
#   run_simulation(f"/home/filip/flickering/flickering_experiments/sim_results/s{i}_k200_dt50-kt4-80s-0.npz", sigma=i*1e-7, seed=i, steps=660*80)

# estimated 7h
# aborted after 7
# for i in tqdm(range(10)):
#    cp.random.seed(i)
#    run_simulation(f"/home/filip/flickering/flickering_experiments/sim_results/s5_k200_dt50-kt4-300s-{i}.npz", seed=i, steps=660*300)

# j=0
# for decay_time_fixed in tqdm(np.logspace(-3,1,30)[20:]):
#    cp.random.seed(j)
#    dts = np.repeat(decay_time_fixed,len(mean_dts_23c))
#    run_simulation(f"/home/filip/flickering/flickering_experiments/sim_results/fixed-{j}_s5_k200_dt750-kt4-80s_cap11.npz", mean_dts=dts, seed=j, steps=660*80, dt=750e-6)
#    j+=1

# j=0
# decay_times = reversed(np.logspace(-3,-0.3,51))
# for i in range(10):
#    cp.random.seed(j)
#    dts = decay_times
#    run_simulation(f"/home/filip/flickering/flickering_experiments/sim_results/const-spectrum_dt50-kt4-80s-{i}.npz", mean_dts=dts, seed=j, steps=660*80, dt=50e-6, ss=FlatPsTest())

# j=0
# decay_times = np.logspace(-3,-0.3,51)
# for i in range(10):
#    cp.random.seed(j)
#    dts = decay_times
#    run_simulation(f"/home/filip/flickering/flickering_experiments/sim_results/const-spectrum-long_dt50-kt4-80s-{i}.npz", mean_dts=dts, seed=j, steps=660*220, dt=50e-6, ss=FlatPsTest())


def main():
    parser = argparse.ArgumentParser(
        description="Run membrane fluctuation simulations using real decay times and spectrum shape"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed number for the simulation."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./sim_results",
        help="Output folder for results.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="Number of seeds to run (default: 10).",
    )
    parser.add_argument(
        "--shutter_time", type=float, default=0, help="Shutter time for the simulation."
    )
    args = parser.parse_args()

    # decay_times = np.logspace(-3, -0.3, 51)
    os.makedirs(args.output_folder, exist_ok=True)
    for j in range(args.seed, args.seed + args.num_seeds):
        cp.random.seed(j)
        # dts = decay_times
        output_file = os.path.join(
            args.output_folder,
            f"kicked-realspectrum-realdecay-long_dt50-kt4-120s-{j}.npz",
        )

        run_simulation(
            output_file,
            # mean_dts=dts,
            seed=j,
            steps=660 * 120,
            dt=50e-6,
            ss=KickedSim(),
            shutter_time=args.shutter_time,
        )


if __name__ == "__main__":
    main()
