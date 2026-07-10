from flickering.simulations import simulator
from flickering.simulations.simulator import (
    StringSim,
    StringSimFixedDecays,
    FlatPsTest,
    KickedSim,
)
import numpy as np
import numba
from tqdm.auto import tqdm
import argparse
import os

# decay times from long results
# TODO test with 37C for comparison?
mean_dts_23c = np.array(
    [
        0.05613313,
        0.02050804,
        0.04917177,
        0.04673029,
        0.04583308,
        0.04247826,
        0.0423381,
        0.03960465,
        0.03859304,
        0.03339286,
        0.03089866,
        0.02648897,
        0.02379568,
        0.01984618,
        0.0175166,
        0.01533255,
        0.01323029,
        0.01164717,
        0.01017018,
        0.00883263,
        0.00749877,
        0.00637521,
        0.00543611,
        0.00474048,
        0.00390226,
        0.00343481,
        0.0028427,
        0.00238398,
        0.00203197,
        0.00165856,
        0.00138085,
        0.00107863,
        0.00084845,
        0.00074123,
        0.0006737,
        0.00061683,
        0.00057151,
        0.00053515,
        0.0005056,
        0.00047794,
        0.0004546,
        0.00043585,
        0.00041597,
        0.0003991,
        0.00038714,
        0.00037212,
        0.00035684,
        0.00034402,
        0.00033021,
        0.00030429,
        0.0003141,
        0.00031099,
        0.00031125,
        0.00026428,
        0.00028153,
        0.0002828,
        0.00026309,
        0.00025512,
        0.00024673,
        0.00025399,
        0.00025907,
        0.00025684,
        0.00025339,
        0.00026315,
        0.00025669,
        0.00026116,
        0.00025154,
        0.00025451,
        0.00022688,
        0.0002472,
        0.00021387,
        0.00026038,
        0.00025165,
        0.00020545,
        0.00027561,
        0.00026892,
        0.0002399,
        0.00025374,
        0.0002622,
        0.00025335,
        0.0002688,
        0.00022433,
        0.00025665,
        0.00026571,
        0.00024848,
        0.00027766,
        0.00024488,
        0.00025172,
        0.0002908,
        0.00026119,
        0.00026051,
        0.00028711,
        0.00028967,
        0.00023907,
        0.00026126,
        0.00029438,
        0.00029091,
        0.00029507,
        0.0002951,
        0.00028551,
        0.00029163,
        0.00031014,
        0.00030713,
        0.00028804,
        0.00032056,
        0.00031344,
        0.00031405,
        0.00032317,
        0.00032478,
        0.00032032,
        0.00034305,
        0.00033016,
        0.0003403,
        0.00034469,
        0.0003654,
        0.00033775,
        0.00038915,
        0.00039017,
        0.00046023,
        0.00067602,
        0.0007801,
        0.00068919,
        0.00046554,
        0.00043919,
        0.00046617,
        0.00045568,
        0.00090449,
        0.00116532,
        0.0013016,
        0.00132701,
        0.00133534,
        0.00126639,
        0.00142105,
        0.00159779,
        0.00144581,
        0.00178335,
        0.0015417,
        0.00137864,
        0.00127058,
        0.00103872,
        0.0005605,
        0.00051755,
        0.00053815,
        0.00051942,
        0.0005241,
        0.00049924,
        0.00051515,
        0.00048796,
        0.00050756,
        0.00048993,
        0.00048584,
        0.00046823,
        0.00046384,
        0.00045023,
        0.00045209,
        0.00044001,
        0.00044019,
        0.00043817,
        0.00043985,
        0.00043324,
        0.00043681,
        0.00043411,
        0.00043393,
        0.00043985,
        0.00043909,
        0.00044685,
        0.00044121,
        0.00045458,
        0.00044763,
        0.00045939,
        0.00045451,
        0.0004645,
        0.00045868,
        0.00047164,
        0.00045451,
        0.00047084,
        0.00045266,
        0.00047687,
        0.00044052,
        0.00047912,
        0.00046571,
        0.00047912,
        0.00044052,
        0.00047687,
        0.00045266,
        0.00047084,
        0.00045451,
        0.00047164,
        0.00045868,
        0.0004645,
        0.00045451,
        0.00045939,
        0.00044763,
        0.00045458,
        0.00044121,
        0.00044685,
        0.00043909,
        0.00043985,
        0.00043393,
        0.00043411,
        0.00043681,
        0.00043324,
        0.00043985,
        0.00043817,
        0.00044019,
        0.00044001,
        0.00045209,
        0.00045023,
        0.00046384,
        0.00046823,
        0.00048584,
        0.00048993,
        0.00050756,
        0.00048796,
        0.00051515,
        0.00049924,
        0.0005241,
        0.00051942,
        0.00053815,
        0.00051755,
        0.0005605,
        0.00103872,
        0.00127058,
        0.00137864,
        0.0015417,
        0.00178335,
        0.00144581,
        0.00159779,
        0.00142105,
        0.00126639,
        0.00133534,
        0.00132701,
        0.0013016,
        0.00116532,
        0.00090449,
        0.00045568,
        0.00046617,
        0.00043919,
        0.00046554,
        0.00068919,
        0.0007801,
        0.00067602,
        0.00046023,
        0.00039017,
        0.00038915,
        0.00033775,
        0.0003654,
        0.00034469,
        0.0003403,
        0.00033016,
        0.00034305,
        0.00032032,
        0.00032478,
        0.00032317,
        0.00031405,
        0.00031344,
        0.00032056,
        0.00028804,
        0.00030713,
        0.00031014,
        0.00029163,
        0.00028551,
        0.0002951,
        0.00029507,
        0.00029091,
        0.00029438,
        0.00026126,
        0.00023907,
        0.00028967,
        0.00028711,
        0.00026051,
        0.00026119,
        0.0002908,
        0.00025172,
        0.00024488,
        0.00027766,
        0.00024848,
        0.00026571,
        0.00025665,
        0.00022433,
        0.0002688,
        0.00025335,
        0.0002622,
        0.00025374,
        0.0002399,
        0.00026892,
        0.00027561,
        0.00020545,
        0.00025165,
        0.00026038,
        0.00021387,
        0.0002472,
        0.00022688,
        0.00025451,
        0.00025154,
        0.00026116,
        0.00025669,
        0.00026315,
        0.00025339,
        0.00025684,
        0.00025907,
        0.00025399,
        0.00024673,
        0.00025512,
        0.00026309,
        0.0002828,
        0.00028153,
        0.00026428,
        0.00031125,
        0.00031099,
        0.0003141,
        0.00030429,
        0.00033021,
        0.00034402,
        0.00035684,
        0.00037212,
        0.00038714,
        0.0003991,
        0.00041597,
        0.00043585,
        0.0004546,
        0.00047794,
        0.0005056,
        0.00053515,
        0.00057151,
        0.00061683,
        0.0006737,
        0.00074123,
        0.00084845,
        0.00107863,
        0.00138085,
        0.00165856,
        0.00203197,
        0.00238398,
        0.0028427,
        0.00343481,
        0.00390226,
        0.00474048,
        0.00543611,
        0.00637521,
        0.00749877,
        0.00883263,
        0.01017018,
        0.01164717,
        0.01323029,
        0.01533255,
        0.0175166,
        0.01984618,
        0.02379568,
        0.02648897,
        0.03089866,
        0.03339286,
        0.03859304,
        0.03960465,
        0.0423381,
        0.04247826,
        0.04583308,
        0.04673029,
        0.04917177,
        0.02050804,
    ]
)


def run_simulation(
    output_file,
    sigma=5e-7,
    kappa=200e-21,
    mean_dts=mean_dts_23c,
    dt=4.90e-5,
    dt_video=1 / 660,
    R=4e-6,
    steps=200000,
    seed=1,
    ss=None,
    shutter_time=0,
    kick_ampl=None,
    kick_time=None,
    warmup_steps=0,
    bench=False,
):
    if ss is None:
        ss = StringSim()
    ss.shutter_time = shutter_time
    ss.sigma = sigma
    ss.kappa = kappa
    ss.mean_dts = mean_dts
    ss.R = R
    ss.warmup_steps = warmup_steps
    ss.bench_mode = bench
    # ss.position_calc_threads = 1
    ss.prune_survivors_every = 1
    # ss.mode_discard_cap = 1e-13

    if hasattr(ss, "kick_ampl") and kick_ampl is not None:
        ss.kick_ampl = kick_ampl
    if hasattr(ss, "kick_time") and kick_time is not None:
        ss.kick_time = kick_time

    ss.dt = dt  # this is the simulation dt, error ~ dt/tau due to the discretisation of mode generation here. mode 30 has tau ~1ms
    ss.dt_video = dt_video  # based on real video
    ss.time_steps = steps  # 20s
    ss.prep_sim()
    step_results, mode_counts = ss.simulate()
    step_results = np.array(step_results) + ss.R

    with open(output_file, "wb") as f:
        data = {
            "step_results": step_results,
            "mode_counts": mode_counts,
            "sigma": sigma,
            "kappa": kappa,
            "dt": dt,
            "dt_video": dt_video,
            "kt": ss.kT,
            "decay_times": mean_dts,
            "seed": seed,
            "mode_discard": ss.mode_discard,
            "shutter_time": ss.shutter_time,
        }
        if hasattr(ss, "kick_ampl"):
            data["kick_ampl"] = ss.kick_ampl
        if hasattr(ss, "kick_time"):
            data["kick_time"] = ss.kick_time
        np.savez(f, **data)


# 5h
# this was broken, rerunnin
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
        description="Run membrane fluctuation simulations."
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
    parser.add_argument(
        "--steps", type=int, default=120 * 660, help="Number of frames to simulate."
    )
    parser.add_argument(
        "--dt", type=float, default=50e-6, help="Simulation time step (dt)."
    )
    parser.add_argument(
        "--fps", type=float, default=660, help="Frames per second (sets dt_video)."
    )
    parser.add_argument(
        "--kicked", action="store_true", help="Use KickedSim instead of StringSim."
    )
    parser.add_argument(
        "--flat_ps", action="store_true", help="Use FlatPsTest instead of StringSim."
    )
    parser.add_argument(
        "--kick_ampl", type=float, default=1.6e-9, help="Amplitude for KickedSim."
    )
    parser.add_argument(
        "--kick_time", type=float, default=0.1, help="Kick time for KickedSim."
    )
    parser.add_argument(
        "--decay_min_ms", type=float, default=1.0, help="Minimum decay time in ms."
    )
    parser.add_argument(
        "--decay_max_ms", type=float, default=500.0, help="Maximum decay time in ms."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of warmup steps to ignore."
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Enable benchmark mode (print timing markers).",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Run on CPU using numpy instead of GPU/cupy."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads for Numba (CPU mode).",
    )

    args = parser.parse_args()

    decay_times = np.logspace(
        np.log10(args.decay_min_ms * 1e-3),
        np.log10(args.decay_max_ms * 1e-3),
        51,
    )
    os.makedirs(args.output_folder, exist_ok=True)

    if args.kicked:
        ss_class = KickedSim
        mode_str = "kick"
    elif args.flat_ps:
        ss_class = FlatPsTest
        mode_str = "flatps"
    else:
        ss_class = StringSim
        mode_str = "nokick"

    if args.cpu:
        if args.threads:
            numba.set_num_threads(args.threads)
        simulator.set_device("cpu")

    for j in range(args.seed, args.seed + args.num_seeds):
        simulator.cp.random.seed(j)
        dts = decay_times
        output_file = os.path.join(
            args.output_folder,
            f"{mode_str}-spectrum_dt{int(args.dt*1e6)}-fps{int(args.fps)}-{j}.npz",
        )
        print(f"Running simulation mode={mode_str}, seed={j}, output={output_file}")
        run_simulation(
            output_file,
            mean_dts=dts,
            seed=j,
            steps=args.steps,
            dt=args.dt,
            dt_video=1 / args.fps,
            ss=ss_class(),
            shutter_time=args.shutter_time,
            kick_ampl=args.kick_ampl,
            kick_time=args.kick_time,
            warmup_steps=args.warmup_steps,
            bench=args.bench,
        )


if __name__ == "__main__":
    main()
