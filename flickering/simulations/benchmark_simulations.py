import subprocess
import time
import os
import sys
import psutil
import threading
import numpy as np
import re
import argparse

# Configuration
SIM_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_simulations.py"
)
OUTPUT_FOLDER = "/tmp/bench_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Base command arguments as requested by the user
WARMUP_STEPS = 500
MEASURE_STEPS = 200

BASE_ARGS = [
    "--steps",
    str(MEASURE_STEPS),
    "--warmup_steps",
    str(WARMUP_STEPS),
    "--fps",
    "2000",
    "--dt",
    "20e-6",
    "--kicked",
    "--decay_min_ms",
    "0.5",
    "--decay_max_ms",
    "100",
    "--num_seeds",
    "1",
    "--output_folder",
    OUTPUT_FOLDER,
    "--bench",
]

TEST_COUNTS = [1, 2, 4, 8, 12, 16, 20, 24, 32]
THREAD_COUNTS = list(reversed([4, 8, 12, 16, 20, 24]))  # For CPU mode scaling


def get_gpu_stats():
    """Returns (memory_used_mb, gpu_utilization_percent)."""
    try:
        res = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            .split(",")
        )
        return float(res[0]), float(res[1])
    except:
        return 0.0, 0.0


def monitor_resources(stop_event, stats):
    """Background thread to sample resource usage."""
    while not stop_event.is_set():
        gpu_mem, gpu_util = get_gpu_stats()
        cpu_util = psutil.cpu_percent(interval=None)
        ram_util = psutil.virtual_memory().used / (1024**3)  # GB

        stats["gpu_mem"].append(gpu_mem)
        stats["gpu_util"].append(gpu_util)
        stats["cpu_util"].append(cpu_util)
        stats["ram_util"].append(ram_util)

        time.sleep(0.5)


def run_benchmark(n_parallel, n_threads=None, is_cpu=False):
    """Runs n_parallel simulations and returns performance metrics for steady state."""
    thread_str = f", Threads/Sim: {n_threads}" if n_threads else ""
    print(
        f"\n>>> Testing {n_parallel} parallel simulations{thread_str} (Warmup: {WARMUP_STEPS}, Measure: {MEASURE_STEPS})..."
    )

    stats = {"gpu_mem": [], "gpu_util": [], "cpu_util": [], "ram_util": []}
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_resources, args=(stop_event, stats)
    )

    processes = []

    current_args = list(BASE_ARGS)
    if is_cpu:
        current_args.append("--cpu")
        if n_threads:
            current_args.extend(["--threads", str(n_threads)])

    for i in range(n_parallel):
        cmd = [
            sys.executable,
            SIM_SCRIPT,
            "--seed",
            str(i + 1000),
        ] + current_args

        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        processes.append(p)

    monitor_thread.start()

    durations = []

    def wait_for_process(proc, results):
        out, err = proc.communicate()
        if proc.returncode != 0 and err:
            print(f"\n[Process Error (Code {proc.returncode})]\n{err}")
        match = re.search(r"BENCH_MEASURE_END ([\d\.]+)", out)
        if match:
            results.append(float(match.group(1)))
        else:
            print(f"    Warning: Could not find BENCH_MEASURE_END. Return code: {proc.returncode}")
            print(f"    --- Output Start ---\n{out[:500]}\n--- ... ---\n--- Output End ---\n{out[-500:]}\n----------------")

    threads = []
    for p in processes:
        t = threading.Thread(target=wait_for_process, args=(p, durations))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    stop_event.set()
    monitor_thread.join()

    if not durations:
        print("    Error: No timing data collected from simulations.")
        return None

    max_duration = np.max(durations)
    avg_duration = np.mean(durations)
    throughput = n_parallel / (max_duration / 60)  # Sims per minute

    res = {
        "n": n_parallel,
        "threads": n_threads or 1,
        "avg_measure_duration": avg_duration,
        "max_measure_duration": max_duration,
        "throughput": throughput,
        "avg_gpu_util": np.mean(stats["gpu_util"]) if stats["gpu_util"] else 0,
        "peak_gpu_mem": np.max(stats["gpu_mem"]) if stats["gpu_mem"] else 0,
        "avg_cpu_util": np.mean(stats["cpu_util"]) if stats["cpu_util"] else 0,
        "peak_ram_gb": np.max(stats["ram_util"]) if stats["ram_util"] else 0,
    }

    print(
        f"    Steady-state duration: avg={avg_duration:.2f}s, max={max_duration:.2f}s"
    )
    print(f"    Throughput: {throughput:.2f} sims/min")

    return res


def main():
    parser = argparse.ArgumentParser(description="Steady-State Simulation Benchmark")
    parser.add_argument(
        "--cpu", action="store_true", help="Run benchmark on CPU instead of GPU."
    )
    parser.add_argument(
        "--mode",
        choices=["parallel", "threads"],
        default="parallel",
        help="Benchmark parallel scaling or thread scaling (CPU only).",
    )
    args_cli = parser.parse_args()

    results = []
    try:
        if args_cli.cpu and args_cli.mode == "threads":
            print(f"Starting Thread-Scaling Benchmark (CPU MODE)")
            for t in THREAD_COUNTS:
                res = run_benchmark(n_parallel=1, n_threads=t, is_cpu=True)
                if res:
                    results.append(res)
                time.sleep(1)
        else:
            mode_name = "GPU" if not args_cli.cpu else "CPU"
            print(f"Starting Parallel-Scaling Benchmark ({mode_name} MODE)")
            for n in TEST_COUNTS:
                if (
                    not args_cli.cpu
                    and results
                    and results[-1]
                    and results[-1]["peak_gpu_mem"] > 35000
                ):
                    print(f"Skipping N={n} as GPU memory is getting full.")
                    break

                res = run_benchmark(n_parallel=n, n_threads=1, is_cpu=args_cli.cpu)
                if res:
                    results.append(res)
                time.sleep(2)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")

    # Calculate video_mod for Steps/s reporting
    fps = 2000
    dt = 20e-6
    for i, arg in enumerate(BASE_ARGS):
        if arg == "--fps":
            fps = float(BASE_ARGS[i+1])
        if arg == "--dt":
            dt = float(BASE_ARGS[i+1])
    video_mod = int(1.0 / (fps * dt))

    print("\n" + "=" * 130)
    print(
        f"{'N':>3} | {'Threads':>7} | {'Avg Dur (s)':>12} | {'Sims/min':>10} | {'Steps/s':>12} | {'Steps/s/Core':>14} | {'GPU Util (%)':>12} | {'CPU Util (%)':>12}"
    )
    print("-" * 130)
    for r in results:
        steps_per_s = (MEASURE_STEPS * video_mod) / r['avg_measure_duration']
        steps_per_s_per_core = steps_per_s / r['threads']
        print(
            f"{r['n']:3d} | {r['threads']:7d} | {r['avg_measure_duration']:12.2f} | {r['throughput']:10.2f} | {steps_per_s:12.1f} | {steps_per_s_per_core:14.1f} | {r['avg_gpu_util']:12.1f} | {r['avg_cpu_util']:12.1f}"
        )
    print("=" * 130)


if __name__ == "__main__":
    main()
