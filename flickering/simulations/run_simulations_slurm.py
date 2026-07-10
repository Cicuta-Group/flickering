import argparse
import os
import subprocess
import math
import sys
import numpy as np
from multiprocessing import Process
import time

# Imports will be done lazily in worker mode to avoid dependency issues on submission nodes

def get_slurm_script(args, job_seed, num_sims_in_job):
    """Generates the SLURM batch script content."""
    
    # Calculate total CPUs for this job
    if args.cpu:
        total_cpus = args.threads + 2
        num_sims_in_job = 1 # Run 1 simulation per job as requested
    else:
        total_cpus = num_sims_in_job * args.cpus_per_sim
    
    # Construct the command to run this script in worker mode
    # We pass all the simulation arguments forward
    worker_cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        f"--seed {job_seed}",
        f"--num_seeds {num_sims_in_job}",
        f"--output_folder {args.output_folder}",
        f"--shutter_time {args.shutter_time}",
        f"--steps {args.steps}",
        f"--dt {args.dt}",
        f"--fps {args.fps}",
        f"--kick_ampl {args.kick_ampl}",
        f"--kick_time {args.kick_time}",
        f"--decay_min_ms {args.decay_min_ms}",
        f"--decay_max_ms {args.decay_max_ms}",
    ]
    if args.kicked:
        worker_cmd.append("--kicked")
    if args.flat_ps:
        worker_cmd.append("--flat_ps")
    if args.cpu:
        worker_cmd.append("--cpu")
        worker_cmd.append(f"--threads {args.threads}")
        
    worker_cmd_str = " ".join(worker_cmd)

    gpu_req = "#SBATCH --gres=gpu:1" if not args.cpu else ""

    script = f"""#!/bin/bash
#SBATCH --job-name=sim_{"kick" if args.kicked else "nokick"}_{job_seed}
#SBATCH --output={args.output_folder}/job_{job_seed}.out
#SBATCH --error={args.output_folder}/job_{job_seed}.err
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
{gpu_req}
#SBATCH --cpus-per-task={total_cpus}
#SBATCH --mem={args.mem_gb}G
#SBATCH --time={args.time}

echo "Starting worker with {num_sims_in_job} parallel simulations..."
{worker_cmd_str}
"""
    return script

def worker_main(args):
    """Runs multiple simulations in parallel using subprocess for robustness."""
    
    # Path to the original run_simulations.py
    sim_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_simulations.py")
    
    processes = []
    
    for j in range(args.seed, args.seed + args.num_seeds):
        cmd = [
            sys.executable,
            sim_script,
            "--seed", str(j),
            "--num_seeds", "1",
            "--output_folder", args.output_folder,
            "--shutter_time", str(args.shutter_time),
            "--steps", str(args.steps),
            "--dt", str(args.dt),
            "--fps", str(args.fps),
            "--kick_ampl", str(args.kick_ampl),
            "--kick_time", str(args.kick_time),
            "--decay_min_ms", str(args.decay_min_ms),
            "--decay_max_ms", str(args.decay_max_ms),
        ]
        if args.kicked:
            cmd.append("--kicked")
        if args.flat_ps:
            cmd.append("--flat_ps")
        if args.cpu:
            cmd.append("--cpu")
            # In CPU mode, run_simulations.py handles threads
            if hasattr(args, 'threads'):
                cmd.extend(["--threads", str(args.threads)])
            
        print(f"[Worker] Launching simulation: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        processes.append(p)

    # Wait for all to finish
    exit_codes = [p.wait() for p in processes]
    
    if any(ec != 0 for ec in exit_codes):
        print(f"[Worker] Some simulations failed with exit codes: {exit_codes}")
        sys.exit(1)
    
    print("[Worker] All simulations in this job finished successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Run membrane fluctuation simulations on SLURM."
    )
    
    # Simulation parameters (same as run_simulations.py)
    parser.add_argument("--seed", type=int, default=0, help="Starting seed number.")
    parser.add_argument("--num_seeds", "--num_repeats", type=int, dest="num_seeds", default=10, 
                        help="Number of seeds to run in total (repeats).")
    parser.add_argument("--output_folder", type=str, default="./sim_results", help="Output folder.")
    parser.add_argument("--shutter_time", type=float, default=0)
    parser.add_argument("--steps", type=int, default=120 * 660)
    parser.add_argument("--dt", type=float, default=50e-6)
    parser.add_argument("--fps", type=float, default=660)
    parser.add_argument("--kicked", action="store_true")
    parser.add_argument("--flat_ps", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--kick_ampl", type=float, default=1.6e-9)
    parser.add_argument("--kick_time", type=float, default=0.1)
    parser.add_argument("--decay_min_ms", type=float, default=1.0)
    parser.add_argument("--decay_max_ms", type=float, default=500.0)

    # SLURM specific parameters
    parser.add_argument("--account", type=str, required=True, help="SLURM account name.")
    parser.add_argument("--partition", type=str, default="gpu", help="SLURM partition.")
    parser.add_argument("--sims_per_job", type=int, default=5, help="Number of parallel simulations per SLURM job.")
    parser.add_argument("--cpus_per_sim", type=int, default=8, help="CPUs to request per parallel simulation.")
    parser.add_argument("--mem_gb", type=int, default=40, help="Total memory in GB to request per SLURM job.")
    parser.add_argument("--time", type=str, default="04:00:00", help="SLURM time limit (HH:MM:SS).")
    parser.add_argument("--dry_run", action="store_true", help="Generate scripts but do not submit.")
    
    # Internal mode
    parser.add_argument("--worker", action="store_true", help="Internal flag: run in worker mode.")

    args = parser.parse_args()
    
    if args.worker:
        worker_main(args)
        return

    # Submission mode
    os.makedirs(args.output_folder, exist_ok=True)
    if args.cpu:
        args.sims_per_job = 1
    num_jobs = math.ceil(args.num_seeds / args.sims_per_job)
    
    print(f"Submitting {args.num_seeds} simulations in {num_jobs} jobs ({args.sims_per_job} sims per job).")
    
    for i in range(num_jobs):
        job_seed = args.seed + i * args.sims_per_job
        num_sims_in_job = min(args.sims_per_job, args.num_seeds - i * args.sims_per_job)
        
        script_content = get_slurm_script(args, job_seed, num_sims_in_job)
        script_path = os.path.join(args.output_folder, f"submit_{'kick' if args.kicked else 'nokick'}_{job_seed}.sh")
        
        with open(script_path, "w") as f:
            f.write(script_content)
            
        if not args.dry_run:
            print(f"Submitting job for seeds {job_seed} to {job_seed + num_sims_in_job - 1}...")
            subprocess.run(["sbatch", script_path])
        else:
            print(f"[Dry Run] Generated script: {script_path}")

if __name__ == "__main__":
    main()
