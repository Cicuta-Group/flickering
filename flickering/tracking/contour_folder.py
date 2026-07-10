import os
import pyinotify
from queue import Queue
from re import match
from flickering.tracking.correlation_tracker import CorrelationContourTracker as CCT

from glob import glob
import threading
import matplotlib as mpl
import logging
from tqdm.auto import tqdm
import logging
from time import time, sleep
import sys
from os.path import abspath
import argparse
from flickering.utils.standard_configs import default_tracker, vesicle_tracker
from random import shuffle
from datetime import datetime

mpl.use("Agg")


# chatgpt
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process all movie file in folder, tracking changes."
    )

    parser.add_argument(
        "--recurse",
        "-r",
        action="store_true",
        default=False,
        help="Recurse into subfolders",
    )
    parser.add_argument(
        "--overwrite",
        "-w",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default="2026-04-21",
        help="Overwrite contours older than this date (YYYY-MM-DD), default: 2026-04-21",
    )
    parser.add_argument(
        "--reversed",
        "-i",
        action="store_true",
        default=False,
        help="Process existing files in reverse order",
    )
    parser.add_argument(
        "--existing",
        "-e",
        action="store_true",
        default=False,
        help="Process existing files",
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=8, help="Number of threads"
    )
    parser.add_argument(
        "--savemode",
        "-s",
        default="XY",
        help="Save contours as XY coordinates or R (radial + centers)",
    )
    parser.add_argument(
        "--nofollow",
        "-n",
        action="store_true",
        default=False,
        help="Do not monitor folder and track new movies",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="Maximum logging"
    )
    parser.add_argument(
        "--vesicle", "-g", action="store_true", default=False, help="Vesicle settings"
    )
    parser.add_argument(
        "--min-frames",
        "-m",
        type=int,
        default=6000,
        help="Minimum number of frames to process (to avoid processing incomplete copies)",
    )
    parser.add_argument(
        "--polling",
        "-p",
        action="store_true",
        default=False,
        help="Use polling instead of inotify (useful for network shares)",
    )
    parser.add_argument(
        "--randomised",
        "-a",
        action="store_true",
        default=False,
        help="Randomise processing order (minimise chance of processing same file on multiple machines at the same time TODO: implement locking)",
    )
    parser.add_argument(
        "folders", nargs="*", help="List of folders (only first is used!)"
    )
    args = parser.parse_args()

    return args


queue = None
config = None
process_current = False
watch_folder = ""
logfile = ""
pbar = None
process_set = set()


def process_movie(movie_file, delete_incomplete=True):
    logging.debug(f"Processing {movie_file}")
    if not config.vesicle:
        cct = default_tracker()
    else:
        cct = vesicle_tracker()
    cct.threads = config.threads

    cct.save_mode = config.savemode

    try:
        contours, valid = cct.process_movie(
            movie_file, prevent_overwrite=config.overwrite.timestamp()
        )
    except Exception as e:
        logging.error(f"Failed to process {movie_file}", exc_info=e)
        return -1

    if contours is None or len(contours) < config.min_frames:
        if contours is not None:
            logging.warning(
                f"Only {len(contours)} frames found in {movie_file}, minimum is {config.min_frames}, deleting contour file"
            )
        else:
            logging.warning(f"No contours found in {movie_file}, deleting contour file")
        # TODO: move this logic to a CCT method and use it in process_movie too
        contour_filename = movie_file.replace(".movie", f"_contour")
        try:
            if delete_incomplete:
                os.remove(contour_filename)
        except Exception as e:
            logging.error(
                f"Failed to remove contour file {contour_filename}", exc_info=e
            )
        return False

    return True


def get_all_files(folders, recurse=False):
    files = []
    for folder in folders:
        files += glob(f"{folder}/*.movie")
        files += glob(f"{folder}/*.mkv")
        if recurse:
            files += glob(f"{folder}/**/*.movie")
            files += glob(f"{folder}/**/*.mkv")

    return files


def process_queue(queue: Queue):
    while (not config.nofollow) or not queue.empty():
        path = queue.get()
        logging.debug(f"Processing: {path}")
        res = process_movie(path)
        if res is False:
            logging.info(f"File {path} incomplete, retrying")
            res = process_movie(path, False)
            if res is False:
                logging.warning(f"File {path} incomplete, giving up")
        else:
            pbar.update()


def onChange(ev):
    if str(ev.pathname).endswith(".movie"):
        logging.debug(f"Queuing file: {ev.pathname}")
        queue.put(ev.pathname)
        pbar.total += 1
        pbar.refresh()


if __name__ == "__main__":
    queue = Queue()
    config = parse_arguments()
    process_current = config.existing

    # TODO: handle multiple folders
    watch_folder = abspath(config.folders[0]) + "/"

    if len(config.folders) > 1 and not config.nofollow:
        raise NotImplementedError("Watching multiple folders is not supported")

    logfile = watch_folder + f"tracking_{time():.0f}.log"
    print(f"Logfile: {logfile}")
    logging.basicConfig(
        filename=logfile,
        encoding="utf-8",
        level=logging.DEBUG if config.verbose else logging.INFO,
    )
    logging.getLogger("numba").setLevel(logging.WARNING)

    process_set = set()

    if process_current:
        total = 0
        skipped = 0
        files = get_all_files(config.folders, config.recurse)
        process_set = set(files)

        if config.reversed:
            files = reversed(files)

        if config.randomised:
            shuffle(files)

        for f in files:
            if (
                not CCT.contour_exists(f, config.overwrite.timestamp())
                and f not in queue.queue
            ):
                queue.put(f)
                total += 1
                logging.info(f"Queued {f}")
            else:
                skipped += 1
                logging.info(f"Skipped {f}")

        print(f"Scanned current: {total}/{total+skipped} files queued")

        pbar = tqdm(total=1)
        pbar.total = total
        pbar.refresh()
    else:
        pbar = tqdm(total=0)

    thread = threading.Thread(target=process_queue, args=(queue,), daemon=True)
    # start processing thread
    thread.start()

    if not config.nofollow:
        if not config.polling:
            wm = pyinotify.WatchManager()
            wm.add_watch(
                watch_folder, pyinotify.IN_CLOSE_WRITE, onChange, rec=config.recurse
            )
            notifier = pyinotify.Notifier(wm)
            try:
                notifier.loop()
            except KeyboardInterrupt:
                print(f"Aborted. Queue contains {queue.qsize()} items")
        else:
            # polling mode
            try:
                while True:
                    files = get_all_files(config.folders, config.recurse)
                    for f in files:
                        if (
                            not CCT.contour_exists(f, config.overwrite.timestamp())
                            and f not in process_set
                        ):
                            logging.info(f"Queuing file: {f}")
                            queue.put(f)
                            process_set.add(f)
                            pbar.total += 1
                            pbar.refresh()
                    sleep(20)
            except KeyboardInterrupt:
                print(f"Aborted. Queue contains {queue.qsize()} items")
    else:
        # wait for processing to finish
        while thread.is_alive():
            sleep(0.1)
