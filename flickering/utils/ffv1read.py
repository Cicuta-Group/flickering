from typing import Union, Dict
from pathlib import Path
import subprocess
import numpy as np
import ffmpeg
from time import time

class FFV1Movie:
    """Read FFV1 encoded 12 bit video files. This is used primarily for archival storage.
    """
    def __init__(self, filename: Union[Path, str], preload_to_memory=False) -> None:
        self.filename = filename
        self.width, self.height = self._get_video_dimensions()

        self.process = (
            ffmpeg
            .input(filename)
            .output('pipe:', format='rawvideo', pix_fmt='gray12le',loglevel="quiet")
            .run_async(pipe_stdout=True)
        )

        decoded_frames = []
        #TODO: gen n_frames separatenly and turn this into a generator
        while True:
            frame_bytes = self.process.stdout.read(self.width * self.height * 2)
            if not frame_bytes:
                break
            decoded_frame = np.frombuffer(frame_bytes, dtype=np.uint16).reshape(self.height, self.width)
            unpacked_frame = decoded_frame << 4
            decoded_frames.append(unpacked_frame)

        self.n_frames = len(decoded_frames)
        self.frame_data = decoded_frames

    def _get_video_dimensions(self) -> tuple[int, int]:
        """
        Uses ffprobe to get the width and height of the video.
        """
        try:
            probe = ffmpeg.probe(self.filename)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                return width, height
            else:
                raise ValueError("No video stream found in the input file.")
        except ffmpeg.Error as e:
            raise ValueError(f"Error probing video: {e.stderr.decode()}")

    def frames(self, *args):
        if len(args) > 0:
            raise NotImplementedError("Frame selection not implemented for FFV1Movie")
        return self.frame_data

    def destroy(self) -> None:
        pass



from flickering.utils.movie_reader import register_movie_reader
register_movie_reader(".mkv", FFV1Movie)
register_movie_reader(".mp4", FFV1Movie)