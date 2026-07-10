from avifilelib.avi import AviFile
from typing import Union
from pathlib import Path
import numpy as np


class AviMovie:
    """Create instances of .avi files with equivalent versions of relevant methods from the Movie class.
    In its current form, this allows .avi files to be processed by python_contour_trackers.contour_tracker.combo_tracker
    and also by autoimager.correlation_contour.process_movie in the same way as movie files
    (although there seem to be some issues with this, and process_movie needs an extra if/else).

    TODO try
    also may need to subtract the background if this is the dirty pointgrey camera (not sure whether that would work nicely)
    """

    def __init__(self, filename: Union[Path, str]) -> None:
        self.filename = filename
        self.avifile = AviFile(self.filename)
        self.n_frames = self.avifile.avih.total_frames
        self.frame_iterable = self.avifile.iter_frames(
            stream_id=0
        )  # a generator not a list

    def get_metadata(self):
        """Get metadata...

        Returns: TODO
            _type_: _description_
        """
        return self.avifile.avih

    def get_frame(self, i: int) -> np.ndarray:
        """Mapped to do the same as pytmk.Movie.get_frame

        Args:
            i (int): frame number of the video

        Returns:
            np.ndarray: the frame as an array of pixels
        """
        self.frame_list = list(self.frame_iterable)
        return np.array(self.frame_list[i])  # TODO

    def frames(self, *args):
        """Returns a generator of frames. Should work like range(). Taken directly from pytmk.Movie. TODO: workout typing"""

        return self.frame_iterable
        """
        iterable = False
        if len(args) == 0:
            start = 0
            stop = self.n_frames
            step = 1

        elif len(args) == 1:
            if hasattr(args[0], "__iter__"):
                iterable = True
            else:
                start = 0
                stop = args[0]
                step = 1

        elif len(args) == 2:
            start = args[0]
            stop = args[1]
            step = 1

        elif len(args) == 3:
            start = args[0]
            stop = args[1]
            step = args[2]

        elif len(args) > 3:
            raise TypeError

        if iterable:
            for j in args[0]:
                yield self.get_frame(i=j)

        else:
            for k in range(start, stop, step):
                yield self.get_frame(i=k)"""

    def destroy(self) -> None:
        pass



from flickering.utils.movie_reader import register_movie_reader
register_movie_reader(".avi", AviMovie)

