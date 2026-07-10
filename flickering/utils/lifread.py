from readlif.reader import LifFile
from typing import Union
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import logging

# import autoimager.correlation_contour as cc


class LifMovie:
    """Create instances of .lif files with equivalent versions of relevant methods from the Movie class.
    In its current form, this allows .lif files to be processed by python_contour_trackers.contour_tracker.combo_tracker
    and also by autoimager.correlation_contour.process_movie in the same way as movie files
    (although there seem to be some issues with this, and process_movie needs an extra if/else).

    Note also that LifMovie.file is actually the video object we want to pass to any Movie analysis;
    the class however initialised the entire .lif experiment along with the metadata, and then the actual video
    is chosen in __init__ through the series and channel args.

    TODO try autoimager tracking on confocals and also work out why contour_tracker not working on BF lifs

    channel = 0 --> confocal
    channel = 1 --> bright field
    """

    def __init__(
        self, filename: Union[Path, str], series: int, channel: int = 0
    ) -> None:
        self.filename = filename
        self.series = series
        self.channel = channel
        self.experiment = LifFile(filename)
        self.num_series = self.experiment.num_images
        self.xml_header = self.experiment.xml_header
        self.xml_root = self.experiment.xml_root
        self.offsets = self.experiment.offsets
        self.video_list = [video for video in self.experiment.get_iter_image()]
        self.file = self.experiment.get_image(self.series)
        self.n_frames = self.file.dims.t

        self.get_metadata()

        # to be initialised during correlation_contour.process_lif_movie()
        # TODO: different way to load without re-detecting contours
        self.center = None
        self.radius_px = None
        self.contour_file = None
        self.contour_exists = False  # can this be saved?

    def get_metadata(self):
        """Get metadata from the .lif file. This is used to make it more pythonic as I found the
        XML tree unintuitive. The key names have been taken over from the LIF style.

        Returns:
            dict: containing all relevant parameters in the form {"ParameterName": {info}, ...}
        """

        etroot = ET.fromstring(self.xml_header)

        self.attrib_dict = {}
        for Element in etroot[0][2][self.series][0][0][2][0][0]:
            self.attrib_dict[Element.attrib["Identifier"]] = {
                "Variant": Element.attrib["Variant"],
                "Unit": Element.attrib["Unit"],
                # Data
                # Description
                # VariantType
            }

        for Element in etroot[0][2][self.series][0][0][1][1]:
            self.attrib_dict["Pixels_" + Element.attrib["DimID"]] = {
                "Origin": Element.attrib["Origin"],
                "NumberOfElements": Element.attrib["NumberOfElements"],
                # Length
                # Unit
                # BitInc
                # BytesInc
            }

        for Element in etroot[0][2][self.series][0][0][2][0][1]:
            self.attrib_dict[Element.attrib["ObjectName"]] = {
                "ClassName": Element.attrib["Variant"],
                "Attribute": Element.attrib["Attribute"],
                "Description": Element.attrib["Description"],
                "Data": Element.attrib["Data"],
                "Variant": Element.attrib["Variant"],
                "VariantType": Element.attrib["VariantType"],
            }

        return self.attrib_dict

    def get_frame(self, i: int) -> np.ndarray:
        """Mapped to do the same as pytmk.Movie.get_frame

        Args:
            i (int): frame number of the video

        Returns:
            np.ndarray: the frame as an array of pixels
        """
        return np.array(self.file.get_frame(z=0, t=i, c=self.channel))

    def frames(self, *args):
        """Returns a generator of frames. Should work like range(). Taken directly from pytmk.Movie. TODO: workout typing"""
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
                yield self.get_frame(i=k)
    def destroy(self) -> None:
        pass

    # def get_contour(self):
    #    cc.process_lif_movie(self.filename, self.series, self.channel)


from flickering.utils.movie_reader import register_movie_reader
register_movie_reader(".lif", LifMovie)

    # def load_contour(self, contour_filename):
    #     self.contour_file =
    #     self.contour_exists =

    # def get_center_radius(self):
    #     if self.contour_exists:
    #         self.center =
    #         self.radius_px =


if __name__ == "__main__":
    logging.basicConfig(filename="lifread.log", level=logging.INFO)
    movie = LifMovie(
        r"\\sf3\cicutagroup\rg614\40xlens_resonant.lif", series=4, channel=0
    )

    # within this dict, should have all relevant parameters, e.g.
    pinhole_size_m = movie.attrib_dict["dblPinhole"][
        "Variant"
    ]  # needs conversion to focal depth (quadratic -- see Rautu fig S.1)
    pixel_size_m = float(movie.attrib_dict["dblSizeX"]["Variant"]) / float(
        movie.attrib_dict["Pixels_1"]["NumberOfElements"]
    )

    print(f"Pinhole size = {pinhole_size_m} m, pixel = {pixel_size_m} m/pix")

    ## print out prettier xml formatting to understand it
    # etroot = ET.fromstring(self.xml_header)
    # tree = ET.ElementTree(etroot)
    # ET.indent(tree, space="\t", level=0)
    # tree.write("lifread.log", encoding="utf-8")
