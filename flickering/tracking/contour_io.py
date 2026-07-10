
import numpy as np
import os
from threading import RLock

class ContourIO:
    def __init__(self, file=None):
        self.contours = []
        self.tracker_version = None
        self.valid_indices = []
        self.rlock = RLock()
        self._extras = {}
        self.mode = "XY"
        self.centers = []

        if file is not None and os.path.isfile(file):
            self.load(file)

    def write(self, file):
        # had some trouble saving before doing this
        #this might cause issues with large arrays (memory allocated multiple times)
        contours_to_save = self.contours #np.array(self.contours, dtype=object)
        validity_to_save = np.array(self.valid_indices, dtype=object)
        centers = np.array(self.centers)
        with self.rlock:
            np.savez(
                file + ".npz", contours=contours_to_save, validity=validity_to_save, version=self.tracker_version, centers=centers, mode=self.mode, **self._extras
            )

    def add_contour(self, contour, is_valid=True, center=None):
        """Add contour

        Args:
            contour (list|np.ndarray): list of xy points
            is_valid (bool, optional): is contour considered valid. Defaults to True
        """
        with self.rlock:
            # logger.info(f"Contour length: {len(contour)}")
            self.contours.append(contour)
            self.valid_indices.append(is_valid)  # default to valid
            if center is not None:
                self.centers.append(center)

    def load(self, file):
        data = np.load(
            file, allow_pickle=True
        )  # allow_pickle added as quick fix on 27.04
        self.contours = list(data["contours"])
        if "mode" in data and data["mode"] == "R":
            self.mode = "R"
            self.centers = list(data["centers"])

        self.valid_indices = list(data["validity"])
        if "version" in data:
            self.tracker_version = data["version"]
        else:
            self.tracker_version = 0

        self._extras = {}
        for k in data.keys():
            if k not in ["contours", "validity", "version", "centers"]:
                self._extras[k] = data[k]

    def display(self, index=0):
        # TODO: for debugging purposes, show the contour array after loaded/added
        print(self.contours[index])
        print(self.valid_indices[index])
        print(len(self.contours[index]))

    def get_extras(self):
        return self._extras

    def add_extras(self, extras_dict):
        for k, v in extras_dict.items():
            if not isinstance(v, np.ndarray):
                raise TypeError("Invalid data type in extras")
        self._extras = extras_dict