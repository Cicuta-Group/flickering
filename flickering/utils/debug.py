import matplotlib as mpl
import matplotlib.pyplot as plt

# plt.set_loglevel("warning")
import numpy as np
import cv2

DISPLAY = False
DEBUG = True
WAIT_KEY = 0

DISPLAY_MODE = "PLT"

#DISPLAY_MODE = "CV"

# can control which images should be shown
DISPLAY_IMAGES = {
    "UNROLLED": True,
    "CONTOUR": True,
    "MASK": True,
    "CONTOUR_DETECTION": True,
    "CORRELATION_AREAS": False,
    "CORRELATIONS": True,
}


def debug_display(type: str, image: np.ndarray, title=""):
    """
    Helper to show a debug image. I will either be shown with cv2.imshow or plt.imshow for more flexibility.

    DISPLAY_IMAGES defines which image types are shown
    """
    if not DISPLAY:
        return False

    #print(f"Display image {type}: {title}")
    if type.upper() in DISPLAY_IMAGES and not DISPLAY_IMAGES[type.upper()]:
        return False

    if len(np.array(image).shape) == 1:
        # plt.plot for 1D functions
        plt.figure()
        plt.plot(image)
        plt.title(f"{type}: {title}")
        plt.show()
        return

    if DISPLAY_MODE == "PLT":
        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.title(f"{type}: {title}")
        plt.show()
        return

    cv2.imshow(f"{type}: {title}", image)
    cv2.waitKey(WAIT_KEY)
