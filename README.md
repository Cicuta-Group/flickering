# High Throughput Flickering

This repository provides tools for performing, tracking, and analyzing membrane flickering experiments on red blood cells.

## Key Features

- **Automated Acquisition**: Automated cell recording logic and configs, hardware agnostic.
- **Contour Tracking**: High-performance correlation-based boundary tracking for extracting cell contours from video files (flexible format support)
- **Fluctuation Analysis**: Calculation of power spectra (mean squared thermal fluctuations) and autocorrelation functions and extraction of physical parameters like bending modulus ($\kappa$) and membrane tension ($\sigma$) by fitting theoretical models to the experimental power spectra.
- **Simulation Engine**: Used to generate synthetic data for testing.

## Installation

Install the core `flickering` analysis and tracking package:

```bash
# Setup virtual env
python -m venv .venv
source .venv/bin/activate

# then this package
pip install -e .
```

Requirements will be automatically resolved. This repository requires Python >= 3.9.

This is not needed to use the contour tracking or contour analysis.

## Flickering experiment steps

1.  **Automated Imaging (Acquisition)**: Core logic and experiment setups to automate cell finding, autofocusing, and high-speed movie recording.
    > To use the automated recording logic a compatible microscope driver will be required see below for how to implement it for your microscope.
2.  **Contour Tracking**: Cell shape tracking from video files using correlation-based edge detection.
3.  **Physical Parameter Extraction from fluctuations**: Extraction of spectra, filtering, and fitting to model with membrane tension ($\sigma$) and bending modulus ($\kappa$).

## Step 1: Automated Imaging

Automated acquisition is orchestrated by the `CellFindingExperiment` class (and its specialized subclasses) in combination with a hardware class that implements the `Microscope` Protocol.

### Supported Experiment Classes
*   `ShutterSpeedExperiment`: Used to record same cell at different exposure times
*   `MultiWellExperiment`: Used for high-throughput multi-well assays (e.g., glutaraldehyde effect runs).
*   `ZstackExperiment`: Used to acquire focus calibration stacks.

### Example Setup Script (Custom Microscope)

To run automated acquisition on your own custom hardware, you can implement the `Microscope` interface/protocol defined in [microscope.py](flickering/acquisition/microscope.py):

```python
import numpy as np
from flickering.acquisition.microscope import Microscope
from flickering.experiments.ShutterSpeedExperiment import ShutterSpeedExperiment

# 1. Define your own microscope communication layer
class MyCustomMicroscope(Microscope):
    camera_name = "MyCamera"
    _z = 0.0

    def get_stage_position(self, update=True):
        # Return [x, y] coordinates in microns
        return np.array([0.0, 0.0])

    def get_z(self, update=True):
        return self._z

    def move_stage(self, move_by, absolute=False, speed=1000.0):
        # Move the stage hardware
        return True

    def move_z(self, move_by, absolute=False, wait=True, speed=50.0):
        if absolute:
            self._z = move_by
        else:
            self._z += move_by
        return True

    def get_image(self, trigger=True):
        # Capture and return a 2D numpy array frame from the camera
        return np.zeros((512, 512), dtype=np.uint16)

    # Implement other required methods defined in the Microscope Protocol...

# 2. Initialize your custom microscope interface
m = MyCustomMicroscope()

# 3. Configure search locations (e.g., coordinates)
wells = [np.array([100.0, 100.0]), np.array([200.0, 200.0])]

# 4. Instantiate the experiment
output_folder = "./experiments/run_data/"
e = ShutterSpeedExperiment(
    microscope=m,
    wells=wells,
    data_folder=output_folder
)
e.duration = 10  # seconds

# 5. Start the automated run
e.run()
```

### Running and Modifying Experiment Scripts

The experiment scripts in `flickering/experiments/` (such as `MultiWellExperiment.py` or `ShutterSpeedExperiment.py`) are structured to make it easy to adapt them to custom hardware:
- **Top-level Class Definition**: The actual experiment logic is encapsulated in class definitions at the top level of the file, completely free of any microscope-specific dependencies.
- **Main Execution Block**: The default script execution setup is isolated inside `if __name__ == "__main__":` at the bottom of the files. By default, it targets the Temika microscope setup and coordinates. If you run these scripts directly without `temika` and `TemikaXML` packages installed the script will not work. To use you should modify this main block to instantiate and pass your own custom `Microscope` driver and coordinate arrays or use the class from a separate executable file which sets this up for your microscope driver.

## Step 2: Contour Tracking

The tracker (`CorrelationContourTracker`) detects cell contours. It supports `.lif` (LIF), and FFV1-encoded video containers (`.avi`, `.mkv`, `.mp4`).

> [!TIP]
> **Custom Formats**: To run the tracker with other proprietary or custom video formats, you can implement and register a custom reader conforming to the `MovieReader` Protocol. See the [Registering Custom Movie Readers](#1-registering-custom-movie-readers-for-different-formats) section below for instructions and code examples.

### Programmatic Contour Tracking

Contours are loaded and saved using `ContourIO` in `.npz` format. They can be saved as xy or center and radial coordinates:

```python
from flickering.utils.movie_reader import get_movie_reader
from flickering.tracking.correlation_tracker import CorrelationContourTracker as CCT
from flickering.tracking.contour_io import ContourIO

# 1. Load the video file (supports registered formats)
movie = get_movie_reader("/path/to/cell_recording.mp4")

# 2. Initialize the tracker
tracker = CCT()

# 3. Track the contours
results = tracker.track_movie(movie)

# 4. Save contours to NPZ format using ContourIO
cio = ContourIO()
cio.contours = results["contours"]
cio.valid_indices = results["valid_indices"]
cio.write("/path/to/contours")  # Writes /path/to/contours.npz
```

### Batch Directory Tracking (CLI)

To batch-process a directory of videos, use the command-line driver. Use `--existing`, `--recurse`, and `--nofollow` to scan subdirectories, process all existing files, and exit. The script without a no-follow flag can track a directory and automatically track any new files (this is restricted to single folder and may not work on network drives - check the script and help for details):

```bash
python3 -m flickering.tracking.contour_folder /path/to/movies_folder/ --threads 8 --existing --recurse --nofollow
```

---

## Step 3: Contour analysis, FT and Fitting

Parameters like membrane tension ($\sigma$) and bending modulus ($\kappa$) are extracted by calculating the fluctuation power spectrum from contours and fitting it to theoretical models.

### Parameter Extraction Example

To process a single cell's contours, initialize the configuration using the standard helper `default_fitter()` or configure it directly:

```python
from flickering.tracking.contour_io import ContourIO
from flickering.utils.standard_configs import default_fitter

# 1. Load the tracked contours (.npz format)
cio = ContourIO("/path/to/contours.npz")

# 2. Instantiate the fitter using standard configurations
cf = default_fitter()
cf.delay_between_frames_ms = 10.0  # Configure frame delay (10ms = 100 fps)
cf.exposure_time_ms = 0.8          # Configure shutter speed (800 us)
# (Note: Other parameters like temperatures, viscosity, and fitting ranges
# are available and pre-configured in default_fitter)

# 3. Run the processing pipeline
results = cf.process_contours(cio.contours, cio.valid_indices)

# 4. Access fitted physical parameters directly at the top level
tension = results["sigma"]["value"]          # Tension (units of 10^-7 N/m)
bending_modulus = results["kappa"]["value"]  # Bending rigidity (units of 10^-21 J)

print(f"Tension (sigma): {tension:.2f} x 10^-7 N/m")
print(f"Bending Rigidity (kappa): {bending_modulus:.2f} x 10^-21 J")
```

### Batch Processing of Folders

For full experimental folder structures (as produced by the automated imaging), the `process_folders` utility processes all files in parallel, manages caching, and merges results into a unified `pandas` DataFrame. This uses the cell log with all the metadata saved during the automated imaging. This script is designed so it also shows the results in a plot for easy checking.

```python
import pandas as pd
from flickering.utils.process_multiwell import process_folders
from flickering.utils.standard_configs import default_fitter

# 1. Define paths and experiment metadata
folders = ["/data/run_ga_0.00%/", "/data/run_ga_0.01%/"]
labels = ["Control", "0.01% GA"]
well_labels = [["A1", "A2"], ["B1", "B2"]]
fit_starts = [6, 6]  # Starting Fourier mode
fit_ends = [26, 26]  # Ending Fourier mode
temperatures = [37.0, 37.0]

# 2. Run batch processing
df, wells, processed_log, metadata = process_folders(
    folders=folders,
    folder_labels=labels,
    well_labelss=well_labels,
    fit_starts=fit_starts,
    fit_ends=fit_ends,
    temperatures_c=temperatures,
    fitter=default_fitter(),
    reprocess=False
)

# 3. Save combined results
df.to_csv("/data/combined_results.csv", index=False)
```

---

## Package Structure

The codebase is organized as follows:

### `flickering/` (Core Package)
Contains the main logic and experimental pipelines.
- `analysis/`: Scientific analysis logic (`fitter.py`).
- `tracking/`: Vision and boundary tracking (`correlation_tracker.py`, `contour_io.py`).
- `acquisition/`: Automation orchestration (`autoimager.py`, `multifocus.py`, `microscopes/`).
- `experiments/`: Dedicated protocols (`ZstackExperiment.py`, `MultiWellExperiment.py`, etc.).
- `simulations/`: High-performance simulating (`simulator.py`, `real_decay_sims.py`).
- `utils/`: Encoders, configs, visualisations, file readers, and constants.


---

## Customizing and Extending Hardware & Video Formats

To enable running the automated cell acquisition and tracking tools with custom hardware setups or proprietary movie formats, you can implement the interfaces/protocols defined in `flickering` using Python's structural subtyping (protocols).

### 1. Registering Custom Movie Readers for Different Formats

The tracking and analysis pipelines are format-agnostic. They use `flickering.utils.movie_reader.get_movie_reader` to open files. By default, readers are registered for `.lif`, `.avi`, `.mkv`, and `.mp4`.

To add a new reader for a different video format (e.g. `.czi`, `.tif`, or a proprietary format):

1. Define a class that implements the `MovieReader` Protocol (defined in [movie_reader.py](flickering/utils/movie_reader.py)):
   ```python
   from typing import Generator, Iterable, Union, Type
   from pathlib import Path
   import numpy as np
   from flickering.utils.movie_reader import register_movie_reader

   class CustomMovieReader:
       def __init__(self, filename: Union[str, Path], preload_to_memory: bool = False, *args, **kwargs):
           self.filename = filename
           # Initialize your reader here
           self.n_frames = 1000 # Set the total frames
           self.size_x = 512    # Frame width (optional)
           self.size_y = 512    # Frame height (optional)

       def get_frame(self, i: int) -> np.ndarray:
           # Read and return the i-th frame as a 2D numpy array
           ...

       def frames(self, *args) -> Union[Generator[np.ndarray, None, None], Iterable[np.ndarray]]:
           # Yield frames sequentially
           for i in range(self.n_frames):
               yield self.get_frame(i)

       def destroy(self) -> None:
           # Close any open file handles or release resources
           pass
   ```

2. Register your reader with the global registry:
   ```python
   register_movie_reader(".czi", CustomMovieReader)
   ```

Now, any calls to `get_movie_reader("path/to/movie.czi")` in tracking or analysis scripts will automatically instantiate and use your custom reader.

### 2. Implementing Custom Microscope Interfaces

The automated acquisition scripts in `flickering.acquisition` (such as [autoimager.py](flickering/acquisition/autoimager.py) and [multifocus.py](flickering/acquisition/multifocus.py)) are decoupled from specific microscope drivers and rely on the `Microscope` protocol.

To use custom hardware (e.g., custom stages, focus controllers, or cameras):

1. Create a class that implements the `Microscope` Protocol defined in [microscope.py](flickering/acquisition/microscope.py):
   ```python
   from flickering.acquisition.microscope import Microscope
   import numpy as np

   class MyCustomMicroscope:
       camera_name = "MyCustomCamera"
       _z = 0.0

       def get_stage_position(self, update: bool = True) -> np.ndarray:
           # Return stage position as [x, y] in microns
           ...

       def get_z(self, update: bool = True) -> float:
           # Return autofocus/z-stage position in microns
           ...

       def move_stage(self, move_by: np.ndarray, absolute: bool = False, speed: float = 1000.0) -> bool:
           # Move stage
           ...

       def get_image(self, trigger: bool = True) -> np.ndarray:
           # Capture a single frame from the camera
           ...

       # Implement other protocol methods (move_z, pfs_focus, configure_camera, etc.)
   ```

2. Instantiate your custom microscope object and pass it to the automated experiments:
   ```python
   from flickering.acquisition.autoimager import RepeatsMultiWellCellFindingExperiment

   microscope = MyCustomMicroscope()
   experiment = RepeatsMultiWellCellFindingExperiment(microscope=microscope, ...)
   ```

> [!NOTE]
> Temika-specific features like using a temporary file for autofocus zstack will not work without Temika based microscopes - implement a custom Preprocessor if needed for your setup.