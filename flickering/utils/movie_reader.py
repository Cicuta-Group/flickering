from __future__ import annotations
from typing import Protocol, runtime_checkable, Generator, Union, Iterable, Any, Dict, Type
from pathlib import Path
import numpy as np

@runtime_checkable
class MovieReader(Protocol):
    filename: Union[str, Path]
    n_frames: int

    def get_frame(self, i: int) -> np.ndarray:
        """Returns the i-th frame as a numpy array."""
        ...

    def frames(self, *args) -> Union[Generator[np.ndarray, None, None], Iterable[np.ndarray]]:
        """Returns an iterable or generator of frames."""
        ...

    def destroy(self) -> None:
        """Destroys the reader and closes any open resources."""
        ...

# Extension registry mapping lowercase extension strings (e.g. '.lif') to reader classes
_movie_reader_registry: Dict[str, Type[MovieReader]] = {}

def register_movie_reader(extension: str, reader_cls: Type[MovieReader]) -> None:
    """Register a movie reader class for a specific file extension (e.g. '.lif')."""
    _movie_reader_registry[extension.lower()] = reader_cls

def get_movie_reader(movie_source: Union[str, Path, Any], **kwargs) -> MovieReader:
    """
    Factory function to retrieve a MovieReader instance for the given movie source.
    If movie_source is a string or Path, matches the extension using the registry.
    If movie_source is already a MovieReader (duck-typed), returns it directly.
    """
    if isinstance(movie_source, (str, Path)):
        filename = str(movie_source)
        ext = Path(filename).suffix.lower()
        
        if ext in _movie_reader_registry:
            return _movie_reader_registry[ext](filename, **kwargs)
        
        if ext == ".movie":
            # Lazily load Temika movie reader to avoid hard dependency on temika package
            try:
                from temika.movie_reader import Movie
                return Movie(filename, **kwargs)
            except ImportError as e:
                raise ImportError(
                    "The 'temika' package is required to read .movie files. "
                    "Please install the temika sub-package (pip install -e ./temika)."
                ) from e
                
        raise ValueError(
            f"Unsupported movie format: {ext}. "
            f"Registered extensions: {list(_movie_reader_registry.keys())} + ['.movie']"
        )
        
    return movie_source
