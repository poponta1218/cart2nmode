from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REFERENCE_PARSERS: list[tuple[int, type]] = []
TRAJECTORY_PARSERS: list[tuple[int, type]] = []


def register_reference(priority: int = 100) -> Callable:

    def decorator(cls: type) -> type:
        REFERENCE_PARSERS.append((priority, cls))
        REFERENCE_PARSERS.sort(key=lambda x: x[0])
        return cls

    return decorator


def register_trajectory(priority: int = 100) -> Callable:

    def decorator(cls: type) -> type:
        TRAJECTORY_PARSERS.append((priority, cls))
        TRAJECTORY_PARSERS.sort(key=lambda x: x[0])
        return cls

    return decorator


@dataclass
class ReferenceData:
    """
    Data class to hold reference information extracted from the reference file (e.g., `.fchk`).
    """

    atomnos: np.ndarray
    masses: np.ndarray
    coords: np.ndarray
    gradient: np.ndarray | None
    hessian: np.ndarray
    coords_unit: str


class BaseReferenceParser(ABC):
    """
    Base class for reference file parsers.
    """

    def __init__(self, file_path: Path):
        """
        Initializes a BaseReferenceParser object.

        Parameters
        ----------
        file_path : Path
            Path to the reference file to be parsed.

        Raises
        ------
        FileNotFoundError
            If the reference file is not found at the specified path.
        """
        if not file_path.exists():
            emsg = f"Reference file not found: {file_path}"
            raise FileNotFoundError(emsg)

        self.file_path: Path = file_path

    @classmethod
    @abstractmethod
    def is_applicable(cls, file_path: Path) -> bool:
        """
        Determines if this parser is applicable to the given file path.

        Parameters
        ----------
        file_path : Path
            Path to the reference file to be checked.

        Returns
        -------
        bool
            True if this parser can handle the given file, False otherwise.
        """

    @abstractmethod
    def parse(self) -> ReferenceData:
        """
        Parses the reference file and extracts necessary data for normal mode analysis.

        Returns
        -------
        ReferenceData
            A ReferenceData object containing atom numbers, masses, coordinates, vibrational frequencies, and displacements.
        """  # noqa: E501


class BaseTrajectoryParser(ABC):
    """
    Base class for trajectory parsers.
    """

    def __init__(self, file_path: Path):
        """
        Initializes a BaseTrajectoryParser object.

        Parameters
        ----------
        file_path : Path
            Path to the trajectory file to be parsed.
        """
        if not file_path.exists():
            emsg = f"Trajectory file not found: {file_path}"
            raise FileNotFoundError(emsg)

        self.file_path: Path = file_path

    @classmethod
    @abstractmethod
    def is_applicable(cls, file_path: Path) -> bool:
        """
        Determines if this parser is applicable to the given file path.

        Parameters
        ----------
        file_path : Path
            Path to the trajectory file to be checked.

        Returns
        -------
        bool
            True if this parser can handle the given file, False otherwise.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of frames in the trajectory.

        Returns
        -------
        int
            Number of frames in the trajectory.
        """

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate over the trajectory data.

        Yields
        ------
        np.ndarray
            A 2D NumPy array of shape ``(natoms, 3)`` containing the x, y, and z coordinates of each atom in the current frame.
        """  # noqa: E501
