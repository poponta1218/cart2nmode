from abc import ABC, abstractmethod
from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

import numpy as np

logger = getLogger(__name__)


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
        self.file_path: Path = file_path

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
        -------
        np.ndarray
            A 3D NumPy array containing the coordinates of each atom in the current frame.
        """


class XYZParser(BaseTrajectoryParser):
    """
    Parser for XYZ trajectory files.
    """

    def __init__(self, file_path: Path):
        """
        Initializes an XYZParser object.

        Parameters
        ----------
        file_path : Path
            Path to the XYZ trajectory file to be parsed.

        Attributes
        ----------
        _num_frames : int | None
            The number of frames in the trajectory.
        """
        super().__init__(file_path)
        self._num_frames: int | None = None

    def __len__(self) -> int:
        """
        Returns the number of frames in the XYZ trajectory file.

        If the number of frames has already been calculated, it is returned immediately.
        Otherwise, the number of frames is calculated by counting the number of newline characters in the file and dividing by the number of lines per frame (natoms + 2).

        Returns
        -------
        int
            Number of frames in the XYZ trajectory file.

        Raises
        ------
        ValueError
            If the number of frames is 0 or cannot be determined.
        """  # noqa: E501
        if self._num_frames is not None:
            return self._num_frames

        logger.debug(f"Calculating number of frames in XYZ file: {self.file_path}")

        with self.file_path.open(mode="r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                self._num_frames = 0
                return self._num_frames

            natoms = int(first_line)

        lines_per_frame = natoms + 2

        total_lines = 0
        with self.file_path.open(mode="rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                total_lines += chunk.count(b"\n")

        self._num_frames = total_lines // lines_per_frame
        logger.debug(f"Found {self._num_frames} frames in total.")

        if self._num_frames == 0:
            raise ValueError(self.file_path, "No frames found in XYZ file")

        if self._num_frames is None:
            raise ValueError(self.file_path, "Could not determine number of frames in XYZ file")

        return self._num_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterates over the frames in the XYZ trajectory file.

        Yields each frame as a 2D NumPy array with shape (natoms, 3), where each row represents the coordinates of an atom in the frame.

        Raises
        ------
        ValueError
            If the XYZ file contains invalid or malformed data (e.g., non-numeric atom coordinates, missing/extra lines, etc.).
        """  # noqa: E501
        logger.debug(f"Iterating over XYZ file: {self.file_path}")

        with self.file_path.open(mode="r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    num_atoms = int(line)
                except ValueError as e:
                    emsg = f"Invalid XYZ format. Expected number of atoms, got: {line}"
                    logger.exception(emsg)
                    raise ValueError(emsg) from e

                _comment = f.readline()

                coords = np.zeros((num_atoms, 3), dtype=np.float64)
                for i in range(num_atoms):
                    atom_line = f.readline().strip().split()
                    if len(atom_line) < 4:
                        emsg = f"Invalid XYZ format. Expected at least 4 columns, got: {atom_line}"
                        logger.error(emsg)
                        raise ValueError(emsg)

                    try:
                        coords[i] = [float(coord) for coord in atom_line[1:4]]
                    except ValueError as e:
                        emsg = f"Invalid coordinate format in XYZ file: {atom_line[1:4]}"
                        logger.exception(emsg)
                        raise ValueError(emsg) from e

                yield coords


def get_trajectory_parser(file_path: Path) -> BaseTrajectoryParser:
    """
    Returns a trajectory parser object based on the file extension.

    Parameters
    ----------
    file_path : Path
        Path to the trajectory file to be parsed.

    Returns
    -------
    BaseTrajectoryParser
        A trajectory parser object that can be used to parse the trajectory file.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = file_path.suffix.lower()
    logger.debug(f"Getting trajectory parser for file: {file_path} with extension: {ext}")

    match ext:
        case ".xyz":
            return XYZParser(file_path)
        case _:
            emsg = f"Unsupported file format: {file_path.suffix}"
            logger.error(emsg)
            raise ValueError(emsg)
