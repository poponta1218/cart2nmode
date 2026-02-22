from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import cast

import numpy as np
from cclib.io.ccio import ccread

logger = getLogger(__name__)


@dataclass
class ReferenceData:
    """
    Data class to hold reference information extracted from the reference file (e.g., `.fchk`).
    """

    atomnos: np.ndarray
    masses: np.ndarray
    coords: np.ndarray
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

    @abstractmethod
    def parse(self) -> ReferenceData:
        """
        Parses the reference file and extracts necessary data for normal mode analysis.

        Returns
        -------
        ReferenceData
            A ReferenceData object containing atom numbers, masses, coordinates, vibrational frequencies, and displacements.
        """  # noqa: E501


class CclibReferenceParser(BaseReferenceParser):
    """
    Parser for reference files using cclib (e.g., `.fchk`).
    """

    def parse(self) -> ReferenceData:
        """
        Parses the reference file using cclib and extracts necessary data for normal mode analysis.

        Returns
        -------
        ReferenceData
            A ReferenceData object containing atom numbers, masses, coordinates, vibrational frequencies, and displacements.

        Raises
        ------
        ValueError
            If the reference file cannot be parsed or does not contain the required data.
        """  # noqa: E501
        logger.debug(f"Parsing reference file with cclib: {self.file_path}")

        data = ccread(str(self.file_path))

        if data is None:
            emsg = f"Failed to parse the reference molecule file: {self.file_path}"
            raise ValueError(emsg)

        required_attrs = ["atomnos", "atommasses", "atomcoords", "hessian"]
        for attr in required_attrs:
            if not hasattr(data, attr):
                emsg = f"Missing required attribute '{attr}' in the reference molecule file: {self.file_path}"
                raise ValueError(emsg)

        atomnos = cast("np.ndarray", getattr(data, "atomnos"))  # noqa: B009
        atommasses = cast("np.ndarray", getattr(data, "atommasses"))  # noqa: B009
        atomcoords = cast("np.ndarray", getattr(data, "atomcoords"))  # noqa: B009
        hessian = cast("np.ndarray", getattr(data, "hessian"))  # noqa: B009

        return ReferenceData(
            atomnos=atomnos,
            masses=atommasses,
            coords=atomcoords[-1],
            hessian=hessian,
            coords_unit="angstrom",
        )


class FChkReferenceParser(BaseReferenceParser):
    def parse(self) -> ReferenceData:
        """
        Parses the reference file using a manual parser for `.fchk` files and extracts necessary data for normal mode analysis.

        Returns
        -------
        ReferenceData
            A ReferenceData object containing atom numbers, masses, coordinates, vibrational frequencies, and displacements.

        Raises
        ------
        ValueError
            If the reference file cannot be parsed or does not contain the required data.
        """  # noqa: E501
        logger.debug(f"Parsing reference file with manual FChk parser: {self.file_path}")

        atomnos: np.ndarray | None = None
        coords_1d: np.ndarray | None = None
        masses: np.ndarray | None = None
        hessian_1d: np.ndarray | None = None

        with self.file_path.open(mode="r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if line.startswith("Atomic numbers"):
                    n_items = int(line.split("N=")[1].strip())
                    atomnos = self._read_array(f, n_items, int)
                elif line.startswith("Current cartesian coordinates"):
                    n_items = int(line.split("N=")[1].strip())
                    coords_1d = self._read_array(f, n_items, float)
                elif line.startswith("Real atomic weights"):
                    n_items = int(line.split("N=")[1].strip())
                    masses = self._read_array(f, n_items, float)
                elif line.startswith("Cartesian Force Constants"):
                    n_items = int(line.split("N=")[1].strip())
                    hessian_1d = self._read_array(f, n_items, float)

        if not all(var is not None for var in [atomnos, coords_1d, masses, hessian_1d]):
            emsg = f"Failed to parse all required data from the reference file: {self.file_path}"
            raise ValueError(emsg)

        atomnos = cast("np.ndarray", atomnos)
        coords_1d = cast("np.ndarray", coords_1d)
        masses = cast("np.ndarray", masses)
        hessian_1d = cast("np.ndarray", hessian_1d)

        coords = coords_1d.reshape(-1, 3)

        natoms = len(atomnos)
        dim = 3 * natoms
        expected_hessian_size = dim * (dim + 1) // 2
        if len(hessian_1d) != expected_hessian_size:
            emsg = (
                f"Unexpected size of the Hessian data in the reference file: {self.file_path}. "
                f"Expected {expected_hessian_size} elements, got {len(hessian_1d)}."
            )
            raise ValueError(emsg)

        hessian = np.zeros((dim, dim), dtype=np.float64)
        row_idx, col_idx = np.tril_indices(dim)
        hessian[row_idx, col_idx] = hessian_1d
        hessian[col_idx, row_idx] = hessian_1d

        logger.debug(f"Successfully parsed reference data from file: {self.file_path}")

        return ReferenceData(
            atomnos=atomnos,
            masses=masses,
            coords=coords,
            hessian=hessian,
            coords_unit="bohr",
        )

    def _read_array(self, f, n_items: int, dtype: type) -> np.ndarray:
        data = []
        while len(data) < n_items:
            line = f.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            data.extend([dtype(item) for item in line.split()])

        logger.debug(f"Read {len(data)} items, expected {n_items} items.")
        return np.array(data, dtype=dtype)


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

        num_frames = self._count_frames_fast()
        if num_frames is not None:
            logger.debug(f"Fast path successful. Found {num_frames} frames in total.")
            self._num_frames = num_frames
        else:
            msg = (
                f"Fast path failed for XYZ file: {self.file_path} "
                "This may be due to blank lines or malformed frames in the file. "
                "Falling back to line-by-line counting. This may take some time for large files."
            )
            logger.debug(msg)

            self._num_frames = self._count_frames_safe()
            logger.debug(f"Safe parsing successful. Found exactly {self._num_frames} frames.")

        if self._num_frames == 0:
            emsg = f"No frames found in XYZ file: {self.file_path}"
            raise ValueError(emsg)

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
                    raise ValueError(emsg) from e

                _comment = f.readline()

                coords = np.zeros((num_atoms, 3), dtype=np.float64)
                for i in range(num_atoms):
                    atom_line = f.readline().strip().split()
                    if len(atom_line) < 4:
                        emsg = f"Invalid XYZ format. Expected at least 4 columns, got: {atom_line}"
                        raise ValueError(emsg)

                    try:
                        coords[i] = [float(coord) for coord in atom_line[1:4]]
                    except ValueError as e:
                        emsg = f"Invalid coordinate format in XYZ file: {atom_line[1:4]}"
                        raise ValueError(emsg) from e

                yield coords

    def _count_frames_fast(self) -> int | None:
        """
        Counts the number of frames in the XYZ file using a fast method that counts newline characters.

        Returns
        -------
        int | None
            The number of frames in the XYZ file.
            Returns None if the number of frames cannot be determined due to blank lines or malformed frames.

        Raises
        ------
        ValueError
            If the number of frames cannot be determined or is zero.
        """
        with self.file_path.open(mode="r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return None

            try:
                natoms = int(first_line)
            except ValueError:
                return None

        lines_per_frame = natoms + 2

        total_lines = 0
        with self.file_path.open(mode="rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                total_lines += chunk.count(b"\n")

        if total_lines > 0 and total_lines % lines_per_frame == 0:
            return total_lines // lines_per_frame

        return None

    def _count_frames_safe(self) -> int:
        """
        Counts the number of frames in the XYZ file using a safe method that reads each frame and counts them.

        Returns
        -------
        int
            The number of frames in the XYZ file.

        Raises
        ------
        ValueError
            If invalid atom counts are encounterd during parsing.
        """
        frames = 0
        with self.file_path.open(mode="r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    current_natoms = int(line)
                except ValueError as e:
                    emsg = f"Invalid atom count at frame {frames} in {self.file_path}. Got: '{line}'"
                    raise ValueError(emsg) from e

                frames += 1

                for _ in range(current_natoms + 1):
                    f.readline()

        return frames


def get_reference_parser(file_path: Path) -> BaseReferenceParser:
    """
    Returns a reference parser object based on the file extension.

    Parameters
    ----------
    file_path : Path
        Path to the reference file to be parsed.

    Returns
    -------
    BaseReferenceParser
        A reference parser object that can be used to parse the reference file.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = file_path.suffix.lower()
    logger.debug(f"Getting reference parser for file: {file_path} with extension: {ext}")

    match ext:
        case ".fchk":
            return FChkReferenceParser(file_path)
        case ".log" | ".out":
            return CclibReferenceParser(file_path)
        case _:
            emsg = f"Unsupported file format: {file_path.suffix}"
            raise ValueError(emsg)


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
            raise ValueError(emsg)
