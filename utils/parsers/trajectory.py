from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

import numpy as np

from .core import BaseTrajectoryParser, register_trajectory

logger = getLogger(__name__)


@register_trajectory(priority=20)
class XYZTrajectoryParser(BaseTrajectoryParser):
    """
    Parser for XYZ trajectory files.
    """

    @classmethod
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
        return file_path.suffix.lower() == ".xyz"

    def __init__(self, file_path: Path):
        """
        Initializes an XYZTrajectoryParser object.

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
                    if len(atom_line) < 4:  # noqa: PLR2004
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
            If invalid atom counts are encountered during parsing.
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


@register_trajectory(priority=30)
class GRRMLogTrajectoryParser(BaseTrajectoryParser):
    """
    Parser for GRRM log trajectory files.
    """

    @classmethod
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
        if file_path.suffix.lower() not in (".log", ".out"):
            return False

        with file_path.open(mode="r", encoding="utf-8") as f:
            for _ in range(50):
                line = f.readline()
                if "GRRM" in line or "Global Reaction Route Mapping" in line:
                    return True
        return False

    def parse(self) -> Iterator[np.ndarray]:
        """
        Parses a GRRM log trajectory file and yields each frame as a 2D NumPy array with shape (natoms, 3),
        where each row represents the coordinates of an atom in the frame.

        Raises
        ------
        NotImplementedError
            GRRM log trajectory parsing is not yet implemented.
        """
        emsg = "GRRMLogTrajectoryParser.parse is not yet implemented yet."
        raise NotImplementedError(emsg)


@register_trajectory(priority=35)
class GaussianLogTrajectoryParser(BaseTrajectoryParser):
    """
    Parser for Gaussian log trajectory files.
    """

    @classmethod
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
        if file_path.suffix.lower() not in (".log", ".out"):
            return False

        with file_path.open(mode="r", encoding="utf-8") as f:
            for _ in range(50):
                line = f.readline()
                if "Gaussian" in line or "Gaussian, Inc." in line:
                    return True
        return False

    def parse(self) -> Iterator[np.ndarray]:
        """
        Parses a Gaussian log trajectory file and yields each frame as a 2D NumPy array with shape (natoms, 3),
        where each row represents the coordinates of an atom in the frame.

        Raises
        ------
        NotImplementedError
            Gaussian log trajectory parsing is not yet implemented.
        """
        emsg = "GaussianLogTrajectoryParser.parse is not yet implemented yet."
        raise NotImplementedError(emsg)
