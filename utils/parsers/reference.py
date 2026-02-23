import logging
from pathlib import Path
from typing import cast

import numpy as np
from cclib.io.ccio import ccread

from .core import BaseReferenceParser, ReferenceData, register_reference

logger = logging.getLogger(__name__)


@register_reference(priority=10)
class FChkReferenceParser(BaseReferenceParser):
    @classmethod
    def is_applicable(cls, file_path: Path) -> bool:
        """
        Checks if the given file path is applicable to this parser.

        Parameters
        ----------
        file_path : Path
            Path to the file to be checked.

        Returns
        -------
        bool
            True if the file path is applicable to this parser, False otherwise.
        """
        return file_path.suffix.lower() == ".fchk"

    def parse(self) -> ReferenceData:  # noqa: C901
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
        gradient_1d: np.ndarray | None = None
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
                elif line.startswith("Cartesian Gradient"):
                    n_items = int(line.split("N=")[1].strip())
                    gradient_1d = self._read_array(f, n_items, float)
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

        gradient = None
        if gradient_1d is not None:
            gradient = gradient_1d.reshape(-1, 3)

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
            gradient=gradient,
            hessian=hessian,
            coords_unit="bohr",
        )

    def _read_array(self, f, n_items: int, dtype: type) -> np.ndarray:
        """
        Reads a list of items of type `dtype` from a file-like object `f` until `n_items` items are read.

        Parameters
        ----------
        f : file-like object
            A file-like object to read from.
        n_items : int
            The number of items to read.
        dtype : type
            The type of the items to read.

        Returns
        -------
        np.ndarray
            A NumPy array containing the read items.

        Notes
        -----
        This function will stop reading when it reaches the end of the file, even if `n_items` items have not been read.
        """  # noqa: E501
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


@register_reference(priority=30)
class GRRMLogReferenceParser(BaseReferenceParser):
    """
    Parser for GRRM log reference files.
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

    def parse(self) -> ReferenceData:
        """
        Parses a GRRM log reference file and returns the reference data.

        Returns
        -------
        ReferenceData
            The parsed reference data.

        Raises
        ------
        NotImplementedError
            GRRM log reference parsing is not yet implemented.
        """
        emsg = "GRRMLogReferenceParser.parse is not implemented yet."
        raise NotImplementedError(emsg)


@register_reference(priority=35)
class GaussianLogReferenceParser(BaseReferenceParser):
    """
    Parser for Gaussian log reference files.
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

    def parse(self) -> ReferenceData:
        """
        Parses a Gaussian log reference file and returns the reference data.

        Returns
        -------
        ReferenceData
            The parsed reference data.

        Raises
        ------
        NotImplementedError
            Gaussian log reference parsing is not yet implemented.
        """
        emsg = "GaussianLogReferenceParser.parse is not implemented yet."
        raise NotImplementedError(emsg)


@register_reference(priority=100)
class CclibReferenceParser(BaseReferenceParser):
    """
    Parser for reference files using cclib (e.g., `.fchk`).
    """

    @classmethod
    def is_applicable(cls, file_path: Path) -> bool:
        """
        Checks if the given file path is applicable to this parser.

        Parameters
        ----------
        file_path : Path
            Path to the file to be checked.

        Returns
        -------
        bool
            True if the file path is applicable to this parser, False otherwise.
        """
        return file_path.suffix.lower() in (".log", ".out", ".xyz")

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

        gradient = None
        if hasattr(data, "grads"):
            gradient = cast("np.ndarray", getattr(data, "grads"))  # noqa: B009

        return ReferenceData(
            atomnos=atomnos,
            masses=atommasses,
            coords=atomcoords[-1],
            gradient=gradient,
            hessian=hessian,
            coords_unit="angstrom",
        )
