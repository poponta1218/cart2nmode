import argparse
import sys
import warnings
from datetime import datetime
from logging import (
    DEBUG,
    INFO,
    FileHandler,
    Formatter,
    NullHandler,
    StreamHandler,
    captureWarnings,
    getLogger,
)
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pint
import polars as pl
from cclib.io.ccio import ccread
from pint.facets.plain import PlainQuantity
from pydantic import BaseModel, ConfigDict, Field

logger = getLogger(__name__)
logger.addHandler(NullHandler())

ureg = pint.UnitRegistry(system="atomic", auto_reduce_dimensions=True)
ureg.enable_contexts("energy", "spectroscopy")

PROJECT_ROOT = Path(__file__).parent.resolve()


class SnapshotMolecule(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    coords: PlainQuantity[Any] = Field(..., description="Coordinates of the atoms in the molecule")

    @classmethod
    def from_file(cls, file_path: Path, input_unit: pint.Unit) -> "SnapshotMolecule":
        """
        Reads a snapshot molecule from a file.

        Parameters
        ----------
        file_path : Path
            Path to the file containing the snapshot molecule.
        input_unit : pint.Unit
            The unit of the coordinates in the file. Must be a unit of length.

        Returns
        -------
        SnapshotMolecule
            The snapshot molecule read from the file.

        Raises
        ------
        ValueError
            If the file does not contain the required attributes.
        """
        logger.debug(f"Reading snapshot molecule from file: {file_path} (Input unit: {input_unit})")

        data = ccread(str(file_path))
        if data is None:
            emsg = f"Failed to parse the snapshot molecule file: {file_path}"
            logger.error(emsg)
            raise ValueError(emsg)

        if not hasattr(data, "atomcoords"):
            emsg = f"Missing required attribute 'atomcoords' in the snapshot molecule file: {file_path}"
            logger.error(emsg)
            raise ValueError(emsg)

        atomcoords = cast("np.ndarray", getattr(data, "atomcoords"))  # noqa: B009
        if len(atomcoords) > 1:
            logger.warning(
                f"Multiple frames found in the snapshot molecule file: {file_path}. "
                f"Only the last frame will be used for projection. "
                f"Consider splitting the file into separate frames if you want to project multiple snapshots."
            )

        coords_q = ureg.Quantity(atomcoords[-1], input_unit).to("bohr")  # TODO(poponta): handle multiple frames
        return cls(coords=coords_q)


class ReferenceMolecule(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    atomnos: np.ndarray = Field(..., description="Atomic numbers of the atoms in the molecule")
    masses: PlainQuantity[Any] = Field(..., description="Masses of the atoms in the molecule")
    coords: PlainQuantity[Any] = Field(..., description="Coordinates of the atoms in the molecule")
    hessian: PlainQuantity[Any] = Field(..., description="Hessian matrix of the molecule")

    eigenvalues: PlainQuantity[Any] | None = Field(None, description="Eigenvalues of the Hessian matrix")
    eigenvectors: np.ndarray | None = Field(None, description="Eigenvectors of the Hessian matrix")

    @classmethod
    def from_file(cls, file_path: Path) -> "ReferenceMolecule":
        """
        Reads a reference molecule from a file.

        Parameters
        ----------
        file_path : Path
            Path to the file containing the reference molecule.

        Returns
        -------
        ReferenceMolecule
            The reference molecule read from the file.

        Raises
        ------
        ValueError
            If the file does not contain the required attributes.
        """
        logger.debug(f"Reading reference molecule from file: {file_path}")

        data = ccread(str(file_path))

        if data is None:
            emsg = f"Failed to parse the reference molecule file: {file_path}"
            logger.error(emsg)
            raise ValueError(emsg)

        required_attrs = ["atomnos", "atommasses", "atomcoords", "hessian"]
        for attr in required_attrs:
            if not hasattr(data, attr):
                emsg = f"Missing required attribute '{attr}' in the reference molecule file: {file_path}"
                logger.error(emsg)
                raise ValueError(emsg)

        atomnos = cast("np.ndarray", getattr(data, "atomnos"))  # noqa: B009
        atommasses = cast("np.ndarray", getattr(data, "atommasses"))  # noqa: B009
        atomcoords = cast("np.ndarray", getattr(data, "atomcoords"))  # noqa: B009
        hessian = cast("np.ndarray", getattr(data, "hessian"))  # noqa: B009

        masses_q = ureg.Quantity(atommasses, "amu")
        coords_q = ureg.Quantity(atomcoords[-1], "angstrom").to("bohr")  # TODO(poponta): handle multiple frames
        hessian_q = ureg.Quantity(hessian, "hartree / bohr**2")

        return cls(
            atomnos=atomnos,
            masses=masses_q,
            coords=coords_q,
            hessian=hessian_q,
            eigenvalues=None,
            eigenvectors=None,
        )

    def model_post_init(self, _context: Any) -> None:
        """
        Post-initialization hook for the ReferenceMolecule model.

        Diagonalizes the mass-weighted Hessian matrix to obtain eigenvalues and eigenvectors.

        This function is called automatically after the model is initialized.
        """
        self._diag_mw_hessian()

    def _diag_mw_hessian(self) -> None:
        """
        Diagonalizes the mass-weighted Hessian matrix to obtain eigenvalues and eigenvectors.

        The mass-weighted Hessian matrix is constructed by repeating the masses along the diagonal and taking the square root inverse.
        The resulting matrix is then diagonalized using NumPy's `eigh` function to obtain the eigenvalues and eigenvectors.

        The eigenvalues are converted to units of "hartree / (amu * bohr**2)" and stored in the `eigenvalues` attribute.
        The eigenvectors are stored in the `eigenvectors` attribute.
        """  # noqa: E501
        logger.debug("Diagonalizing the mass-weighted Hessian matrix to obtain eigenvalues and eigenvectors.")
        m_reperted = np.repeat(self.masses.magnitude, 3)
        inv_sqrt_m = 1 / np.sqrt(m_reperted)
        mw_hessian = self.hessian.magnitude * inv_sqrt_m[:, np.newaxis] * inv_sqrt_m[np.newaxis, :]

        evals, evecs = np.linalg.eigh(mw_hessian)
        self.eigenvalues = ureg.Quantity(evals, "hartree / (amu * bohr**2)")
        self.eigenvectors = evecs

        logger.debug(f"Diagonalization complete. Found {len(evals)} eigenvalues and eigenvectors.")

    def reduce_trans_rot(self) -> None:
        """
        Reduces the number of vibrational modes by identifying and removing translational and rotational modes.

        This function assumes that the eigenvalues and eigenvectors have already been computed.

        Raises
        ------
        ValueError
            If the eigenvalues and eigenvectors have not been computed.
        UserWarning
            If the expected number of full modes is not found.
        """
        if self.eigenvalues is None or self.eigenvectors is None:
            emsg = "Eigenvalues and eigenvectors have not been computed."
            raise ValueError(emsg)

        natom = len(self.atomnos)

        current_dim = len(self.eigenvalues.magnitude)
        expected_full_dim = 3 * natom
        logger.debug("Attempting to reduce translational and rotational modes from the reference molecule.")
        if current_dim < expected_full_dim:
            wmsg = (
                f"Expected {expected_full_dim} eigenvalues for a full set of modes, but only found {current_dim}. "
                f"This suggests that the modes may have already been reduced. "
                f"Skipping reduction of translational and rotational modes."
            )
            logger.warning(wmsg)
            warnings.warn(
                wmsg,
                UserWarning,
                stacklevel=2,
            )
            return

        if natom == 1:
            n_trans_rot = 3
        elif natom == 2:
            n_trans_rot = 5
        else:
            n_trans_rot = 6

        logger.debug(
            f"Identifying {n_trans_rot} translational and rotational modes to reduce based on absolute eigenvalue magnitudes."  # noqa: E501
        )

        idx = np.argsort(np.abs(self.eigenvalues.magnitude))
        vib_idx = np.sort(idx[n_trans_rot:])

        eigenvalues = self.eigenvalues.magnitude[vib_idx]
        self.eigenvalues = ureg.Quantity(eigenvalues, self.eigenvalues.units)
        self.eigenvectors = self.eigenvectors[:, vib_idx]

        logger.debug("Reduction completed.")

    def frequency(self) -> PlainQuantity[Any]:
        """
        Computes the vibrational frequencies of the molecule.

        Returns
        -------
        PlainQuantity[Any]
            The vibrational frequencies of the molecule in cm**-1.

        Raises
        ------
        ValueError
            If the eigenvalues have not been computed.
        """
        if self.eigenvalues is None:
            emsg = "Eigenvalues have not been computed. Cannot compute frequencies."
            logger.error(emsg)
            raise ValueError(emsg)

        vals = self.eigenvalues.magnitude
        sign = np.sign(vals)
        omega_signed = np.sqrt(np.abs(vals)) * sign
        nu = omega_signed / (2 * np.pi)
        return ureg.Quantity(nu, "(hartree / (amu * bohr**2))**0.5").to("cm**-1")


class Projector:
    def __init__(self, ref: ReferenceMolecule, output_unit: pint.Unit, kabsch_threshold: float) -> None:
        """
        Initializes a Projector object.

        Parameters
        ----------
        ref : ReferenceMolecule
            The reference molecule to project the snapshot molecule onto.
        output_unit : pint.Unit
            The unit to output the projected coordinates in. Must be a unit of length.
        kabsch_threshold : float
            The threshold for the Kabsch algorithm to consider two molecules aligned. Must be between 0 and 1.

        Raises
        ------
        ValueError
            If the reference molecule does not have eigenvalues and eigenvectors computed.
        """
        if ref.eigenvalues is None or ref.eigenvectors is None:
            emsg = "Reference molecule must have eigenvalues and eigenvectors computed."
            logger.error(emsg)
            raise ValueError(emsg)

        self.ref = ref
        self.output_unit = ureg.Unit("amu**0.5") * output_unit
        self.ref_cent = np.average(self.ref.coords.magnitude, weights=self.ref.masses.magnitude, axis=0)
        self.ref_centered = self.ref.coords.magnitude - self.ref_cent
        self.kabsch_threshold = kabsch_threshold

        logger.debug(f"Projector initialized. Output unit: {self.output_unit}.")

    def project(self, snapshot: SnapshotMolecule) -> PlainQuantity[Any]:
        """
        Projects a snapshot molecule onto the normal modes of the reference molecule.

        Parameters
        ----------
        snapshot : SnapshotMolecule
            The molecule to be projected onto the normal modes of the reference molecule.

        Returns
        -------
        nmode_coords_q : PlainQuantity[Any]
            The projected coordinates in the normal mode basis, with units of "amu**0.5 * bohr".

        Notes
        ----
        The function first aligns the snapshot molecule to the reference molecule using the Kabsch algorithm.
        It then computes the difference between the aligned snapshot molecule and the reference molecule.
        The difference is then weighted by the square root of the atomic masses and projected onto the normal modes of the reference molecule.
        The resulting coordinates are expressed in units of "amu**0.5 * bohr".
        If the RMSD of the Kabsch alignment exceeds the threshold set by the `kabsch_threshold` attribute, a warning is raised.
        """  # noqa: E501
        logger.debug("Projecting snapshot coordinates onto the normal modes of the reference molecule.")

        if self.ref.eigenvectors is None:
            emsg = "Reference molecule must have eigenvectors computed."
            logger.error(emsg)
            raise ValueError(emsg)

        natom_ref = len(self.ref.atomnos)
        natom_curr = len(snapshot.coords.magnitude)

        if natom_curr != natom_ref:
            emsg = f"Number of atoms in snapshot ({natom_curr}) does not match reference ({natom_ref})."
            logger.error(emsg)
            raise ValueError(emsg)

        curr_cent = np.average(snapshot.coords.magnitude, weights=self.ref.masses.magnitude, axis=0)
        curr_centered = snapshot.coords.magnitude - curr_cent

        weights = self.ref.masses
        cov_mat = (curr_centered.T * weights.magnitude) @ self.ref_centered
        U, _S, V_t = np.linalg.svd(cov_mat)  # noqa: N806
        rot_mat = V_t.T @ U.T
        if np.linalg.det(rot_mat) < 0:
            V_t[-1, :] *= -1
            rot_mat = V_t.T @ U.T

        curr_aligned = curr_centered @ rot_mat

        diff = curr_aligned - self.ref_centered
        weighted_sq_diff = np.sum((diff**2) * weights.magnitude[:, np.newaxis], axis=1)
        rmsd = np.sqrt(np.sum(weighted_sq_diff) / np.sum(weights.magnitude))

        logger.info(f"Kabsch Alignment RMSD: {rmsd:.3f} bohr")

        if rmsd > self.kabsch_threshold:
            wmsg = (
                f"RMSD after Kabsch alignment is {rmsd:.3f} bohr, "
                f"which exceeds the threshold of {self.kabsch_threshold:.3f} bohr. "
                f"Projection results may be unreliable. "
                f"Consider increasing the threshold or checking the input structures."
            )
            logger.warning(wmsg)
            warnings.warn(wmsg, UserWarning, stacklevel=2)

        mass_mat = np.repeat(self.ref.masses.magnitude, 3)
        nmode_mw_coefs = diff.flatten() * np.sqrt(mass_mat)

        nmode = self.ref.eigenvectors
        nmode_coords = nmode.T @ nmode_mw_coefs
        nmode_coords_q = ureg.Quantity(nmode_coords, "amu**0.5 * bohr")
        logger.debug("Projection complete.")
        return nmode_coords_q.to(self.output_unit)

    def to_csv(self, csv_path: Path, nmode_coords: PlainQuantity[Any], *, align: bool) -> None:
        """
        Saves projected normal mode coordinates to a CSV file.

        Parameters
        ----------
        csv_path : Path
            Path to the output CSV file.
        nmode_coords : PlainQuantity[Any]
            Projected normal mode coordinates.
        align : bool, optional
            Aligns the columns in the output CSV by padding with spaces, by default False.
        """
        logger.debug(f"Saving projected normal mode coordinates to CSV file: {csv_path}")

        if self.ref.eigenvalues is None:
            emsg = "Reference molecule must have eigenvalues computed."
            logger.error(emsg)
            raise ValueError(emsg)

        if not hasattr(self.ref, "frequency"):
            emsg = "Reference molecule must have a method to compute frequencies."
            logger.error(emsg)
            raise ValueError(emsg)

        natoms = len(self.ref.atomnos)
        n_full_modes = 3 * natoms
        n_skipped = n_full_modes - len(self.ref.eigenvalues.magnitude)

        mode_idx = np.arange(len(self.ref.eigenvalues.magnitude))
        frequency = self.ref.frequency().magnitude
        if n_skipped > 0:
            if frequency[0] < 0:
                mode_idx[1:] += n_skipped
            else:
                mode_idx += n_skipped + 1

        df = pl.DataFrame(
            {
                "mode": mode_idx,
                "frequency": frequency,
                "displacement": nmode_coords.magnitude,
            }
        )

        logger.debug(f"Dataframe created with {len(df)} rows.")
        logger.debug(f"Aligning columns in the output CSV: {align}.")

        if align:
            df_formatted = df.select(
                pl.col("mode").map_elements(lambda x: f"{x:>3d}"),
                pl.col("frequency").map_elements(lambda x: f"{x:>9.2f}"),
                pl.col("displacement").map_elements(lambda x: f"{x:>13.4E}"),
            )
        else:
            df_formatted = df.select(
                pl.col("mode").map_elements(lambda x: f"{x:d}"),
                pl.col("frequency").map_elements(lambda x: f"{x:.2f}"),
                pl.col("displacement").map_elements(lambda x: f"{x:.4E}"),
            )
        df_formatted.write_csv(csv_path, quote_style="never")

        logger.info(f"CSV file saved successfully: {csv_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the program.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Project Cartesian coordinates onto normal modes")
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        help="Path to the reference molecule file (e.g., FChk file with Hessian)",
    )
    parser.add_argument(
        "-s",
        "--snapshot",
        type=Path,
        help="Path to the snapshot molecule file (e.g., XYZ)",
    )
    parser.add_argument(
        "-u",
        "--input-unit",
        type=str,
        default="angstrom",
        choices=["angstrom", "bohr", "a_u_length"],
        help="Unit of the snapshot coordinates (default: angstrom)",
    )
    parser.add_argument(
        "-U",
        "--output-unit",
        type=str,
        default="bohr",
        choices=["angstrom", "bohr", "a_u_length"],
        help="Unit for the output normal mode coordinates in (amu)**0.5 XXX (default: bohr)",
    )
    parser.add_argument(
        "-o",
        "--output-csv-name",
        type=Path,
        help="Path to the output CSV file for normal mode coordinates",
    )
    parser.add_argument(
        "-a",
        "--align-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to align columns in the output CSV for better readability (default: True)",
    )
    parser.add_argument(
        "-k",
        "--kabsch-threshold",
        type=float,
        default=3.0,
        help="RMSD threshold in bohr for Kabsch alignment warning (default: 3.0 bohr)",
    )
    parser.add_argument(
        "-R",
        "--reduce-trans-rot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to reduce translational and rotational modes from the reference (default: True)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    return parser.parse_args()


def validate_length_unit(unit_str: str) -> pint.Unit:
    """
    Validates that the provided unit is a unit of length.

    Parameters
    ----------
    unit : pint.Unit
        The unit to validate.

    Returns
    -------
    pint.Unit
        The validated unit if it is a unit of length.

    Raises
    ------
    ValueError
        If the provided unit is not a unit of length.
    """
    try:
        unit = ureg.parse_units(unit_str)
    except pint.errors.UndefinedUnitError as e:
        emsg = f"Unknown unit: {unit_str}."
        logger.exception(emsg)
        raise ValueError(emsg) from e

    if unit.dimensionality != ureg.Unit("meter").dimensionality:
        emsg = f"Invalid unit: {unit_str}. The unit must be a unit of length (e.g., 'angstrom', 'bohr')."
        logger.error(emsg)
        raise ValueError(emsg)
    return unit


def setup_logging(*, verbose: bool) -> Path | None:
    """
    Set up logging for the script.

    Parameters
    ----------
    verbose : bool
        Whether to enable verbose logging.

    Returns
    -------
    log_file : Path | None
        The path to the log file if a log file is created, otherwise None.
    """
    timestamp = datetime.now(tz=ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT.joinpath("log")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir.joinpath(f"{timestamp}.log")

    root_logger = getLogger()
    if root_logger.hasHandlers():
        return None

    formatter = Formatter("[%(asctime)s][%(levelname)s] %(name)s: %(message)s")
    root_logger.setLevel(DEBUG)

    console_level = DEBUG if verbose else INFO
    console_handler = StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    captureWarnings(capture=True)
    return log_file


def main():
    args = parse_args()
    logname = setup_logging(verbose=args.verbose)

    logger.info("Projecting Cartesian coordinates onto normal modes")
    logger.debug(f"Arguments: {args}")

    if args.reference is None or args.snapshot is None:
        logger.error("Both --reference and --snapshot arguments are required.")
        sys.exit(1)

    try:
        input_unit = validate_length_unit(args.input_unit)
        output_unit = validate_length_unit(args.output_unit)

        ref = ReferenceMolecule.from_file(file_path=args.reference)
        snap = SnapshotMolecule.from_file(file_path=args.snapshot, input_unit=input_unit)

        if args.reduce_trans_rot:
            ref.reduce_trans_rot()
        else:
            logger.info("Skipping reduction of translational and rotational modes as per user request.")

        projector = Projector(ref=ref, output_unit=output_unit, kabsch_threshold=args.kabsch_threshold)
        nmode_coords = projector.project(snap)

        if args.output_csv_name is not None:
            if args.output_csv_name.is_absolute() or len(args.output_csv_name.parts) > 1:
                csv_path = args.output_csv_name
            else:
                csv_dir = PROJECT_ROOT.joinpath("data/csv")
                csv_path = csv_dir.joinpath(args.output_csv_name.name)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            projector.to_csv(csv_path=csv_path, nmode_coords=nmode_coords, align=args.align_csv)

    except Exception:
        logger.exception("An unexpected error occurred")
        sys.exit(1)

    if logname:
        logger.info(f"Log saved to: {logname}")

    logger.info("Projection completed successfully.")


if __name__ == "__main__":
    main()
