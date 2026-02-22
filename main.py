import argparse
import sys
import warnings
from collections.abc import Iterator
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
from pint.facets.plain import PlainQuantity
from pydantic import BaseModel, ConfigDict, Field

from utils.parsers import get_reference_parser, get_trajectory_parser

logger = getLogger(__name__)
logger.addHandler(NullHandler())

ureg = pint.UnitRegistry(system="atomic", auto_reduce_dimensions=True)
ureg.enable_contexts("energy", "spectroscopy")

PROJECT_ROOT = Path(__file__).parent.resolve()


class ReferenceMolecule(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    atomnos: np.ndarray = Field(..., description="Atomic numbers of the atoms in the molecule")
    masses: PlainQuantity[Any] = Field(..., description="Masses of the atoms in the molecule")
    coords: PlainQuantity[Any] = Field(..., description="Coordinates of the atoms in the molecule")
    gradient: PlainQuantity[Any] | None = Field(None, description="Gradient of the molecule, if available")
    hessian: PlainQuantity[Any] = Field(..., description="Hessian matrix of the molecule")

    is_projected: bool = Field(default=False, description="Whether to project out translation, rotation, and gradient")
    n_zero_modes: int = Field(default=0, description="Number of modes to project out based on zero eigenvalues")

    eigenvalues: PlainQuantity[Any] | None = Field(None, description="Eigenvalues of the Hessian matrix")
    eigenvectors: np.ndarray | None = Field(None, description="Eigenvectors of the Hessian matrix")

    @classmethod
    def from_file(cls, file_path: Path, *, is_projected: bool = False) -> "ReferenceMolecule":
        """
        Reads a reference molecule from a file.

        Parameters
        ----------
        file_path : Path
            Path to the file containing the reference molecule.
        is_projected : bool, optional
            Whether to project out translation, rotation, and gradient modes (default: False).

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

        parser = get_reference_parser(file_path)
        data = parser.parse()

        if data is None:
            emsg = f"Failed to parse the reference molecule file: {file_path}"
            logger.error(emsg)
            raise ValueError(emsg)

        required_attrs = ["atomnos", "masses", "coords", "hessian", "coords_unit"]
        for attr in required_attrs:
            if not hasattr(data, attr):
                emsg = f"Missing required attribute '{attr}' in the reference molecule file: {file_path}"
                logger.error(emsg)
                raise ValueError(emsg)

        atomnos = cast("np.ndarray", getattr(data, "atomnos"))  # noqa: B009
        masses = cast("np.ndarray", getattr(data, "masses"))  # noqa: B009
        coords = cast("np.ndarray", getattr(data, "coords"))  # noqa: B009
        hessian = cast("np.ndarray", getattr(data, "hessian"))  # noqa: B009
        coords_unit = cast("str", getattr(data, "coords_unit"))  # noqa: B009

        gradient = cast("np.ndarray | None", getattr(data, "gradient", None))

        masses_q = ureg.Quantity(masses, "amu")
        coords_q = ureg.Quantity(coords, coords_unit).to("bohr")
        hessian_q = ureg.Quantity(hessian, "hartree / bohr**2")

        gradient_q = None
        if is_projected:
            if gradient is not None:
                gradient_q = ureg.Quantity(gradient, "hartree / bohr")
            else:
                emsg = "Projected Hessian requested but no gradient information found in the reference molecule file. "
                logger.error(emsg)
                raise ValueError(emsg)

        return cls(
            atomnos=atomnos,
            masses=masses_q,
            coords=coords_q,
            gradient=gradient_q,
            hessian=hessian_q,
            is_projected=is_projected,
            n_zero_modes=0,
            eigenvalues=None,
            eigenvectors=None,
        )

    def model_post_init(self, _context: Any) -> None:
        """
        Post-initialization hook for the ReferenceMolecule model.

        Diagonalizes the mass-weighted Hessian matrix to obtain eigenvalues and eigenvectors.

        This function is called automatically after the model is initialized.
        """
        if self.is_projected:
            self._diag_projected_mw_hessian()
        else:
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

    def _diag_projected_mw_hessian(self) -> None:
        """
        Diagonalizes the mass-weighted Hessian matrix after projecting out translation, rotation, and gradient modes.

        This function first identifies the modes to be projected out based on zero eigenvalues and the presence of gradient information.
        It then constructs a projection operator to remove these modes from the mass-weighted Hessian matrix.
        Finally, it diagonalizes the projected mass-weighted Hessian matrix to obtain the eigenvalues and eigenvectors.

        The eigenvalues are converted to units of "hartree / (amu * bohr**2)" and stored in the `eigenvalues` attribute.
        The eigenvectors are stored in the `eigenvectors` attribute.
        """  # noqa: E501
        logger.debug("Projecting out translation, rotation, and gradient directions from the Hessian matrix.")

        natom = len(self.atomnos)
        masses = self.masses.magnitude
        coords = self.coords.magnitude
        hessian = self.hessian.magnitude

        cent = np.average(coords, weights=masses, axis=0)
        coords_cent = coords - cent

        sqrt_m = np.sqrt(masses)
        sqrt_m_repeated = np.repeat(sqrt_m, 3)
        inv_sqrt_m_repeated = 1 / sqrt_m_repeated

        mw_coords = coords_cent * sqrt_m[:, np.newaxis]

        vecs = np.zeros((3 * natom, 7))

        total_mass = np.sum(masses)

        # Translational modes
        vecs[0::3, 0] = sqrt_m * np.ones(natom) / np.sqrt(total_mass)
        vecs[1::3, 1] = sqrt_m * np.ones(natom) / np.sqrt(total_mass)
        vecs[2::3, 2] = sqrt_m * np.ones(natom) / np.sqrt(total_mass)

        # Rotational modes
        vecs[1::3, 3] = -mw_coords[:, 2]
        vecs[2::3, 3] = -mw_coords[:, 1]
        vecs[0::3, 4] = -mw_coords[:, 2]
        vecs[2::3, 4] = mw_coords[:, 0]
        vecs[0::3, 5] = mw_coords[:, 1]
        vecs[1::3, 5] = -mw_coords[:, 0]

        if self.gradient is not None:
            mw_gradient = self.gradient.magnitude.flatten() * inv_sqrt_m_repeated
            mw_grad_norm = np.linalg.norm(mw_gradient)
            if mw_grad_norm > 1e-12:  # noqa: PLR2004
                vecs[:, 6] = mw_gradient / mw_grad_norm

        U, S, _Vt = np.linalg.svd(vecs, full_matrices=False)  # noqa: N806
        rank = np.sum(S > 1e-10)  # noqa: PLR2004
        self.n_zero_modes = int(rank)
        logger.debug(f"Identified {self.n_zero_modes} modes to project out based on SVD of the mode matrix.")

        expected_min_rank = 3 if natom == 1 else (5 if natom == 2 else 6)  # noqa: PLR2004
        if not (expected_min_rank <= rank <= 7):  # noqa: PLR2004
            wmsg = (
                f"Unexpected number of projected modes identified for projection: {rank}. "
                f"Nomally expected between {expected_min_rank} and 7. "
            )
            logger.warning(wmsg)
            warnings.warn(wmsg, UserWarning, stacklevel=2)

        trg_basis = U[:, : self.n_zero_modes]
        trg_proj_mat = trg_basis @ trg_basis.T
        identity_mat = np.eye(3 * natom)
        proj_mat = identity_mat - trg_proj_mat

        mw_hessian = hessian * inv_sqrt_m_repeated[:, np.newaxis] * inv_sqrt_m_repeated[np.newaxis, :]
        mw_hessian_proj = proj_mat.T @ mw_hessian @ proj_mat

        evals, evecs = np.linalg.eigh(mw_hessian_proj)
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

        if self.is_projected:
            n_discard = self.n_zero_modes
        elif natom == 1:
            n_discard = 3
        elif natom == 2:  # noqa: PLR2004
            n_discard = 5
        else:
            n_discard = 6

        logger.debug(
            f"Identifying {n_discard} translational and rotational modes to reduce based on absolute eigenvalue magnitudes."  # noqa: E501
        )

        idx = np.argsort(np.abs(self.eigenvalues.magnitude))
        vib_idx = np.sort(idx[n_discard:])

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


class SnapshotMolecule(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    coords: PlainQuantity[Any] = Field(..., description="Coordinates of the atoms in the molecule")


class SnapshotTrajectory:
    def __init__(self, file_path: Path, input_unit: pint.Unit):
        """
        Initializes a SnapshotTrajectory object.

        Parameters
        ----------
        file_path : Path
            Path to the snapshot trajectory file to be parsed.
        input_unit : pint.Unit
            Unit of coordinates in the snapshot trajectory file (e.g., angstrom, bohr).

        Notes
        ----
        The function creates a trajectory parser object based on the file extension.
        """
        self.file_path = file_path
        self.input_unit = input_unit

        self.parser = get_trajectory_parser(file_path)
        logger.debug(f"Reading snapshot trajectory from file: {file_path} (Input unit: {input_unit})")

    def __len__(self) -> int:
        """
        Returns the number of frames in the snapshot trajectory file.

        Returns
        -------
        int
            Number of frames in the snapshot trajectory file.
        """
        return len(self.parser)

    def __iter__(self) -> Iterator[tuple[int, SnapshotMolecule]]:
        """
        Iterate over the frames in the snapshot trajectory file.

        Yields each frame as a tuple of (`frame_idx`, `SnapshotMolecule`),
        where `frame_idx` is the index of the frame and `SnapshotMolecule` is an object containing the coordinates of the atoms in the frame.

        The coordinates are in units of bohr, regardless of the input unit specified during initialization.
        """  # noqa: E501
        for frame_idx, coords in enumerate(self.parser):
            coords_q = ureg.Quantity(coords, self.input_unit).to("bohr")
            yield frame_idx, SnapshotMolecule(coords=coords_q)


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
        rot_mat = U @ V_t
        if np.linalg.det(rot_mat) < 0:
            V_t[-1, :] *= -1
            rot_mat = U @ V_t

        curr_aligned = curr_centered @ rot_mat

        diff = curr_aligned - self.ref_centered
        weighted_sq_diff = np.sum((diff**2) * weights.magnitude[:, np.newaxis], axis=1)
        rmsd = np.sqrt(np.sum(weighted_sq_diff) / np.sum(weights.magnitude))

        logger.debug(f"Kabsch Alignment RMSD: {rmsd:.3f} bohr")

        if rmsd > self.kabsch_threshold:
            wmsg = (
                f"RMSD after Kabsch alignment is {rmsd:.3f} bohr, "
                f"which exceeds the threshold of {self.kabsch_threshold:.3f} bohr. "
                f"Projection results may be unreliable. "
                f"Consider increasing the threshold or checking the input structures."
            )
            logger.warning(wmsg)
            warnings.warn(wmsg, UserWarning, stacklevel=2)

        mass_mat = np.repeat(weights.magnitude, 3)
        nmode_mw_coefs = diff.flatten() * np.sqrt(mass_mat)

        nmode = self.ref.eigenvectors
        nmode_coords = nmode.T @ nmode_mw_coefs
        nmode_coords_q = ureg.Quantity(nmode_coords, "amu**0.5 * bohr")
        logger.debug("Projection complete.")
        return nmode_coords_q.to(self.output_unit)

    def to_csv(self, csv_path: Path, projections: list[tuple[int, PlainQuantity[Any]]], *, align: bool) -> None:
        """
        Saves projected normal mode coordinates to a CSV file.

        Parameters
        ----------
        csv_path : Path
            Path to the output CSV file.
        projections : list[tuple[int, PlainQuantity[Any]]]
            List of tuples containing frame indices and projected normal mode coordinates.
        align : bool, optional
            Aligns the columns in the output CSV by padding with spaces, by default False.
        """
        logger.debug(f"Saving projected normal mode coordinates to CSV file: {csv_path}")

        if self.ref.eigenvalues is None:
            emsg = "Reference molecule must have eigenvalues computed."
            logger.error(emsg)
            raise ValueError(emsg)

        if not projections:
            emsg = "No projection results to save. The 'projections' list is empty."
            logger.warning(emsg)
            return

        natoms = len(self.ref.atomnos)
        n_full_modes = 3 * natoms
        n_skipped = n_full_modes - len(self.ref.eigenvalues.magnitude)

        eigenvalues = self.ref.eigenvalues.magnitude
        mode_idx = np.arange(len(eigenvalues))
        if n_skipped > 0:
            if eigenvalues[0] < 0:
                mode_idx[1:] += n_skipped
            else:
                mode_idx += n_skipped + 1

        rows = []
        for frame_idx, nmode_coords in projections:
            row_data = {
                "frame": frame_idx,
            }
            for m_idx, disp in zip(mode_idx, nmode_coords.magnitude, strict=True):
                row_data[f"mode_{m_idx}"] = disp
            rows.append(row_data)

        df = pl.DataFrame(rows)
        frame_fmt_width = len(str(len(projections) - 1))

        logger.debug(f"Dataframe created with {len(df)} rows.")
        logger.debug(f"Aligning columns in the output CSV: {align}.")

        if align:
            df_formatted = df.select(
                pl.col("frame").map_elements(lambda x: f"{x:>{frame_fmt_width}d}"),
                *[pl.col(f"mode_{mode}").map_elements(lambda x: f"{x:>13.4E}") for mode in mode_idx],
            )
        else:
            df_formatted = df.select(
                pl.col("frame"),
                *[pl.col(f"mode_{mode}").map_elements(lambda x: f"{x:.4E}") for mode in mode_idx],
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
        help="Path to the output CSV file for normal mode coordinates (default: data/csv/<snapshot_name>-nmode.csv)",
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
        "-p",
        "--projected",
        action="store_true",
        default=False,
        help="Whether to use projected Hessian matrix (default: False)",
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

        ref = ReferenceMolecule.from_file(file_path=args.reference, is_projected=args.projected)
        trajectory = SnapshotTrajectory(file_path=args.snapshot, input_unit=input_unit)

        if args.reduce_trans_rot:
            ref.reduce_trans_rot()
        else:
            logger.info("Skipping reduction of translational and rotational modes as per user request.")

        logger.debug(f"Computed vibrational frequencies:\n{ref.frequency().magnitude} cm^-1")
        projector = Projector(ref=ref, output_unit=output_unit, kabsch_threshold=args.kabsch_threshold)

        projections = []
        for frame_idx, snapshot in trajectory:
            logger.debug(f"Processing frame {frame_idx + 1}/{len(trajectory)}...")
            nmode_coords = projector.project(snapshot)
            projections.append((frame_idx, nmode_coords))
            logger.debug(f"Frame {frame_idx} projection complete.")

        if args.output_csv_name is not None:
            if args.output_csv_name.is_absolute() or len(args.output_csv_name.parts) > 1:
                csv_path = args.output_csv_name
            else:
                csv_dir = PROJECT_ROOT.joinpath("data/csv")
                csv_path = csv_dir.joinpath(args.output_csv_name.name)
        else:
            default_csv_name = f"{args.snapshot.stem}-nmode.csv"
            csv_dir = PROJECT_ROOT.joinpath("data/csv")
            csv_path = csv_dir.joinpath(default_csv_name)

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        projector.to_csv(csv_path=csv_path, projections=projections, align=args.align_csv)

    except Exception:
        logger.exception("An unexpected error occurred")
        sys.exit(1)

    if logname:
        logger.info(f"Log saved to: {logname}")

    logger.info("Projection completed successfully.")


if __name__ == "__main__":
    main()
