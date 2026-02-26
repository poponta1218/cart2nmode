# cart2nmode

A robust Python utility for projecting the Cartesian coordinates of molecular snapshots (e.g., from MD trajectories or scans) onto the vibrational normal modes of a reference structure.

This tool performs Mass-Weighted Normal Mode Analysis, automatically handling translational/rotational (TR) mode removal and structural alignment via the Kabsch algorithm.

## Features

* **Multi-frame Trajectory Support**: Processes entire trajectory files (e.g., `.xyz`) instead of single snapshots.
* **Robust Unit Handling**: Powered by Pint, allowing seamless conversion between Angstrom, Bohr, and other length units.
* **Structural Alignment**: Automatically aligns the snapshot to the reference using the Kabsch algorithm to minimize RMSD before projection.
* **TR Mode Reduction**: Automatically identifies and removes translational and rotational modes from the reference Hessian.
* **Modern Data Stack**: Uses Polars for high-performance CSV output and Pydantic for strict data validation.
* **Comprehensive Logging**: Detailed execution logs are saved to the `log/` directory.

## Directory Structure

```text
.
├── data/                 # Input files and output CSVs
│   ├── csv/              # Default output directory for CSV results
│   └── ...               # Example inputs (XYZ, FChk, etc.)
├── log/                  # Execution logs (timestamped)
├── utils/                # Utility functions and modules
│   └── parsers/          # Trajectory and reference file parsers
│       ├── __init__.py   # Parser package initialization
│       ├── core.py       # Core parsing logic and base classes
│       ├── reference.py  # Reference file parsers (e.g., FChkParser)
│       └── trajectory.py # Trajectory file parsers (e.g., XYZParser)
├── main.py               # Entry point
├── pyproject.toml        # Project configuration (uv)
├── uv.lock               # Lock file
├── requirements.txt      # Dependency file (synced with uv)
└── README.md             # This file
```

## Installation

This project is managed using [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

### Option 1: Using uv (Recommended)

Ensure you have `uv` installed. Then, simply sync the project:

```bash
uv sync
```

To run the script directly with dependencies:

```bash
uv run main.py [arguments]
```

### Option 2: Using standard pip

If you prefer standard pip, a `requirements.txt` is provided.

```bash
# Recommended: Create a virtual environment first
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
python main.py [arguments]
```

## Usage

The script requires two primary inputs: a **Reference** file (containing Hessian/Frequency data) and a **Snapshot** file (geometry to project).

### Basic Example

```bash
uv run main.py \
  --reference data/TS1_freq-CCC.fchk \
  --snapshot data/CCC-mode007-00050.xyz \
  --output-csv-name projection_result.csv
```

### Command Line Arguments

| Argument | Flag | Type | Default | Description |
| --- | --- | --- | --- | --- |
| **Reference** | `-r`, `--reference` | Path | **Required** | Path to the reference file (e.g., `.fchk`). Must contain Hessian/Frequency data. |
| **Snapshot** | `-s`, `--snapshot` | Path | **Required** | Path to the snapshot file (e.g., `.xyz`). |
| **Output Name** | `-o`, `--output-csv-name` | Path | `<snapshot_name>-nmode.csv` | Filename for the output CSV. By default, it saves to `data/csv/` using the snapshot's base name appended with `-nmode`. It also respects absolute paths if provided. |
| **Input Unit** | `-u`, `--input-unit` | Str | `angstrom` | Unit of coordinates in the snapshot file (e.g., `angstrom`, `bohr`). |
| **Output Unit** | `-U`, `--output-unit` | Str | `bohr` | Unit for the projected coordinates (sqrt(amu) * Unit). |
| **Align CSV** | `-a`, `--align-csv` | Bool | `True` | Format CSV columns with padding for readability. Use `--no-align-csv` to disable. |
| **Kabsch Thresh** | `-k`, `--kabsch-threshold` | Float | `3.0` | RMSD threshold (in Bohr) to trigger a warning during alignment. |
| **Reduce TR** | `-R`, `--reduce-trans-rot` | Bool | `True` | Remove Translation/Rotation modes. Use `--no-reduce-trans-rot` to keep them. |
| **Verbose** | `-v`, `--verbose` | Bool | `False` | Enable debug logging to console. |

### Output Format

The output CSV will be saved in `data/csv/` (unless an absolute path is provided).

**Columns:**

* `frame`: Frame index from the snapshot trajectory (0-based index).
* `mode_i`: The projection coefficient of the normal mode $`Q_i`$ in units of $`(\mathrm{amu})^{1/2} a_0`$ (by default).

Example content:

```csv
frame,       mode_0,       mode_7,       mode_8
    0,   1.2345E-01,  -4.5678E-03,   8.9012E-02
    1,   1.2400E-01,  -4.5000E-03,   8.9500E-02
...
```

## Methodology

The projection workflow consists of two main steps: Structural Alignment and Normal Mode Projection.

### Structural Alignment (Kabsch Algorithm)

To isolate internal vibrational motions, the snapshot structure is aligned to the reference structure by minimizing the Mass-Weighted RMSD.

Let matrices $`X_{\text{snap}}`$ and $`X_{\text{ref}}`$ be the centered, $`N \times 3`$ coordinates of the snapshot and reference structures, respectively.  
The optimal rotation matrix $`R`$ is derived via SVD:

**1. Compute the Mass-Weighted Covariance Matrix $C$**:

```math
C = {X_{\text{snap}}}^\mathsf{T} W X_{\text{ref}}
```

where $`W`$ is the diagonal matrix of atomic masses ($`N \times N`$).

**2. Singular Value Decomposition (SVD)**:
Decompose $`C`$ into unitary matrices $`U`$ and $`V^\mathsf{T}`$:

```math
C = U \Sigma V^\mathsf{T}
```

**3. Compute Rotation Matrix $`R`$**:

```math
R = U V^\mathsf{T}
```

(*Determinant check and reflection correction are applied as necessary.*)

**4. Apply Rotation and Compute Displacement**:

```math
X_{\text{snap}}^{\text{aligned}} = X_{\text{snap}} R
```

### Normal Mode Projection

After alignment, the coordinates are flattened into vectors ($`3N`$-dimensional).  
The projection coordinate $`Q_i`$ for the $`i`$-th normal mode is calculated as:

```math
Q_i = {l_i}^\mathsf{T} M^{\frac{1}{2}} (\mathbf{\mathit{x}}_\text{snap}^{\text{aligned}} - \mathbf{\mathit{x}}_{\text{ref}})
```

where:

* $`\mathbf{\mathit{x}}_\text{snap}^{\text{aligned}}`$: Flattened coordinate vector ($`3N \times 1`$) derived from $`X_{\text{snap}}^{\text{aligned}}`$.
* $`\mathbf{\mathit{x}}_\text{ref}`$: Flattened reference coordinate vector derived from $`X_{\text{ref}}`$ ($`3N \times 1`$).
* $`M`$: Diagonal mass matrix ($`3N \times 3N`$). Each atomic mass is repeated 3 times along the diagonal.
* $`l_i`$: The $`i`$-th normal mode eigenvector ($`3N \times 1`$) of the mass-weighted Hessian.

> [!NOTE]
> The mass-weighted displacement vector $`\mathbf{\mathit{q}}`$ can be reconstructed from the normal mode coordinates $`Q_i`$:
>
> ```math
> \begin{align*}
>     \mathbf{\mathit{q}} & = LQ \\
>         & = \sum_{i} Q_i \mathbf{l}_i
> \end{align*}
> ```
>
> where $`\mathbf{\mathit{q}} = M^{\frac{1}{2}} (\mathbf{\mathit{x}}_{\text{snap}}^{\text{aligned}} - \mathbf{\mathit{x}}_{\text{ref}})`$.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
