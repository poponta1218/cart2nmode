import importlib
import pkgutil
from logging import getLogger
from pathlib import Path

from .core import REFERENCE_PARSERS, TRAJECTORY_PARSERS, BaseReferenceParser, BaseTrajectoryParser

logger = getLogger(__name__)

for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")


def get_reference_parser(file_path: Path) -> BaseReferenceParser:
    """
    Returns a reference parser object based on the priority of the registered parsers.

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
        If no suitable reference parser is found for the given file.
    """
    for priority, parser_cls in REFERENCE_PARSERS:
        if parser_cls.is_applicable(file_path):
            logger.debug(f"Using {parser_cls.__name__} (Priority: {priority}) for {file_path}")
            return parser_cls(file_path)

    emsg = f"No suitable reference parser found for file: {file_path}"
    logger.error(emsg)
    raise ValueError(emsg)


def get_trajectory_parser(file_path: Path) -> BaseTrajectoryParser:
    """
    Returns a trajectory parser object based on the priority of the registered parsers.

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
        If no suitable trajectory parser is found for the given file.
    """
    for priority, parser_cls in TRAJECTORY_PARSERS:
        if parser_cls.is_applicable(file_path):
            logger.debug(f"Using {parser_cls.__name__} (Priority: {priority}) for {file_path}")
            return parser_cls(file_path)

    emsg = f"No suitable trajectory parser found for file: {file_path}"
    logger.error(emsg)
    raise ValueError(emsg)
