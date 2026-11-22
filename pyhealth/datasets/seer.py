"""Dataset adapter for locally prepared SEER registry case listings.

SEER files and PyHealth's cache must remain inside approved research storage.
"""
from __future__ import annotations

from pathlib import Path

from pyhealth.datasets import BaseDataset


class SEERDataset(BaseDataset):
    """Load a locally prepared SEER*Stat breast cancer case listing.

    The dataset inherits from PyHealth's ``BaseDataset`` and reads a
    YAML-described ``seer_cases.csv`` table. This class does not download
    or distribute patient-level SEER records.

    Args:
        root: Approved directory containing ``seer_cases.csv``.
        cache_dir: Approved directory for PyHealth event and task caches.
            This is required to prevent writes to default public caches.
        num_workers: Number of PyHealth parsing workers. Defaults to 1.
        dev: When True, process a small development subset.

    Note:
        Keep ``root`` and ``cache_dir`` outside the public source tree.
        The prepared CSV contains confidential case-level records.

    Example:
        ``SEERDataset(root="/approved/data", cache_dir="/approved/cache")``
    """

    def __init__(
        self,
        root: str | Path,
        cache_dir: str | Path,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        """Initialize the dataset using the bundled SEER YAML configuration.

        Args:
            root: Directory containing locally prepared case records.
            cache_dir: Approved location for PyHealth caches.
            num_workers: Number of processing workers.
            dev: Limit records for local integration tests.

        Raises:
            ValueError: If ``cache_dir`` is missing.
        """
        if cache_dir is None:
            raise ValueError("cache_dir is required to prevent unintended data storage")
        config = Path(__file__).parent / "configs" / "seer.yaml"
        super().__init__(
            root=str(root),
            tables=["seer_cases"],
            dataset_name="SEER",
            config_path=str(config),
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )
