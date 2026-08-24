"""测试共享配置与小型原子构型。"""

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "meia-mpl-cache"),
)

import pytest
from ase import Atoms


@pytest.fixture
def sample_atoms() -> Atoms:
    """返回一个无外部文件依赖的多元素周期构型。"""
    return Atoms(
        symbols=["Ca", "O", "C", "O", "H"],
        positions=[
            [0.2, 0.2, 0.2],
            [2.5, 0.2, 0.2],
            [0.5, 3.0, 0.8],
            [1.7, 3.0, 0.8],
            [0.5, 4.09, 0.8],
        ],
        cell=[6.0, 7.0, 8.0],
        pbc=True,
    )
