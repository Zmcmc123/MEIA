"""原子构型文件读取入口测试。"""

from ase import Atoms
from ase.io import write
import pytest

from meia.i18n import I18n, Locale
from meia.io import StructureReadError, is_supported_structure_filename, read_structure


def test_structure_read_error_preserves_filename_and_has_english_semantics(
    monkeypatch,
):
    def fail_read(*_args, **_kwargs):
        raise RuntimeError("ASE parser failed")

    monkeypatch.setattr("meia.io.read", fail_read)
    with pytest.raises(StructureReadError) as captured:
        read_structure("/tmp/中文构型.xyz")

    assert I18n(Locale.EN).error_text(
        captured.value, "file.structure_read_failed"
    ) == (
        "Could not read /tmp/中文构型.xyz (RuntimeError): "
        "ASE parser failed"
    )


def test_supported_filename_accepts_extensionless_vasp_names():
    """POSCAR/CONTCAR 是标准无后缀文件名，不能被上传层拒绝。"""
    assert is_supported_structure_filename("POSCAR")
    assert is_supported_structure_filename("CONTCAR")
    assert is_supported_structure_filename("sample.data")
    assert not is_supported_structure_filename("notes.txt")


def test_read_structure_handles_lammps_data_extension(tmp_path):
    """`.data` 必须显式走 ASE LAMMPS data 解析器。"""
    source = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    filepath = tmp_path / "structure.data"
    write(filepath, source, format="lammps-data")

    loaded = read_structure(filepath)

    assert len(loaded) == 2
    assert loaded.positions.shape == (2, 3)
    assert loaded.pbc.all()
