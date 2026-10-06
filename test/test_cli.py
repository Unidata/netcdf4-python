import sys
from unittest.mock import patch

import numpy as np
import pytest

from netCDF4 import Dataset, utils


@pytest.mark.parametrize("command", [utils.ncinfo, utils.nc3tonc4, utils.nc4tonc3])
@pytest.mark.parametrize("args", [[], ["--unknown"]])
def test_usage_error(command, args, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [command.__name__, *args])
    with pytest.raises(SystemExit) as exc:
        command()
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize("command, option", [
    (utils.ncinfo, "--group"),
    (utils.nc3tonc4, "--chunk"),
    (utils.nc4tonc3, "--chunk"),
])
def test_missing_option_value(command, option, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [command.__name__, option])
    with pytest.raises(SystemExit) as exc:
        command()
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize("command", [utils.nc3tonc4, utils.nc4tonc3])
@pytest.mark.parametrize("args", [["source.nc"], ["source.nc", "dest.nc", "extra.nc"]])
def test_converter_file_count(command, args, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [command.__name__, *args])
    with pytest.raises(SystemExit) as exc:
        command()
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize("command", [utils.ncinfo, utils.nc3tonc4, utils.nc4tonc3])
@pytest.mark.parametrize("option", ["-h", "--help"])
def test_help(command, option, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [command.__name__, option])
    with pytest.raises(SystemExit) as exc:
        command()
    assert exc.value.code == 0
    output = capsys.readouterr()
    assert "usage:" in output.err
    assert "Error" not in output.out


def test_ncinfo_group_dimension(tmp_path, monkeypatch, capsys):
    filename = str(tmp_path / "groups.nc")
    with Dataset(filename, "w") as dataset:
        dataset.createGroup("g1").createDimension("x", 7)
    monkeypatch.setattr(sys, "argv", ["ncinfo", "-g", "g1", "-d", "x", filename])
    utils.ncinfo()
    assert "name = 'x', size = 7" in capsys.readouterr().out


@pytest.mark.parametrize("chunk", [None, 0, 5])
def test_nc4tonc3_chunk(tmp_path, chunk, monkeypatch):
    source = str(tmp_path / "source.nc")
    destination = str(tmp_path / "destination.nc")
    values = np.arange(25, dtype="i4")
    with Dataset(source, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("record", None)
        dataset.createVariable("values", "i4", ("record",))[:] = values
    args = ["nc4tonc3", "--quiet=1"]
    if chunk is not None:
        args.append("--chunk=%s" % chunk)
    monkeypatch.setattr(sys, "argv", [*args, source, destination])
    with patch.object(utils, "_nc4tonc3", wraps=utils._nc4tonc3) as convert:
        utils.nc4tonc3()
    assert convert.call_args.kwargs["nchunk"] == (10 if chunk is None else chunk)
    with Dataset(destination) as dataset:
        assert dataset.file_format == "NETCDF3_64BIT_OFFSET"
        np.testing.assert_array_equal(dataset.variables["values"][:], values)
