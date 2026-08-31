import h5py
import numpy as np

from relaibotix.cli import main


def _write_valid_input(path):
    with h5py.File(path, "w") as h5_file:
        features = h5_file.create_dataset("features", data=np.ones((4, 2)))
        features.attrs["feature_names"] = np.asarray(
            ["joint_pos_1", "joint_vel_1"],
            dtype=h5py.string_dtype("utf-8"),
        )
        h5_file.create_dataset("timestamps", data=np.arange(4, dtype=float))
        h5_file.create_dataset("labels", data=np.asarray([0, 0, 1, 1]))


def test_h5_validate_command(tmp_path, capsys):
    path = tmp_path / "input.h5"
    _write_valid_input(path)

    exit_code = main(["h5", "validate", str(path)])

    assert exit_code == 0
    assert "VALID:" in capsys.readouterr().out


def test_h5_convert_command(tmp_path, capsys):
    source = tmp_path / "input.h5"
    output = tmp_path / "converted.h5"
    _write_valid_input(source)

    exit_code = main(["h5", "convert", str(source), str(output)])

    assert exit_code == 0
    assert output.exists()
    text = capsys.readouterr().out
    assert "Converted:" in text
    assert "VALID:" in text
