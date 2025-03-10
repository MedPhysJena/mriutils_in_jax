import jax.numpy as jnp
import nibabel as nib
import numpy as onp
import pytest

from mriutils_in_jax.utils import take, update_axis_after_indexing


@pytest.mark.parametrize(
    "target,removed,expected",
    [
        (1, 0, [0, -2]),
        (1, 2, [1, -2]),
        (-2, 0, [0, -2]),
        (-2, -1, [1, -1]),
    ],
)
def test_update_axis_after_indexing(target: int, removed: int, expected: list[int]):
    actual = update_axis_after_indexing(ndim=3, target=target, removed=removed)
    assert actual in expected


def test_update_axis_after_indexing_errors():
    with pytest.raises(ValueError):
        update_axis_after_indexing(ndim=3, target=1, removed=1)


INP = onp.arange(6, dtype=float).reshape((2, 3, 1))
EXP = jnp.arange(3, dtype=float)[:, None]


@pytest.fixture
def dataobj(tmp_path):
    # Create a temporary file path for the NIfTI file.
    nifti_path = tmp_path / "temp.nii.gz"

    # Create a NIfTI image with the dummy data.
    nifti_img = nib.nifti1.Nifti1Image(INP, onp.eye(4))
    nib.save(nifti_img, nifti_path)

    # Yield the file path for use in a test.
    yield nib.nifti1.load(nifti_path).dataobj

    # Cleanup: remove the file after the test run.
    if nifti_path.exists():
        nifti_path.unlink()


@pytest.mark.parametrize("inp", [INP, jnp.array(INP)])
def test_take(inp):
    res = take(inp, indices=0, axis=0)
    assert isinstance(res, jnp.ndarray)
    assert jnp.allclose(res, EXP)


def test_take_dataobj(dataobj):
    res = take(dataobj, indices=0, axis=0)
    assert isinstance(res, jnp.ndarray)
    assert jnp.allclose(res, EXP)
