import numpy as np
import pytest
from recipe_lab.linear import DirectionPatch
from recipe_lab.storage import compile_npy, sha256_file, save_patch, load_patch


def setup(tmp_path):
    rng = np.random.default_rng(3)
    w = rng.normal(size=(13, 32)).astype(np.float32)
    d, _ = np.linalg.qr(rng.normal(size=(13, 2)))
    p = DirectionPatch(d, [.4, -.1])
    src = tmp_path/'base.npy'; np.save(src, w)
    return src, w, p


@pytest.mark.parametrize('tile', [1, 4, 64])
def test_npy_tiling(tmp_path, tile):
    src, w, p = setup(tmp_path); sha=sha256_file(src)
    dst=tmp_path/'edited.npy'
    compile_npy(src,dst,p,sha,tile)
    np.testing.assert_allclose(np.load(dst), p.weights(w), atol=2e-7)
    assert sha256_file(src)==sha


def test_wrong_hash_does_not_write(tmp_path):
    src,w,p=setup(tmp_path); dst=tmp_path/'edited.npy'
    with pytest.raises(ValueError): compile_npy(src,dst,p,'0'*64)
    assert not dst.exists()


def test_existing_destination_not_overwritten(tmp_path):
    src,w,p=setup(tmp_path); dst=tmp_path/'edited.npy';dst.write_bytes(b'keep')
    with pytest.raises(FileExistsError): compile_npy(src,dst,p,sha256_file(src))
    assert dst.read_bytes()==b'keep'


def test_identity_exact(tmp_path):
    src,w,p=setup(tmp_path); dst=tmp_path/'edited.npy'
    compile_npy(src,dst,DirectionPatch(np.eye(13)[:,:1],[0]),sha256_file(src))
    assert src.read_bytes()==dst.read_bytes()


def test_patch_roundtrip(tmp_path):
    _,_,p=setup(tmp_path); f=tmp_path/'p.npz';save_patch(f,p);r=load_patch(f)
    np.testing.assert_allclose(p.directions,r.directions,atol=1e-7)
