import numpy as np
import pytest
from recipe_lab.linear import DirectionPatch, fit_symmetric_patch


def test_weight_runtime_equivalence():
    rng = np.random.default_rng(1)
    d, _ = np.linalg.qr(rng.normal(size=(11, 3)))
    p = DirectionPatch(d, np.array([0.5, 1.1, -0.2]))
    w, x = rng.normal(size=(11, 17)), rng.normal(size=(23, 17))
    np.testing.assert_allclose(x @ p.weights(w).T, p.outputs(x @ w.T), atol=1e-12)


def test_general_rank_one_is_not_direction_projection():
    delta = np.array([[0., 1.], [0., 0.]])
    assert np.linalg.matrix_rank(delta) == 1
    assert not np.allclose(delta, delta.T)  # -alpha*d*d.T*I is always symmetric


def test_two_pass_composition_is_not_simple_sum():
    a = np.array([1., 0.]); b = np.array([1., 1.]) / np.sqrt(2)
    A, B = np.outer(a, a), np.outer(b, b)
    np.testing.assert_allclose((np.eye(2)-B)@(np.eye(2)-A), np.eye(2)-A-B+B@A)
    assert not np.allclose((np.eye(2)-B)@(np.eye(2)-A), np.eye(2)-A-B)


def test_symmetric_fit_solves_local_stationarity():
    rng = np.random.default_rng(12)
    q, _ = np.linalg.qr(rng.normal(size=(20, 5)))
    g, b, t = [rng.normal(size=(40, 20)) for _ in range(3)]
    p = fit_symmetric_patch(g, b, t, q, retain_weight=2, ridge=.1)
    H = q.T @ p.directions @ np.diag(p.strengths) @ p.directions.T @ q
    zg, zb, e = g@q, b@q, (b-t)@q
    A = zb.T@zb/40 + 2*zg.T@zg/40 + .1*np.eye(5)
    C = zb.T@e/40
    np.testing.assert_allclose(A@H + H@A, C+C.T, atol=1e-11)


def test_compress_retain_largest_absolute_eigenvalues():
    p = DirectionPatch(np.eye(4), np.array([.2, -.9, .5, .1]))
    np.testing.assert_allclose(p.compress(2).strengths, [-.9, .5])
    assert p.compress(0).is_identity


@pytest.mark.parametrize('strength', [np.nan, np.inf])
def test_nonfinite_rejected(strength):
    with pytest.raises(ValueError): DirectionPatch(np.eye(2), [strength, 1])


def test_nonorthogonal_rejected():
    with pytest.raises(ValueError): DirectionPatch(np.ones((3, 2)), [1, 1])


def test_hooks_match_linear_with_bias():
    torch = pytest.importorskip('torch')
    from recipe_lab.hooks import output_patch
    rng = np.random.default_rng(2)
    d, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    p = DirectionPatch(d, np.array([.7, -.1]))
    layer = torch.nn.Linear(7, 5, bias=True)
    x = torch.randn(3, 4, 7)
    baseline = layer(x).detach().numpy()
    with output_patch(layer, p):
        actual = layer(x).detach().numpy()
    np.testing.assert_allclose(actual, p.outputs(baseline), atol=3e-7)
    np.testing.assert_allclose(layer(x).detach().numpy(), baseline, atol=0)
