import torch

from fitcsg.grid import create_grid
from fitcsg.primitives import PRIMITIVES, get_primitive
from fitcsg.transforms import euler_deg_to_matrix


def _params(shape):
    base = {"center": torch.zeros(3), "rotation": torch.zeros(3)}
    spec = PRIMITIVES[shape]
    for key in spec.vector_params:
        base[key] = torch.tensor([0.6, 0.6, 0.6])
    for key in spec.scalar_params:
        base[key] = torch.tensor(0.4 if key in ("radius", "height") else 0.15)
    return base


def test_sphere_exact_distance():
    sphere = get_primitive("sphere")
    p = {"center": torch.zeros(3), "rotation": torch.zeros(3), "radius": torch.tensor(0.5)}
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    d = sphere.sdf(pts, p)
    assert torch.allclose(d, torch.tensor([-0.5, 0.5]), atol=1e-6)


def test_box_exact_distance():
    box = get_primitive("box")
    p = {"center": torch.zeros(3), "rotation": torch.zeros(3), "size": torch.ones(3)}
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    d = box.sdf(pts, p)
    assert torch.allclose(d, torch.tensor([-0.5, 0.5]), atol=1e-6)


def test_every_primitive_sign():
    """Each primitive encloses a non-empty solid; far-away points are outside.

    (We check the minimum SDF over a grid rather than the centre point, because
    a torus' centre is in its hole and the ellipsoid approximation is exactly 0
    at the origin.)
    """
    grid = create_grid(24)
    far = torch.tensor([[10.0, 10.0, 10.0]])
    for shape in PRIMITIVES:
        spec = get_primitive(shape)
        params = _params(shape)
        assert spec.sdf(grid, params).min().item() < 0, f"{shape} has empty interior"
        assert spec.sdf(far, params).item() > 0, f"{shape} far point not outside"


def test_box_rotation_equivariance():
    box = get_primitive("box")
    pts = torch.tensor([[0.6, 0.0, 0.0], [0.0, 0.6, 0.0], [0.3, 0.3, 0.0]])
    p0 = {"center": torch.zeros(3), "rotation": torch.zeros(3), "size": torch.tensor([1.0, 0.4, 0.4])}
    p90 = {"center": torch.zeros(3), "rotation": torch.tensor([0.0, 0.0, 90.0]), "size": torch.tensor([1.0, 0.4, 0.4])}
    R = euler_deg_to_matrix(torch.tensor([0.0, 0.0, 90.0]))
    assert torch.allclose(box.sdf(pts, p0), box.sdf(pts @ R.T, p90), atol=1e-5)


def test_size_sign_invariance():
    """abs() on sizes means negative sizes give the same shape."""
    box = get_primitive("box")
    pts = torch.randn(20, 3)
    pos = {"center": torch.zeros(3), "rotation": torch.zeros(3), "size": torch.tensor([0.8, 0.5, 0.6])}
    neg = {"center": torch.zeros(3), "rotation": torch.zeros(3), "size": torch.tensor([-0.8, -0.5, -0.6])}
    assert torch.allclose(box.sdf(pts, pos), box.sdf(pts, neg), atol=1e-6)
