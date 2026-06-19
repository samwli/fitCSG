import torch

from fitcsg.transforms import euler_deg_to_matrix, world_to_local


def test_identity_rotation():
    R = euler_deg_to_matrix(torch.zeros(3))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_rotation_is_orthonormal():
    for angles in [[10, 20, 30], [90, 0, 45], [0, 123, -77], [359, 1, 180]]:
        R = euler_deg_to_matrix(torch.tensor(angles, dtype=torch.float32))
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert abs(float(torch.det(R)) - 1.0) < 1e-5


def test_world_to_local_round_trip():
    center = torch.tensor([0.3, -0.2, 0.5])
    rot = torch.tensor([15.0, 25.0, 40.0])
    pts = torch.randn(50, 3)
    local = world_to_local(pts, center, rot)
    # Distances to the origin are preserved by a rigid transform.
    assert torch.allclose((pts - center).norm(dim=1), local.norm(dim=1), atol=1e-5)
