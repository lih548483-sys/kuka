"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Unit tests for verifying orthonormality of local bases computed by diffusion.

Tests that local coordinate frames (tangent, y-axis, normal) form orthonormal bases
for both pointcloud and Walk-on-Spheres diffusion methods.
"""

import numpy as np
import pytest

from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import Pointcloud


@pytest.fixture
def pointcloud_with_bases():
    """Create a pointcloud and compute local bases."""
    pcloud = Pointcloud(filename="spot.ply")
    scalar_diffusion = PointcloudScalarDiffusion(pcloud, diffusion_scalar=1000)
    scalar_diffusion.get_local_bases()
    return pcloud


@pytest.fixture
def wos_with_grid_bases(pointcloud_with_bases):
    """Create WoS diffusion and compute orientation field on grid."""
    pcloud = pointcloud_with_bases
    boundaries = [pcloud]

    wos_diffusion = WalkOnSpheresDiffusion(
        boundaries=boundaries,
        convergence_threshold=pcloud.get_mean_edge_length() * 2,
    )

    # Compute orientation field on a grid
    grid = pcloud.get_bounding_box_grid(bounding_box_scalar=1, nb_points=11)
    grid.get_center()
    wos_diffusion.diffuse_orientations_on_grid(grid)

    return grid


def test_pointcloud_orthonormality(pointcloud_with_bases):
    """Test that pointcloud local bases are orthonormal (R^T @ R = I)."""
    pcloud = pointcloud_with_bases
    local_bases = pcloud.local_bases

    assert local_bases is not None, "Local bases should be computed"

    identity_target = np.eye(3)
    max_error = 0.0
    non_orthonormal_count = 0

    for i in range(len(local_bases)):
        R = local_bases[i]

        # Check R^T @ R = I
        orthonormality_check = R.T @ R
        error = np.max(np.abs(orthonormality_check - identity_target))
        max_error = max(max_error, error)

        if error > 1e-6:
            non_orthonormal_count += 1

    assert non_orthonormal_count == 0, (
        f"{non_orthonormal_count} / {len(local_bases)} frames have error > 1e-6. "
        f"Max error: {max_error:.3e}"
    )
    assert max_error < 1e-6, (
        f"Max orthonormality error {max_error:.3e} exceeds 1e-6 threshold"
    )


def test_pointcloud_determinant(pointcloud_with_bases):
    """Test that pointcloud local bases have determinant ±1."""
    pcloud = pointcloud_with_bases
    local_bases = pcloud.local_bases

    max_det_error = 0.0

    for i in range(len(local_bases)):
        R = local_bases[i]

        # Check det(R) = ±1
        det = np.linalg.det(R)
        det_error = np.abs(np.abs(det) - 1.0)
        max_det_error = max(max_det_error, det_error)

    assert max_det_error < 1e-6, (
        f"Max determinant error {max_det_error:.3e} exceeds 1e-6 threshold. "
        f"All rotation matrices should have determinant ±1."
    )


def test_pointcloud_pairwise_orthogonality(pointcloud_with_bases):
    """Test pairwise orthogonality of axes (tangent, y, normal) on sample frames."""
    pcloud = pointcloud_with_bases
    local_bases = pcloud.local_bases

    N = len(local_bases)

    # Sample up to 10 random frames
    num_samples = min(10, N)
    sample_indices = np.random.choice(N, size=num_samples, replace=False)

    for idx in sample_indices:
        R = local_bases[idx]
        t = R[:, 0]  # tangent (x-axis)
        y = R[:, 1]  # y-axis
        n = R[:, 2]  # normal (z-axis)

        # Check pairwise orthogonality
        t_dot_y = np.dot(t, y)
        t_dot_n = np.dot(t, n)
        y_dot_n = np.dot(y, n)

        # Check unit length
        t_norm = np.linalg.norm(t)
        y_norm = np.linalg.norm(y)
        n_norm = np.linalg.norm(n)

        # Assertions for this frame
        assert abs(t_dot_y) < 1e-6, f"Frame {idx}: tangent not orthogonal to y-axis (dot={t_dot_y:.3e})"
        assert abs(t_dot_n) < 1e-6, f"Frame {idx}: tangent not orthogonal to normal (dot={t_dot_n:.3e})"
        assert abs(y_dot_n) < 1e-6, f"Frame {idx}: y-axis not orthogonal to normal (dot={y_dot_n:.3e})"

        assert abs(t_norm - 1.0) < 1e-6, f"Frame {idx}: tangent not unit length (|t|={t_norm:.10f})"
        assert abs(y_norm - 1.0) < 1e-6, f"Frame {idx}: y-axis not unit length (|y|={y_norm:.10f})"
        assert abs(n_norm - 1.0) < 1e-6, f"Frame {idx}: normal not unit length (|n|={n_norm:.10f})"


def test_wos_grid_orthonormality(wos_with_grid_bases):
    """Test that WoS-computed grid local bases are orthonormal."""
    grid = wos_with_grid_bases
    local_bases = grid.local_bases

    assert local_bases is not None, "Grid local bases should be computed"

    identity_target = np.eye(3)
    max_error = 0.0
    non_orthonormal_count = 0

    for i in range(len(local_bases)):
        R = local_bases[i]

        # Check R^T @ R = I
        orthonormality_check = R.T @ R
        error = np.max(np.abs(orthonormality_check - identity_target))
        max_error = max(max_error, error)

        if error > 1e-6:
            non_orthonormal_count += 1

    assert non_orthonormal_count == 0, (
        f"{non_orthonormal_count} / {len(local_bases)} frames have error > 1e-6. "
        f"Max error: {max_error:.3e}"
    )
    assert max_error < 1e-6, (
        f"Max orthonormality error {max_error:.3e} exceeds 1e-6 threshold"
    )


def test_wos_grid_determinant(wos_with_grid_bases):
    """Test that WoS-computed grid local bases have determinant ±1."""
    grid = wos_with_grid_bases
    local_bases = grid.local_bases

    max_det_error = 0.0

    for i in range(len(local_bases)):
        R = local_bases[i]

        # Check det(R) = ±1
        det = np.linalg.det(R)
        det_error = np.abs(np.abs(det) - 1.0)
        max_det_error = max(max_det_error, det_error)

    assert max_det_error < 1e-6, (
        f"Max determinant error {max_det_error:.3e} exceeds 1e-6 threshold. "
        f"All rotation matrices should have determinant ±1."
    )


def test_orthonormality_statistics(pointcloud_with_bases, wos_with_grid_bases):
    """Compute and verify statistics on orthonormality errors for both methods."""
    pcloud = pointcloud_with_bases
    grid = wos_with_grid_bases

    all_orthonormality_errors = []
    all_determinant_errors = []
    identity_target = np.eye(3)

    # Check pointcloud bases
    for R in pcloud.local_bases:
        orthonormality_check = R.T @ R
        error = np.max(np.abs(orthonormality_check - identity_target))
        all_orthonormality_errors.append(error)

        det = np.linalg.det(R)
        det_error = np.abs(np.abs(det) - 1.0)
        all_determinant_errors.append(det_error)

    # Check grid bases
    for R in grid.local_bases:
        orthonormality_check = R.T @ R
        error = np.max(np.abs(orthonormality_check - identity_target))
        all_orthonormality_errors.append(error)

        det = np.linalg.det(R)
        det_error = np.abs(np.abs(det) - 1.0)
        all_determinant_errors.append(det_error)

    all_orthonormality_errors = np.array(all_orthonormality_errors)
    all_determinant_errors = np.array(all_determinant_errors)

    # Print statistics (will show in pytest output with -v or -s)
    print(f"\n{'='*60}")
    print(f"ORTHONORMALITY TEST STATISTICS")
    print(f"{'='*60}")
    print(f"Total frames checked: {len(all_orthonormality_errors)}")
    print(f"  - Pointcloud frames: {len(pcloud.local_bases)}")
    print(f"  - Grid frames: {len(grid.local_bases)}")
    print()
    print("ORTHONORMALITY CHECK (R^T @ R = I):")
    print(f"  Mean error: {all_orthonormality_errors.mean():.3e}")
    print(f"  Std error:  {all_orthonormality_errors.std():.3e}")
    print(f"  Max error:  {all_orthonormality_errors.max():.3e}")
    print(f"  Min error:  {all_orthonormality_errors.min():.3e}")
    print()
    print("DETERMINANT CHECK (|det(R)| = 1):")
    print(f"  Mean error: {all_determinant_errors.mean():.3e}")
    print(f"  Std error:  {all_determinant_errors.std():.3e}")
    print(f"  Max error:  {all_determinant_errors.max():.3e}")
    print(f"  Min error:  {all_determinant_errors.min():.3e}")
    print(f"{'='*60}\n")

    # Verify statistics are reasonable
    assert all_orthonormality_errors.mean() < 1e-10, "Mean error should be near machine precision"
    assert all_determinant_errors.mean() < 1e-10, "Mean determinant error should be near machine precision"
