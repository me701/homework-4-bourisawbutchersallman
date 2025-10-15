import unittest
import numpy as np
from reece_homework_4 import (
    aluminum_properties,
    water_props,
    composite_properties,
    build_A_matrix_volume_safe_nonuniform,
    dTdt_vol,
    total_nodes,
    z_nodes,
    orig_T,
    r_interface,
    z_interface_down,
    z_interface_up,
    N_water,
    N_wall,
    N_vert,
    T_top,
    T_bot,
    P_in
)

class TestThermalModel(unittest.TestCase):

    def test_aluminum_properties_increasing_with_T(self):
        """Aluminum cp and k should increase (approximately) with temperature."""
        rho1, cp1, k1 = aluminum_properties(300)
        rho2, cp2, k2 = aluminum_properties(600)
        self.assertGreater(cp2, cp1)
        self.assertGreater(k2, k1)
        self.assertLess(rho2, rho1)  # density decreases with T

class TestCompositeProperties(unittest.TestCase):
    def setUp(self):
        self.r_nodes = np.linspace(0, 0.03, 5)
        self.z_nodes = np.linspace(0, 0.12, 5)
        self.T_flat = np.full((len(self.r_nodes) * len(self.z_nodes)), 300.0)

    def test_array_shapes(self):
        rho, cp, k = composite_properties(self.T_flat, self.r_nodes, self.z_nodes)
        self.assertEqual(rho.shape, self.T_flat.shape)
        self.assertEqual(cp.shape, self.T_flat.shape)
        self.assertEqual(k.shape, self.T_flat.shape)

    def test_interface_assignment(self):
        """Test that wall regions near z=0 use aluminum properties."""
        rho, cp, k = composite_properties(self.T_flat, self.r_nodes, self.z_nodes)
        idx_bottom = np.argmin(self.z_nodes)
        idx_top = np.argmax(self.z_nodes)
        self.assertTrue(np.all(k[idx_bottom:idx_bottom+len(self.r_nodes)] > 100))
        self.assertTrue(np.all(k[idx_top*len(self.r_nodes):(idx_top+1)*len(self.r_nodes)] > 100))

class TestMatrixBuilder(unittest.TestCase):
    def test_matrix_output(self):
        """Ensure A is sparse, square, and b has correct length."""
        dummy_k = np.full((len(total_nodes) * len(z_nodes)), 200.0)
        A, b = build_A_matrix_volume_safe_nonuniform(total_nodes, z_nodes, dummy_k)
        self.assertIsInstance(A, csr_matrix)
        self.assertEqual(A.shape[0], A.shape[1])
        self.assertEqual(b.shape[0], A.shape[0])
        self.assertFalse(np.any(np.isnan(A.data)))

    def test_convective_bc_contribution(self):
        """Check that outer/top/bottom BCs give nonzero b vector."""
        dummy_k = np.full((len(total_nodes) * len(z_nodes)), 200.0)
        A, b = build_A_matrix_volume_safe_nonuniform(total_nodes, z_nodes, dummy_k)
        self.assertTrue(np.any(b != 0.0))


if __name__ == '__main__':
    unittest.main()