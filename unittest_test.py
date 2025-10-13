import unittest
import numpy as np
from reece_homework_4 import (
    aluminum_properties
)

class TestThermalModel(unittest.TestCase):

    def test_aluminum_properties_increasing_with_T(self):
        """Aluminum cp and k should increase (approximately) with temperature."""
        rho1, cp1, k1 = aluminum_properties(300)
        rho2, cp2, k2 = aluminum_properties(600)
        self.assertGreater(cp2, cp1)
        self.assertGreater(k2, k1)
        self.assertLess(rho2, rho1)  # density decreases with T


if __name__ == '__main__':
    unittest.main()