import sys, os, shutil
sys.path.insert(0, os.path.dirname(__file__))

import lafan1 as lf
import unittest, os, logging
import pickle as pkl

class test_eval(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(os.path.dirname(__file__), 'output', 'results.pkl'), 'rb') as f:
            cls.results = pkl.load(f)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_01_zerov_quat_loss(self):
        results = self.results
        self.assertTrue(results[('zerov_quat_loss', 5)] >= 0.550 and results[('zerov_quat_loss', 5)] < 0.565)
        self.assertTrue(results[('zerov_quat_loss', 15)] >= 1.095 and results[('zerov_quat_loss', 15)] < 1.105)
        self.assertTrue(results[('zerov_quat_loss', 30)] >= 1.505 and results[('zerov_quat_loss', 30)] < 1.515)
        self.assertTrue(results[('zerov_quat_loss', 45)] >= 1.805 and results[('zerov_quat_loss', 45)] < 1.815)

    def test_02_interp_quat_loss(self):
        results = self.results
        self.assertTrue(results[('interp_quat_loss', 5)] >= 0.215 and results[('interp_quat_loss', 5)] < 0.225)
        self.assertTrue(results[('interp_quat_loss', 15)] >= 0.615 and results[('interp_quat_loss', 15)] < 0.625)
        self.assertTrue(results[('interp_quat_loss', 30)] >= 0.975 and results[('interp_quat_loss', 30)] < 0.985)
        self.assertTrue(results[('interp_quat_loss', 45)] >= 1.245 and results[('interp_quat_loss', 45)] < 1.255)

    def test_03_zerov_pos_loss(self):
        results = self.results
        self.assertTrue(results[('zerov_pos_loss', 5)] >= 1.52295 and results[('zerov_pos_loss', 5)] < 1.52305)
        self.assertTrue(results[('zerov_pos_loss', 15)] >= 3.69435 and results[('zerov_pos_loss', 15)] < 3.69445)
        self.assertTrue(results[('zerov_pos_loss', 30)] >= 6.60015 and results[('zerov_pos_loss', 30)] < 6.60025)
        self.assertTrue(results[('zerov_pos_loss', 45)] >= 9.32885 and results[('zerov_pos_loss', 45)] < 9.32895)

    def test_04_interp_pos_loss(self):
        results = self.results
        self.assertTrue(results[('interp_pos_loss', 5)] >= 0.37285 and results[('interp_pos_loss', 5)] < 0.37295)
        self.assertTrue(results[('interp_pos_loss', 15)] >= 1.24875 and results[('interp_pos_loss', 15)] < 1.24885)
        self.assertTrue(results[('interp_pos_loss', 30)] >= 2.31575 and results[('interp_pos_loss', 30)] < 2.31585)
        self.assertTrue(results[('interp_pos_loss', 45)] >= 3.44685 and results[('interp_pos_loss', 45)] < 3.44695)

    def test_05_zerov_npss_loss(self):
        results = self.results
        self.assertTrue(results[('zerov_npss_loss', 5)] >= 0.00525 and results[('zerov_npss_loss', 5)] < 0.00535)
        self.assertTrue(results[('zerov_npss_loss', 15)] >= 0.05215 and results[('zerov_npss_loss', 15)] < 0.05225)
        self.assertTrue(results[('zerov_npss_loss', 30)] >= 0.23175 and results[('zerov_npss_loss', 30)] < 0.23185)
        self.assertTrue(results[('zerov_npss_loss', 45)] >= 0.49175 and results[('zerov_npss_loss', 45)] < 0.49185)

    def test_06_interp_npss_loss(self):
        results = self.results
        self.assertTrue(results[('interp_npss_loss', 5)] >= 0.00225 and results[('interp_npss_loss', 5)] < 0.00235)
        self.assertTrue(results[('interp_npss_loss', 15)] >= 0.03905 and results[('interp_npss_loss', 15)] < 0.03915)
        self.assertTrue(results[('interp_npss_loss', 30)] >= 0.20125 and results[('interp_npss_loss', 30)] < 0.20135)
        self.assertTrue(results[('interp_npss_loss', 45)] >= 0.44925 and results[('interp_npss_loss', 45)] < 0.44935)

if __name__ == '__main__':
    unittest.main(verbosity=2)
