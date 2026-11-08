import unittest
import numpy as np
from ga import GAConfig, genetic_algorithm
from instance import generate_instance
from fitness import evaluate, combined_objective
from validation.exact import exact_solve


class GATests(unittest.TestCase):
    def test_reproducible_budget_and_exact_bound(self):
        inst=generate_instance(n=5,m_g=1,m_a=1,seed=0,t_max=1800.)
        cfg=GAConfig(population=20,evaluations=117)
        a=genetic_algorithm(inst,cfg,3); b=genetic_algorithm(inst,cfg,3)
        np.testing.assert_array_equal(a['best_x'],b['best_x'])
        self.assertEqual(a['evaluations'],117)
        r=evaluate(a['best_x'],inst)
        self.assertEqual(sorted(t for route in r.routes for t in route),list(range(1,6)))
        self.assertEqual(r.details['g_capability'],0)
        self.assertLessEqual(r.violation,1e-9)
        optimum,_=exact_solve(inst)
        self.assertGreaterEqual(combined_objective(r,inst)+1e-12,combined_objective(optimum,inst))
        costs=[v for _,v in a['history'] if v is not None]
        self.assertTrue(all(b<=a+1e-12 for a,b in zip(costs,costs[1:])))

    def test_configuration(self):
        for kwargs in ({'population':1},{'mutation':-1},{'elite':60},{'evaluations':1}):
            with self.assertRaises(ValueError): GAConfig(**kwargs)
