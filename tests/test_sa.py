"""Stdlib tests; run with python -m unittest discover -s tests -v."""
import unittest
import numpy as np
from instance import generate_instance
from fitness import combined_objective, decode, evaluate, sa_fitness
from sa import SAConfig, encode_routes, neighbor, simulated_annealing, initial_solution


class TestSimulatedAnnealing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = generate_instance(n=12, m_g=2, m_a=2, seed=0, t_max=1800)

    def test_encode_decode_roundtrip(self):
        routes = [[1, 8], [2, 5, 6], [4, 10, 3], [7, 9, 11, 12]]
        self.assertEqual(decode(encode_routes(routes, self.inst), self.inst), routes)

    def test_neighbor_visits_every_task_once(self):
        inst = self.inst
        rng = np.random.default_rng(99)
        x, r = initial_solution(inst, rng, restarts=10, alpha=0.5, penalty=100.0)
        self.assertEqual(r.details["g_capability"], 0)
        for _ in range(100):
            x = neighbor(x, inst, rng)
            ids = sorted(t for route in decode(x, inst) for t in route)
            self.assertEqual(ids, list(range(1, inst.n + 1)))
            self.assertEqual(evaluate(x, inst).details['g_capability'], 0)

    def test_combined_objective_matches_report(self):
        x = np.full(2 * self.inst.n, 0.5)
        r = evaluate(x, self.inst)
        expected = 0.5*r.f1/self.inst.t_max + 0.5*r.f2/self.inst.battery.sum()
        self.assertAlmostEqual(combined_objective(r, self.inst), expected)
        self.assertGreaterEqual(sa_fitness(r, self.inst), expected)

    def test_common_battery_per_type(self):
        inst = self.inst
        self.assertTrue(np.all(inst.battery[:inst.m_g] == inst.battery[0]))
        self.assertTrue(np.all(inst.battery[inst.m_g:] == inst.battery[inst.m_g]))

    def test_invalid_infinite_deadline_is_rejected(self):
        inst = generate_instance(n=12, m_g=2, m_a=2, seed=0)
        with self.assertRaises(ValueError):
            simulated_annealing(inst, SAConfig(iterations=2))

    def test_sa_produces_feasible_nonworse_result(self):
        cfg = SAConfig(iterations=1000, construction_restarts=25)
        a = simulated_annealing(self.inst, cfg, seed=3)
        self.assertAlmostEqual(a.best_result.violation, 0.0)
        self.assertTrue(np.isfinite(a.best_history).all())
        self.assertTrue(np.all(np.diff(a.best_history) <= 1e-12))
        self.assertLessEqual(a.best_history[-1], a.best_history[0] + 1e-12)
        b = simulated_annealing(self.inst, cfg, seed=3)
        self.assertAlmostEqual(a.best_history[-1], b.best_history[-1])
        self.assertEqual(a.best_result.routes, b.best_result.routes)


if __name__ == '__main__':
    unittest.main()
