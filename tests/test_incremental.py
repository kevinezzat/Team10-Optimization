"""Independent cache, neighborhood, feasibility and exactness regressions."""
import unittest
import numpy as np
from fitness import combined_objective, decode, evaluate
from instance import generate_instance
from sa import SAConfig, encode_routes, initial_solution, simulated_annealing
from sa_engine import RouteState, propose
from validation.exact import exact_solve


class TestIncremental(unittest.TestCase):
    """Exercise every neighborhood and compare against the original evaluator."""

    def test_thousands_of_moves(self):
        """Compare objectives, each constraint, and route metrics after 3000 moves."""
        inst = generate_instance(n=30, seed=4, t_max=900)
        rng = np.random.default_rng(73)
        x, _ = initial_solution(inst, rng, 1, .5, 100)
        state = RouteState(decode(x, inst), inst)
        for step in range(3000):
            op = SAConfig().operators[step % 5]
            candidate = propose(state, inst, rng, op)
            if candidate is None:
                continue
            state = candidate
            full = evaluate(encode_routes(state.routes, inst), inst)
            self.assertEqual(len(state.routes), inst.m)
            self.assertEqual(full.details['g_capability'], 0)
            np.testing.assert_allclose(full[:3], state.result[:3], atol=1e-9, rtol=0)
            for key in ('g_capability', 'g_payload', 'g_energy', 'g_deadline'):
                self.assertAlmostEqual(full.details[key], state.result.details[key], delta=1e-9)

    def test_hand_computed_constraints(self):
        """One incompatible task yields g1=1, g2=1, g3=1, g4=2."""
        inst = generate_instance(n=1, m_g=1, m_a=0, p_ugv_only=0, p_uav_only=0, t_max=2)
        inst.ttype[:] = 2
        inst.demand[:] = 4
        inst.service[:] = 2
        inst.D_ugv[:] = [[0, 2], [2, 0]]
        inst.speed[:] = 1
        inst.e_rate[:] = 1
        inst.payload[:] = 2
        inst.battery[:] = 2
        result = evaluate(encode_routes([[1]], inst), inst)
        self.assertEqual(result[:3], (6., 4., 5.))
        self.assertEqual([result.details[k] for k in ('g_capability', 'g_payload', 'g_energy', 'g_deadline')], [1., 1., 1., 2.])
        self.assertEqual(combined_objective(result, inst), 2.5)

    def test_exact_and_trivial_sa(self):
        """SA finds the enumerated optimum for a four-task single-vehicle tour."""
        inst = generate_instance(n=4, m_g=1, m_a=0, p_ugv_only=0, p_uav_only=0, t_max=10000)
        optimum, _ = exact_solve(inst)
        checked = evaluate(encode_routes(optimum.routes, inst), inst)
        self.assertEqual(optimum[:3], checked[:3])
        result = simulated_annealing(inst, SAConfig(iterations=1000, construction_restarts=1), seed=7)
        self.assertAlmostEqual(combined_objective(result.best_result, inst), combined_objective(optimum, inst))

    def test_infeasible_is_returned(self):
        """An impossible deadline produces a flagged result, not a search exception."""
        inst = generate_instance(n=6, t_max=.1)
        result = simulated_annealing(inst, SAConfig(iterations=100, construction_restarts=1))
        self.assertFalse(result.feasible)
        self.assertGreater(result.best_result.violation, 0)

    def test_config(self):
        """Invalid numerical and neighborhood settings fail before running."""
        for kwargs in ({'iterations': 0}, {'penalty': np.nan}, {'alpha': 2},
                       {'initial_temperature': 0}, {'operators': ()}, {'n_restarts': -1}):
            with self.assertRaises(ValueError):
                SAConfig(**kwargs)

    def test_restarts_and_adaptation_are_deterministic(self):
        """Seeded chains, adaptive selection and reheating reproduce full histories."""
        inst = generate_instance(n=8, seed=6, t_max=1800)
        config = SAConfig(iterations=200, construction_restarts=2, n_restarts=2,
                          adaptive_operators=True, adaptive_penalty=True,
                          reheat=True, stagnation_window=30)
        a = simulated_annealing(inst, config, seed=19)
        b = simulated_annealing(inst, config, seed=19)
        self.assertEqual(a.best_result.routes, b.best_result.routes)
        self.assertEqual(a.best_chain, b.best_chain)
        np.testing.assert_array_equal(a.current_history, b.current_history)
        np.testing.assert_array_equal(a.temperature_history, b.temperature_history)
        self.assertEqual(a.evaluations, b.evaluations)

    def test_manual_schedule_and_hill_climbing(self):
        """Derived temperatures reach the endpoint; zero-temperature costs descend."""
        inst = generate_instance(n=6, seed=2, t_max=1800)
        config = SAConfig(iterations=100, construction_restarts=1,
                          initial_temperature=.1, final_temperature=.001)
        result = simulated_annealing(inst, config)
        self.assertAlmostEqual(result.temperature_history[0], .1)
        self.assertAlmostEqual(result.temperature_history[-2], .001)
        hill = simulated_annealing(inst, SAConfig(iterations=100, construction_restarts=1,
                                                 hill_climbing=True))
        self.assertTrue(np.all(np.diff(hill.current_history) <= 1e-9))

    def test_battery_stress_has_unreachable_aerial_tasks(self):
        """The named battery stress must prevent some direct aerial round trips."""
        from run_cases import make_case
        inst = make_case('battery75')
        k = inst.m_g
        self.assertTrue(np.any(2*inst.D_uav[0, 1:]*inst.e_rate[k] > inst.battery[k]))
