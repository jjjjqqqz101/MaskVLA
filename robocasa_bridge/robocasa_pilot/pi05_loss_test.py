"""Gradient and dimensional regression tests for the pilot-only robot7 loss."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from .pi05_loss import robot_action_mse

class RobotFlowDimensions(unittest.TestCase):
    def test_unused25_dimensions_have_zero_gradient_and_cannot_reduce_loss(self):
        prediction=jnp.ones((2,10,32))
        target=jnp.zeros_like(prediction)
        objective=lambda x:jnp.mean(robot_action_mse(x,target))
        self.assertAlmostEqual(float(objective(prediction)),1.,places=6)
        changed=prediction.at[...,7:].set(1000.)
        self.assertEqual(float(objective(changed)),float(objective(prediction)))
        grad=np.asarray(jax.grad(objective)(changed))
        self.assertTrue(np.all(grad[...,:7] != 0))
        np.testing.assert_array_equal(grad[...,7:],0.)
        self.assertAlmostEqual(float(grad[0,0,0]),2/(2*10*7),places=7)

    def test_real_gripper_and_rotation_dimensions_participate(self):
        prediction=jnp.zeros((1,10,32));target=jnp.zeros_like(prediction)
        for channel in range(7):
            changed=prediction.at[...,channel].set(1.)
            self.assertAlmostEqual(float(robot_action_mse(changed,target).mean()),1/7,places=6)
        self.assertEqual(robot_action_mse(prediction,target).shape,(1,10))

    def test_internal_dimension_is32_and_shape_mismatch_rejected(self):
        for shape in ((1,10,7),(1,10,31),(1,10,33)):
            with self.assertRaises(ValueError):robot_action_mse(jnp.zeros(shape),jnp.zeros(shape))
        with self.assertRaises(ValueError):robot_action_mse(jnp.zeros((1,10,32)),jnp.zeros((1,9,32)))

if __name__ == "__main__":unittest.main()
