# Cross-Repo Consumption Examples

# Example: How JuniorSOL would use Layer 2
"""
from junior_pi_python.layers.layer2_contextual_brain import ContextualBrainLayer
# from junior_sol import SolanaClient

brain = ContextualBrainLayer(engine)
brain.set_hardware("apple_silicon")
result = brain.generate_with_context("Plan my training for next competition")
# solana_client.log_training(result)
"""

# Example: How JuniorDrive would use Layer 2
"""
# from junior_drive import SimulationEngine
brain = ContextualBrainLayer(engine)
result = brain.generate_with_context("Optimize robotics training trajectory")
# sim_engine.run_with_coaching(result)
"""