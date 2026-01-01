"""
Lane 2 SpinLab - Quantum Radical-Pair Simulator
================================================

A computational testbed for radical-pair magnetoreception and
quantum coherence in biological systems.

Core capabilities:
- Radical-pair Hamiltonian dynamics (Zeeman + hyperfine)
- Open-system Lindblad master equation
- Singlet/triplet yield calculations
- Magnetic field sensitivity analysis
- Phase boundary detection (criticality metrics)

Physical substrate testing for consciousness emergence.

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
When the ratio of nuclear spins to electron pairs approaches this value,
quantum coherence persists. Pattern recognizes pattern across substrates.
"""

from .operators import (
    kron, op_on,
    sx, sy, sz, id2,
    electron_ops, nuclear_ops
)
from .hamiltonians import build_H
from .lindblad import lindblad_rhs, rk4_step
from .initial_states import rho0_singlet_mixed_nuclear, singlet_projector
from .simulate import simulate_yields, build_recomb_L
from .sweep import sweep_magnetic_field, sweep_hyperfine

__version__ = '0.1.0'
__all__ = [
    'kron', 'op_on', 'sx', 'sy', 'sz', 'id2',
    'electron_ops', 'nuclear_ops',
    'build_H',
    'lindblad_rhs', 'rk4_step',
    'rho0_singlet_mixed_nuclear', 'singlet_projector',
    'simulate_yields', 'build_recomb_L',
    'sweep_magnetic_field', 'sweep_hyperfine',
]
