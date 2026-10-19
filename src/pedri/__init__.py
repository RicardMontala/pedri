# pedri/__init__.py

from .base_experiment import Experiment
from .optimizer import BaseOptimizer, BayesianOptimizer
from .solver import BaseSolver
from .parametrization import BaseParametrization, BezierParametrization

__all__ = [
    "Experiment",
    "BaseOptimizer",
    "BayesianOptimizer",
    "BaseSolver",
    "NeuralFoilSolver",
    "SOD2DSolver",
    "BaseParametrization",
    "BezierParametrization"
]

def __getattr__(name):
    if name == "NeuralFoilSolver":
        from .solver import NeuralFoilSolver
        return NeuralFoilSolver

    if name == "SOD2DSolver":
        from .solver import SOD2DSolver
        return SOD2DSolver

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
