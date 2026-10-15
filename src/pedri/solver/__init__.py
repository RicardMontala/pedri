from .base_solver import BaseSolver

__all__ = [
    "BaseSolver",
    "NeuralFoilSolver",
    "SOD2DSolver",
]

def __getattr__(name):
    if name == "NeuralFoilSolver":
        from .neuralfoil_solver import NeuralFoilSolver
        return NeuralFoilSolver

    if name == "SOD2DSolver":
        from .sod2d_solver import SOD2DSolver
        return SOD2DSolver

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")