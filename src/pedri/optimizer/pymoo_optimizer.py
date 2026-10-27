import importlib
import logging
import numpy as np

# https://pymoo.org/index.html
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.problems.static import StaticProblem
from pymoo.core.termination import NoTermination
from pymoo.core.algorithm import Algorithm
from pymoo.core.repair import Repair

from pedri.optimizer.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


# Already tested classes from pymoo: https://pymoo.org/algorithms/list.html
# Strict mapping: "alias": ("module.path", "ClassName")
ALGORITHM_REGISTRY = {
    # Single-objective algorithms (soo.nonconvex)
    "ga": ("pymoo.algorithms.soo.nonconvex.ga", "GA"),
    "de": ("pymoo.algorithms.soo.nonconvex.de", "DE"),
    "pso": ("pymoo.algorithms.soo.nonconvex.pso", "PSO"),
    "brkga": ("pymoo.algorithms.soo.nonconvex.brkga", "BRKGA"),
    #"nelder": ("pymoo.algorithms.soo.nonconvex.nelder", "NelderMead"),
    #"pattern": ("pymoo.algorithms.soo.nonconvex.pattern", "PatternSearch"),
    "cmaes": ("pymoo.algorithms.soo.nonconvex.cmaes", "CMAES"),
    "es": ("pymoo.algorithms.soo.nonconvex.es", "ES"),

    # Multi-objective algorithms (moo)
}


def _load_algorithm_class(alg_key: str) -> type:
    """
    Dynamically imports and returns the algorithm class from the registry.
    """

    if alg_key not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unsupported algorithm '{alg_key}'. Available options are: {list(ALGORITHM_REGISTRY.keys())}")

    module_path, class_name = ALGORITHM_REGISTRY[alg_key]
    
    try:
        mod = importlib.import_module(module_path)
        logger.info(f"{class_name} algorithm from pymoo selected.")
        return getattr(mod, class_name)

    except (ImportError, AttributeError) as err:
        raise ImportError(f"Failed to load '{class_name}' from '{module_path}'.") from err


class ParameterTypeRepair(Repair):
    """
    Selective repair operator that rounds designated continuous variables to nearest integers
    when "discrete" parameters are defined.

    Imitates pymoo's native RoundingRepair, applied per-column based on parameter types:
    https://github.com/anyoptimization/pymoo/blob/main/pymoo/operators/repair/rounding.py
    """
    def __init__(self, parameter_names: list[str], parameter_types: dict[str, str],**kwargs) -> None:
        super().__init__(**kwargs)
        self.parameter_names = parameter_names
        self.parameter_types = parameter_types

        # Pre-compute integer column indices for fast vectorized rounding
        self.discrete_indices = np.array([
            i for i, name in enumerate(parameter_names)
            if parameter_types[name] == "discrete"
        ], dtype=int)

    # This is called after the creation of each new population if this Repair operator is provided
    def _do(self, problem, Z, **kwargs):
        # Extract X (handles both Population container and raw 2D array)
        X = Z.get("X") if hasattr(Z, "get") else Z

        # Vectorized rounding across all integer columns simultaneously
        if len(self.discrete_indices) > 0:
            X[:, self.discrete_indices] = np.around(X[:, self.discrete_indices])

        return Z


class PymooOptimizer(BaseOptimizer):
    """
    Generic adapter for single-objective pymoo algorithms.

    Barely any single-objective alogrithm defined in that library
    can be used: https://pymoo.org/algorithms/list.html

    The majority of optimization algorithms implemented
    are evolutionary and population-based.

    We use the Ask and Tell interface of the library:
    algorithm.ask() -> returns a solution set to be evaluated.
    algorithm.tell(solutions) -> provide evaluations to next generation.
    """

    def __init__(
        self,
        pbounds: dict[str, tuple[float, float]],
        algorithm: str | Algorithm,
        parameter_types: dict[str, str] | None = None,
        convergence_threshold: float = 0.1,
        convergence_patience: int = 5,
        max_iterations: int = 50,
        seed: int | None = None,
        **algorithm_kwargs,             # Provide specific kwargs for the desired pymoo algorithm to use
    ):
        super().__init__(
            pbounds=pbounds,
            parameter_types=parameter_types,
            convergence_threshold=convergence_threshold,
            convergence_patience=convergence_patience,
            max_iterations=max_iterations,
        )

        self.seed = seed

        # Automatically inject Repair operator if any integer parameters are present
        if "discrete" in self.parameter_types.values() and "repair" not in algorithm_kwargs:
            algorithm_kwargs["repair"] = ParameterTypeRepair(
                parameter_names=self.parameter_names,
                parameter_types=self.parameter_types,
            )

        # Dynamically import class from registry if string, otherwise use the provided pymoo class (useful for checkpoint restart)
        if isinstance(algorithm, str):
            alg_key = algorithm.lower().strip()
            algorithm_cls = _load_algorithm_class(alg_key)
            self.algorithm = algorithm_cls(**algorithm_kwargs)
        else:
            self.algorithm = algorithm

        # Configure pymoo problem
        self.problem = Problem(
            n_var=self.n_parameters,
            n_obj=1,
            n_constr=0,
            xl=np.array([pbounds[n][0] for n in self.parameter_names]),
            xu=np.array([pbounds[n][1] for n in self.parameter_names]),
        )

        # Current population pymoo object
        self._current_pop = None

        # Setup algorithm only if it has not been initialized yet (fresh init).
        # pymoo uses `is_initialized` to track if setup() has been executed.
        if not getattr(self.algorithm, "is_initialized", False):
            self.algorithm.setup(
                self.problem,
                termination=NoTermination(),
                seed=seed,
                verbose=False,
            )


    def _fit_model(self) -> None:
        """Pymoo manages its own search state; no surrogate is fitted."""
        pass


    def suggest(self, n_samples: int | None = None) -> np.ndarray:
        """Request the next population of candidates from pymoo."""

        if self._current_pop is not None:
            raise RuntimeError("Pending candidates must be evaluated with update() first.")

        population = self.algorithm.ask()

        if population is None or len(population) == 0:
            raise RuntimeError("Pymoo returned no candidates.")

        X = np.asarray(population.get("X"), dtype=float)

        if n_samples is not None and len(X) != n_samples:
            logger.warning(
                "Pymoo returned %d candidates instead of the requested %d. "
                "Check if this is the intended behavior.",
                len(X),
                n_samples,
            )

        self._current_pop = population
        return X.copy()


    def _on_initialize(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ingest custom initial population / warm-start data into pymoo.
        """

        pop = Population.new("X", X)
        static_problem = StaticProblem(self.problem, F=(-y).reshape(-1, 1))
        Evaluator().eval(static_problem, pop)
        self.algorithm.tell(infills=pop)


    def _on_update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ingest newly evaluated candidates into pymoo and advance state.
        """

        if self._current_pop is None:
            raise RuntimeError("Call suggest() before update().")

        expected_X = np.asarray(self._current_pop.get("X"), dtype=float)

        if X.shape != expected_X.shape or not np.allclose(X, expected_X):
            raise ValueError("X must contain the candidates returned by the latest suggest().")

        if len(y) != len(expected_X):
            raise ValueError(f"Expected {len(expected_X)} evaluations, received {len(y)}.")

        # Pymoo minimizes, so negate the objective
        static_problem = StaticProblem(self.problem, F=(-y).reshape(-1, 1))
        Evaluator().eval(static_problem, self._current_pop)

        self.algorithm.tell(infills=self._current_pop)
        self._current_pop = None


    def _get_extra_state(self) -> dict:
        """
        Return specific properties specific from this optimizer
        in order to "save_checkpoint" correctly.
        """
        
        return {"algorithm": self.algorithm, "seed": self.seed, "_current_pop": self._current_pop}


    def _set_extra_state(self, state: dict) -> None:
        """
        Assign specific properties specific from this optimizer
        in order to "load_checkpoint" correctly.
        """

        self.algorithm = state["algorithm"]
        self.seed = state.get("seed")
        self._current_pop = state.get("_current_pop")


    @classmethod
    def _build_from_checkpoint(cls, checkpoint: dict) -> "PymooOptimizer":
        """
        Provide the specific class constructor for this optimizer so that
        the class can be properly restored when "load_checkpoint".
        """

        extra = checkpoint.get("extra_state", {})
        optimizer = cls(
            pbounds=checkpoint["pbounds"],
            algorithm=extra.get("algorithm"),  # Preserves fully restored algorithm object
            parameter_types=checkpoint["parameter_types"],
            convergence_threshold=checkpoint["convergence_threshold"],
            convergence_patience=checkpoint["convergence_patience"],
            max_iterations=checkpoint["max_iterations"],
            seed=extra.get("seed"),
        )

        # Restore pending candidate state (otherwise, it would be None with base initialization)
        optimizer._current_pop = extra.get("_current_pop")

        return optimizer