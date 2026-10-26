import logging
import numpy as np

# https://pymoo.org/index.html
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.problems.static import StaticProblem
from pymoo.core.termination import NoTermination

from pedri.optimizer.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


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
        algorithm,
        parameter_types: dict[str, str] | None = None,
        convergence_threshold: float = 0.1,
        convergence_patience: int = 5,
        max_iterations: int = 50,
        seed: int | None = None,
    ):
        super().__init__(
            pbounds=pbounds,
            parameter_types=parameter_types,
            convergence_threshold=convergence_threshold,
            convergence_patience=convergence_patience,
            max_iterations=max_iterations,
        )

        self.algorithm = algorithm
        self.seed = seed

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
            raise RuntimeError(
                "Pending candidates must be evaluated with update() first."
            )

        population = self.algorithm.ask()

        if population is None or len(population) == 0:
            raise RuntimeError("Pymoo returned no candidates.")

        X = np.asarray(population.get("X"), dtype=float)

        if n_samples is not None and len(X) != n_samples:
            logger.warning(
                "Pymoo returned %d candidates instead of the requested %d. "
                "The algorithm controls its population size.",
                len(X),
                n_samples,
            )

        self._current_pop = population
        return X.copy()


    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Submit solver evaluations and advance the pymoo algorithm."""

        if self._current_pop is None:
            raise RuntimeError("Call suggest() before update().")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        expected_X = np.asarray(self._current_pop.get("X"), dtype=float)

        if X.shape != expected_X.shape or not np.allclose(X, expected_X):
            raise ValueError(
                "X must contain the candidates returned by the latest suggest()."
            )

        if y.shape[0] != len(expected_X):
            raise ValueError("One objective value is required per candidate.")

        if not np.all(np.isfinite(y)):
            raise ValueError("Objective values must be finite.")

        # Keep PEDRI's objective convention: maximize y.
        if self.training_X is None:
            self.training_X = X.copy()
            self.training_y = y.reshape(-1, 1)
        else:
            self.training_X = np.vstack([self.training_X, X])
            self.training_y = np.vstack(
                [self.training_y, y.reshape(-1, 1)]
            )

        # Pymoo minimizes, so negate the objective.
        static_problem = StaticProblem(
            self.problem,
            F=(-y).reshape(-1, 1),
        )

        Evaluator().eval(static_problem, self._current_pop)
        self.algorithm.tell(infills=self._current_pop)
        self._current_pop = None

        self.best_history.append(self.get_best()[1])
        self._fit_model()


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