import os
import logging
import numpy as np
from typing import Callable, Any
from scipy.stats import qmc

from pedri.optimizer.base_optimizer import BaseOptimizer
from pedri.solver.base_solver import BaseSolver

logger = logging.getLogger(__name__)


class Experiment:
    """
    Orchestrates the PEDRI optimization pipeline by tying together an Optimizer
    and a Solver into an automated iterative optimization loop.
    """

    def __init__(
        self,
        optimizer: BaseOptimizer,
        solver: BaseSolver,
        batch_size: int = 1,
        experiment_name: str = "pedri_exp",
        checkpoint_dir: str = "./checkpoints",
        save_checkpoint_step: int = 1,
        on_iteration_end: Callable[["Experiment"], None] | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        optimizer: BaseOptimizer
            Configured PEDRI optimizer instance.
        solver: BaseSolver
            Configured PEDRI solver instance.
        batch_size: int
            Number of candidate evaluations per iteration.
        experiment_name: str
            Identifier for logging and checkpoint naming.
        checkpoint_dir: str
            Directory path to store checkpoint state.
        save_checkpoint_step: int
            Frequency (in iterations) at which checkpoints are saved.
        on_iteration_end: Callable, optional
            Callback hook executed at the end of every iteration (e.g., for plotting).
        seed: int, optional
            Fix the desired seed when generating initial samples (ensure reproducibility across runs).
        """
        self.optimizer = optimizer
        self.solver = solver
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoint_step = save_checkpoint_step
        self.on_iteration_end = on_iteration_end
        self.seed = seed

        # Internal history of raw solver outputs (Cl, Cd, etc.)
        self.history_metrics: list[dict[str, np.ndarray]] = []


    def generate_initial_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate parameter candidates across bounds using Latin Hypercube Sampling (LHS)
        (good coverage of the parameter space).

        Parameters
        ----------
        n_samples: int
            Number of initial candidates to generate.
        seed: int
            Controls the randomness of the LHS.

        Returns
        -------
        candidates: np.ndarray
            Returns the candidates with shape (n_samples, d), where d is the
            dimension of the parameter space (defined by the optimizer bounds).
        """

        pbounds = self.optimizer.pbounds
        sampler = qmc.LatinHypercube(d=len(pbounds), seed=self.seed)
        samples = sampler.random(n=n_samples)

        bounds_min = np.array([b[0] for b in pbounds.values()])
        bounds_max = np.array([b[1] for b in pbounds.values()])

        return qmc.scale(samples, bounds_min, bounds_max)


    def initialize(self, X_init: np.ndarray | None = None, n_initial_samples: int | None = None) -> None:
        """
        Initialize the experiment dataset. Runs solver on initial parameter set
        if optimizer has no existing data.
        """

        if self.optimizer.training_X is not None and len(self.optimizer.training_X) > 0:
            logger.info("Optimizer already initialized with existing dataset. Skipping initial sampling.")
            return

        if X_init is None:
            if n_initial_samples is None:
                raise ValueError("Must provide either X_init or n_initial_samples for initialization.")
            logger.info(f"Generating {n_initial_samples} initial samples via Latin Hypercube Sampling...")
            X_init = self.generate_initial_samples(n_initial_samples)

        logger.info(f"Evaluating {len(X_init)} initial samples with solver...")
        y_init, metrics = self.solver.evaluate(X_init)

        self.optimizer.initialize(X_init, y_init)
        self.history_metrics.append(metrics)

        self.optimizer.save_checkpoint(path=self.checkpoint_dir, dump2csv=True)
        logger.info("Initial dataset created and checkpoint saved.")


    def step(self) -> bool:
        """
        Run a single optimization iteration:
        1. Suggest next candidates
        2. Evaluate candidates using solver
        3. Update optimizer state
        4. Check convergence
        """

        iteration = self.optimizer.get_iter() + 1
        logger.info(f"\n\n--- ITERATION {iteration} ---")

        # 1. Suggest candidate parameters
        new_X = self.optimizer.suggest(self.batch_size)
        logger.info(f"Suggested {len(new_X)} candidate points.")

        # 2. Evaluate with solver
        new_y, metrics = self.solver.evaluate(new_X)
        logger.info("Solver evaluation completed.")

        # 3. Update optimizer
        self.optimizer.update(new_X, new_y)
        self.history_metrics.append(metrics)

        # 4. Save checkpoint according to frequency
        if iteration % self.save_checkpoint_step == 0:
            self.optimizer.save_checkpoint(path=self.checkpoint_dir, dump2csv=True)

        # 5. Trigger user callback if defined (e.g. for custom plotting)
        if self.on_iteration_end is not None:
            self.on_iteration_end(self)

        # 6. Return convergence status
        return self.optimizer.check_convergence()


    def run(self, n_initial_samples: int | None = None) -> tuple[np.ndarray, float]:
        """
        Execute full optimization workflow until convergence or max iterations.

        Returns
        -------
        best_X : np.ndarray
            Best parameter vector found.
        best_y : float
            Best objective value achieved.
        """
        # Auto-initialize if required
        if self.optimizer.training_X is None:
            self.initialize(n_initial_samples=n_initial_samples)

        converged = False
        while not converged:
            converged = self.step()

        best_X, best_y = self.optimizer.get_best()
        logger.info(f"\nOptimization Finished! Best Objective: {best_y:.6e}")
        return best_X, best_y