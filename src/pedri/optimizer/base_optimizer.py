from abc import ABC, abstractmethod
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """
    Abstract Base Class for all PEDRI optimizers.
    
    Provides standardized convergence checking, dataset tracking, and 
    checkpoint management for optimization runs.
    """

    def __init__(
        self,
        pbounds: dict[str, tuple[float, float]],
        parameter_types: dict[str, str] | None = None,
        convergence_threshold: float = 0.1,
        convergence_patience: int = 5,
        max_iterations: int = 50,
    ):
        """
        Parameters
        ----------
        pbounds: dict
            Dictionary with parameter names as keys and (min, max) bounds as values.
            e.g. {'A_DC': [-1.0, 1.0], 'A':[0, 0.5], 'f':[0.1, 1.0]}
        
        parameter_types: dict
            Dictionary with parameter names as keys and parameter type ('discrete' or
            'continuous') as values. If not provided (None), it sets continuous to all.
            e.g. {'A_DC': 'continuous', 'A': 'continuous', 'f': 'continuous'}

        convergence_threshold: float
            Relative change threshold (%) defining convergence.
    
        convergence_patience: int
            Number of consecutive iterations below threshold required to converge.

        max_iterations: int
            Maximum allowable optimization iterations.
        """

        self.pbounds = pbounds
        self.parameter_names = list(pbounds)
        self.n_parameters = len(pbounds)

        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.max_iterations = max_iterations

        # Manage parameter_types dictionary
        if parameter_types is None:
            parameter_types = {name: "continuous" for name in self.parameter_names}

        if set(parameter_types) != set(self.parameter_names):
            raise ValueError("parameter_types must contain exactly the same keys as pbounds.")

        valid_types = {"continuous", "discrete"}
        if not set(parameter_types.values()).issubset(valid_types):
            raise ValueError("parameter_types must be 'continuous' or 'discrete'.")

        self.parameter_types = parameter_types

        # Training dataset storage (stored as numpy arrays at base level)
        self.training_X: np.ndarray | None = None
        self.training_y: np.ndarray | None = None

        # Convergence state
        self.no_improvement_count: int = 0
        self.best_history: list[float] = []


    # ------------------------------------------------------------------
    # Abstract Methods that must be implemented by the subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _fit_model(self) -> None:
        """
        Internal routine to train/refit the underlying surrogate model.
        """

        pass


    @abstractmethod
    def suggest(self, n_samples: int) -> np.ndarray:
        """
        Suggest the next batch of parameter candidates to evaluate.

        Parameters
        ----------
        n_samples: int
            Number of new evaluations requested.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_parameters) containing candidate points.
        """

        pass


    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict model target values and uncertainties for input candidates.

        Override this method only for optimizers that support surrogate predictions.
        
        Parameters
        ----------
        X: np.ndarray
           N evaluation points of shape (N, n_parameters).

        Returns
        -------
        mean: np.ndarray
            Predicted target values of shape (N,).
        
        std: np.ndarray
            Predicted uncertainties (standard deviations) of shape (N,).
        """

        raise NotImplementedError(f"{self.__class__.__name__} does not support predictions.")


    # ------------------------------------------------------------------
    # Shared functionalities across all subclasses
    # ------------------------------------------------------------------
    def initialize(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the optimizer with an initial dataset.
        
        Parameters
        ----------
        X: np.ndarray
            Set of parameter values to train the model (N_samples, n_parameters).
    
        y: np.ndarray
            Set of objective values to train the model (N_samples,).
        """

        self.training_X = np.asarray(X, dtype=np.float64)
        self.training_y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        self._fit_model()
        self.best_history.append(self.get_best()[1])


    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Update the optimizer with newly evaluated solver samples.
        
        Parameters
        ----------
        X: np.ndarray
            New set of parameter values to train the model (N_samples, n_parameters).
    
        y: np.ndarray
            New set of objective values to train the model (N_samples,).
        """
    
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        if self.training_X is None or self.training_y is None:
            self.initialize(X_arr, y_arr)
            return

        self.training_X = np.vstack([self.training_X, X_arr])
        self.training_y = np.vstack([self.training_y, y_arr])

        self._fit_model()
        self.best_history.append(self.get_best()[1])


    def get_best(self) -> tuple[np.ndarray, float]:
        """
        Return best observed parameters and objective value.
        
        Returns
        -------
        best_X: np.ndarray
            Parameters corresponding to the best observed objective, with shape (n_parameters,).

        best_y: float
            Best observed objective value
        """

        if self.training_y is None or len(self.training_y) == 0:
            raise RuntimeError("Optimizer has no evaluated dataset.")

        best_idx = np.argmax(self.training_y)
        return self.training_X[best_idx].copy(), float(self.training_y[best_idx].item())


    def get_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return complete evaluated parameter and objective history.
        
        Returns
        -------
        training_X: np.ndarray
            All observed parameter values with shape (N_samples, n_parameters).

        training_y: np.ndarray
            Observed objective values with shape (N_samples,).
        """

        if self.training_X is None or self.training_y is None:
            return np.empty((0, self.n_parameters)), np.empty((0,))

        return self.training_X.copy(), self.training_y.squeeze(-1).copy()


    def get_iter(self) -> int:
        """
        Return current iteration index (0 represents initial evaluation),
        i.e., [initial, iter1, iter2, iter3, ...].
        """

        return max(0, len(self.best_history) - 1)


    # ------------------------------------------------------------------
    # Convergence Tracking
    # ------------------------------------------------------------------
    def check_convergence(self) -> bool:
        """
        Check whether convergence criterion or max iterations have been reached.
        
        Returns
        -------
        bool
           True if the convergence criterion has been reached.
            This is fulfilled when the "relative_change" is maintained
            below the "convergence_threshold" during more than "convergence_patience"
            BO iterations.
        """

        if len(self.best_history) < 2:
            logger.info("Convergence check skipped: single evaluation state.")
            return False

        best_old = self.best_history[-2]
        best_new = self.best_history[-1]

        relative_change = 100.0 * abs(best_old - best_new) / max(abs(best_old), 1e-12)
        logger.info(
            f"Best objective: {best_new:.6e}. Relative change with previous best: {relative_change:.3f}%"
        )

        if relative_change < self.convergence_threshold:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0

        if self.no_improvement_count >= self.convergence_patience:
            logger.info("CONVERGENCE CRITERIA REACHED!")
            return True

        if self.get_iter() >= self.max_iterations:
            logger.info("MAXIMUM NUMBER OF ITERATIONS REACHED!")
            return True

        return False


    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str = "./", dump2csv: bool = False) -> None:
        """Save optimizer state using NumPy formatting, including the training dataset,
        parameter bounds, and current iteration. The file is called checkpoint.pt.
        
        Parameters
        ----------
        path: sting, optional
            Directory where the checkpoint file will be stored.

        dump2csv: bool, optional
            If True, also save the training dataset to a CSV file.        

        """

        os.makedirs(path, exist_ok=True)
        checkpoint_data = {
            "class_name": self.__class__.__name__,
            "pbounds": self.pbounds,
            "training_X": self.training_X,
            "training_y": self.training_y,
            "best_history": self.best_history,
            "no_improvement_count": self.no_improvement_count,
            "convergence_threshold": self.convergence_threshold,
            "convergence_patience": self.convergence_patience,
            "max_iterations": self.max_iterations,
        }
        np.save(os.path.join(path, "checkpoint.npy"), checkpoint_data)

        if dump2csv and self.training_X is not None:
            X, y = self.get_dataset()
            data = np.column_stack((X, y))
            header = list(self.pbounds.keys()) + ["objective"]
            np.savetxt(
                os.path.join(path, "checkpoint.csv"),
                data,
                delimiter=",",
                header=",".join(header),
            )

        logger.info(f"Optimization saved at iteration {self.get_iter()}.")


    @classmethod
    def load_checkpoint(cls, path: str = "./") -> "BaseOptimizer":
        """
        Universal checkpoint loader. Reads saved state and restores 
        the specific optimizer subclass automatically.

        Parameters
        ----------
        path: str, optional
            Directory where the checkpoint file is stored.
        """

        npy_path = os.path.join(path, "checkpoint.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"No checkpoint file found at {npy_path}")

        # Load dictionary object from .npy file
        checkpoint = np.load(npy_path, allow_pickle=True).item()

        # Target class resolution
        target_class_name = checkpoint.get("class_name", "BayesianOptimizer")
        subclasses = {sub.__name__: sub for sub in cls.__subclasses__()}
        target_cls = subclasses.get(target_class_name, cls)

        # Instantiate concrete subclass
        optimizer = target_cls(
            pbounds=checkpoint["pbounds"],
            convergence_threshold=checkpoint["convergence_threshold"],
            convergence_patience=checkpoint["convergence_patience"],
            max_iterations=checkpoint["max_iterations"],
        )

        # Restore state
        optimizer.training_X = checkpoint["training_X"]
        optimizer.training_y = checkpoint["training_y"]
        optimizer.best_history = checkpoint["best_history"]
        optimizer.no_improvement_count = checkpoint["no_improvement_count"]

        # Re-fit surrogate model if applicable
        optimizer._fit_model()

        logger.info(f"Restored {optimizer.__class__.__name__} from iteration {optimizer.get_iter()}.")

        return optimizer