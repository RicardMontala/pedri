from typing import Callable, Any
import numpy as np

from .base_solver import BaseSolver


class AnalyticalSolver(BaseSolver):
    """
    Solver for analytical benchmark functions and synthetic test problems.
    
    When the evaluation function is already known and no surrogates/CFD evaluations
    are required.
    """

    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], float | np.ndarray | dict[str, Any]],
        vectorized: bool = False,
        objective_fn: Callable[..., np.ndarray | float] | None = None,
        output_name: str = "y",
    ):
        """
        Parameters
        ----------
        eval_fn: Callable
            Target mathematical function to evaluate. Accepts 1D array [x1, x2,..] or
            2D array [n_samples, n_params]=[[x1,x2,...],[x1,x2,...]] if vectorized=True.

        vectorized: bool, default=False
            If True, 'eval_fn(X)' accepts the full 2D matrix X of shape (n_samples, n_params).
            If False, 'eval_fn' is called row-by-row for each sample x_i in X.

        objective_fn: Callable, optional
            Function to map raw metric outputs from 'eval_fn(X)' to objective value y. 
            If None, defaults to passing through the primary output value directly.

            The output dictionary of 'eval_fn' should match the input dictionary of 'objective_fn'.

        output_name: str, default="y"
            Dictionary key used to store raw output values if 'eval_fn(X)' returns array values
            and not a dictionary. 
            
            If 'eval_fn(X)' contains multiple outputs, "1", "2",... would be appended to the output_name.
        """

        def default_obj(**kwargs):
            """
            Default objective function when 'objective_fn' is not provided.
                (1) When 'eval_fn' reuturns a single output the default_obj will be
                that output, and hence, would maximize the function 'eval_fn'.
                (2) When 'eval_fn' returns more than one output, then an explicit 'objective_fn'
                must be defined.
            """
            if output_name in kwargs:
                return kwargs[output_name]

            if len(kwargs) == 1:
                return next(iter(kwargs.values()))

            raise ValueError(
                "Multiple outputs were provided, but no objective_fn "
                f"was defined. Available outputs: {list(kwargs.keys())}"
            )

        # Default objective_fn: pass through raw function metric if no objective_fn is provided
        selected_obj = objective_fn or default_obj
        super().__init__(objective_fn=selected_obj)

        self.eval_fn = eval_fn
        self.vectorized = vectorized
        self.output_name = output_name


    def run(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Evaluate analytical function over candidate sample matrix X.
        
        Parameters
        ----------
        X: np.ndarray
            Matrix with shape (n_samples, n_parameters) containing several sets of parameters
            to evaluate.

        Returns
        -------
        raw_out: dict
            Raw values obtained after evaluating the function 'eval_fn(X)' with the
            parameters in X. It returns a dictionary with n keywords and with
            each keyword containing n_samples results, e.g. {'y1: [...], 'y2:[...], ...}
        """

        X_arr = np.asarray(X, dtype=np.float64)

        if self.vectorized:
            raw_out = self.eval_fn(X_arr)
        else:
            raw_out = [self.eval_fn(row) for row in X_arr]


        # Case 1: When the output of 'eval_fn' is already a dictionary.
        # It uses the keys of that dictionary.
        if len(raw_out) > 0 and isinstance(raw_out[0], dict):
            keys = raw_out[0].keys()
            return {
                k: np.array([item[k] for item in raw_out], dtype=np.float64)
                for k in keys
            }


        # Case 2: When 'eval_fn' returns directly an array of valuees per sample, or
        # multiple arrays of values per sample, without defining a dictionary.

        # if raw_out contains more than one ouput, it assigns "y1", "y2",... to each array of values.
        if isinstance(raw_out, tuple):
            return {
                f"{self.output_name}{i + 1}": np.asarray(y, dtype=np.float64)
                for i, y in enumerate(raw_out)
            }

        # if raw_out contains only one output, it assigns  "y" to those values.
        return {self.output_name: np.asarray(raw_out, dtype=np.float64)}