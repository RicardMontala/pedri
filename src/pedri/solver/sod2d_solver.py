from typing import Callable
import json
import os
import glob
import shutil
import numpy as np

from smartsim import Experiment
from smartsim.settings import MpirunSettings

from pedri.solver.base_solver import BaseSolver

import logging
logger = logging.getLogger(__name__)


class SOD2DSolver(BaseSolver):
    """
    SOD2D CFD Solver using SmartSim for job orchestration.

    This solver launches SOD2D simulations via SmartSim and postprocesses
    the results to extract aerodynamic coefficients (Cl, Cd, Cl_std, Cd_std).
    """

    def __init__(
        self,
        objective_fn: Callable[..., np.ndarray | float] | None = None,
        n_cfds: int = 1,
        n_tasks_per_cfd: int = 1,
        smartsim_launcher: str ="local",
        experiment_name: str ="sod2d_exp",
        sod_exe: str = "sod2d",
        sod_json: str = "BluffBodySolverIncompAFC_pedri",
        cwd: str = "./",  
        surf2analyse: int = 1,  # json
        Lz: float = 1.0,

    ):
        """
        Parameters
        ----------
        objective_fn: Callable, optional
            Mapping function from simulations raw results to objective y values.
        n_cfds: int
            Number of parallel CFD simulations.
        n_tasks_per_cfd: int
            Number of MPI tasks per simulation.
            n_tasks_per_cfd x n_cfds should match the number of allocated tasks.
        smartsim_launcher: str
            SmartSim launcher type ("local" or "slurm").
        experiment_name: str
            Name of the SmartSim experiment.
        sod_exe: str
            Path to SOD2D executable.
        sod_json: str
            Path to the JSON  configuration file used to execture sod2d (without .json).
        initial_avgTime: float
            
        """

        logger.info("Starting SmartSim...")

        super().__init__(objective_fn=objective_fn)

        self.smartsim_launcher = smartsim_launcher
        self.experiment_name = experiment_name
        self.sod_exe = sod_exe
        self.n_tasks_per_cfd = n_tasks_per_cfd
        self.sod_json = sod_json
        self.n_cfds = n_cfds
        self.surf2analyse = surf2analyse
        self.Lz = Lz
        self.cwd = cwd
        self.ensemble = None

        # Read JSON and set some of the parameters defined there
        with open(f"{self.sod_json}.json", "r") as f:
            config = json.load(f)

        self.mesh_name = config["mesh_h5_file_name"]
        self.initial_avgTime = config["initial_avgTime"] # Time to start averaging (trim transient)

        self.rho = config["rho0"]
        self.Uinf = config["v0"]
        self.Lref = config["delta"]
        self.aoa = np.deg2rad(config["aoa"])

        # List of launchers implemented/tested
        launchers = ["local", "slurm"]
        if smartsim_launcher not in launchers:
            raise ValueError(f"Launcher type '{smartsim_launcher}' not implemented. Please use one of these: {launchers}")

        # Suppress SmartSim messages
        logging.getLogger("SmartSim").setLevel(logging.CRITICAL)
        logging.getLogger("ensemble").setLevel(logging.CRITICAL)

        # Generate SmartSim experiment
        self.smartsim_experiment = Experiment(experiment_name, launcher=smartsim_launcher)


    def run(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Execute sod2d simulations for candidate parameters X.

        Parameters
        ----------
        X: np.ndarray
            Parameter matrix of shape (n_cfds, n_parameters) defining the cases to run.

        Returns
        -------
        dict_results: dict
            Dictionary containing the Cl, Cd, C'lrms, C'drms resutls for each set of X parameters.
            Thus, each keyword contains (n_cfds,) values.
        """

        if X.shape[0] != self.n_cfds:
            raise ValueError(f"Number of allocated CFD simulations ({self.n_cfds}) "
                             f"not equal to the number of parameter sets to evaluate ({X.shape[0]})")

        logger.info(f"Launching batch of {X.shape[0]} SOD2D simulations...")

        if not self.ensemble:
            self.ensemble = self._create_mpmd_ensemble(X)
            logger.info(f"New ensamble created")
        else:
            self.ensemble.run_settings = self._generate_mpmd_settings(X)
            logger.info(f"Ensamble args updated for SOD2D simulations")

        logger.info("Running simulations...")
        self.smartsim_experiment.start(self.ensemble, block=True) # If True, wait for simulations to complete.

        logger.info("Simulations finished")
        self._stop_exp()

        # Parse output forces from SOD2D output files
        dict_results = self.postprocess()

        return dict_results


    def _generate_mpmd_settings(self, X: np.ndarray) -> MpirunSettings:
        """
        Generate MPMD settings for the ensemble.

        X: np.ndarray
            Control parameters with shape (n_cfds, n_parameters) used to run
            the sod2d simulations with different set of parameters.
        """

        for i in range(self.n_cfds):

            # Set execution arguments
            X_str = ",".join(str(v) for v in X[i,:])
            sod_args = {"N_control_parameters": X.shape[1], "control_parameters": X_str}
            exe_args = [self.sod_json, *[f"--{k}={v}" for k, v in sod_args.items()]]

            logger.debug(f"Exe arguments: {exe_args}")

            # Set smartsim execution settings
            run = MpirunSettings(exe=self.sod_exe, exe_args=exe_args,)
            run.set_tasks(self.n_tasks_per_cfd)

            if i == 0:
                f_mpmd = run
            else:
                f_mpmd.make_mpmd(run)

        return f_mpmd


    def _create_mpmd_ensemble(self, X: np.ndarray):
        """
        Create the MPMD ensemble.
        """

        f_mpmd = self._generate_mpmd_settings(X)

        return self.smartsim_experiment.create_model("ensemble", f_mpmd, batch_settings=None, path=self.cwd)


    def _stop_exp(self) -> None:
        """
        Stop experiment and move log files.
        """

        if self.smartsim_experiment:
            # Stop ensembles
            self.smartsim_experiment.stop(self.ensemble)

            # Move ensemble log files to the "experiment_name" folder
            for f in glob.glob("ensemble-*"):
                shutil.move(f, f"{self.experiment_name}/")


    def postprocess(self) -> dict[str, np.ndarray]:
        """
        Post-process simulation results and compute aerodynamic coefficients.

        Returns
        -------
        Results: dict
            Dictionary with Cl_avg, Cd_avg, Cl_std, Cd_std arrays for each
            simulation evaluated. Each coefficient array has shape (n_cfds,).
        """

        logger.info("Computing Cl and Cd from simulations...")

        Cl_avg = np.zeros(self.n_cfds)
        Cd_avg = np.zeros(self.n_cfds)
        Cl_std = np.zeros(self.n_cfds)
        Cd_std = np.zeros(self.n_cfds)

        for i in range(self.n_cfds):

            # Load surface file data [ITER, TIME, AREA, FPRE_X, FPRE_Y, FPRE_Z, FTAU_X, FTAU_Y, FTAU_Z]
            surf_path = os.path.join(self.cwd, f"output_{i}", f"surf_code_{self.surf2analyse}-{self.mesh_name}-{self.n_tasks_per_cfd}.dat")
            surf_data = np.loadtxt(surf_path, skiprows=1, delimiter=',')

            # Reference Surface and Dynamic pressure
            S_ref = self.Lref * self.Lz
            q_ref = 0.5 * self.rho * self.Uinf**2

            # Compute instantaneous Lift Coefficient
            Cl_pres = (-surf_data[:, 3] * np.sin(self.aoa) + surf_data[:, 4] * np.cos(self.aoa)) / (q_ref * S_ref)
            Cl_visc = (-surf_data[:, 6] * np.sin(self.aoa) + surf_data[:, 7] * np.cos(self.aoa)) / (q_ref * S_ref)
            Cl = Cl_pres + Cl_visc

            # Compute  instantaneous Drag Coefficient
            Cd_pres = (surf_data[:, 3] * np.cos(self.aoa) + surf_data[:, 4] * np.sin(self.aoa)) / (q_ref * S_ref)
            Cd_visc = (surf_data[:, 6] * np.cos(self.aoa) + surf_data[:, 7] * np.sin(self.aoa)) / (q_ref * S_ref)
            Cd = Cd_pres + Cd_visc

            # Compute average coefficients
            time_mask = surf_data[:, 1] >= self.initial_avgTime # Trim transient period

            Cl_avg[i] = np.mean(Cl[time_mask])
            Cd_avg[i] = np.mean(Cd[time_mask])
            Cl_std[i] = np.std(Cl[time_mask])
            Cd_std[i] = np.std(Cd[time_mask])

        return {
            "Cl": Cl_avg,
            "Cd": Cd_avg,
            "Cl_std": Cl_std,
            "Cd_std": Cd_std,
        }

