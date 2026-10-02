import glob
import shutil
import numpy as np

from smartsim import Experiment
from smartsim.settings import MpirunSettings

import logging
logger = logging.getLogger(__name__)


class SOD2DLauncher:

    def __init__(self,
        smartsim_launcher="local",
        experiment_name="sod2d_exp",
        sod_exe="sod2d",
        n_tasks_per_env=1,
        sod_JSON="BluffBodySolverIncompAFC_pedri",
        N_CFDs=1,
        control_parameters=None,
        cwd="./",
    ):

        logger.info("Starting SmartSim...")

        if control_parameters is None:
            control_parameters = np.zeros((N_CFDs, 3))

        # Store configuration
        self.experiment_name = experiment_name
        self.sod_exe = sod_exe
        self.n_tasks_per_env = n_tasks_per_env
        self.sod_JSON = sod_JSON
        self.N_CFDs = N_CFDs
        self.control_parameters = control_parameters
        self.cwd = cwd
        self.ensemble = None

        # List of launchers implemented/tested
        launchers = ["local", "slurm"]
        if smartsim_launcher not in launchers:
            raise ValueError(f"Launcher type '{smartsim_launcher}' not implemented. Please use one of these: {launchers}")

        # Generate SmartSim experiment
        self.smartsim_experiment = Experiment(experiment_name, launcher=smartsim_launcher)


    def _generate_mpmd_settings(self):

        if self.control_parameters.shape[0] != self.N_CFDs:
            raise ValueError(f"Number of allocated CFD simulations ({self.N_CFDs}) "
                             f"not equal to the number of parameter sets to evaluate "
                             f"({self.control_parameters.shape[0]})")


        for i in range(self.N_CFDs):

            # Set execution arguments
            control_parameters_str = ",".join(str(v) for v in self.control_parameters[i,:])
            sod_args = {"N_control_parameters": self.control_parameters.shape[1], "control_parameters": control_parameters_str}
            exe_args = [self.sod_JSON, *[f"--{k}={v}" for k, v in sod_args.items()]]

            logger.debug(f"Exe arguments: {exe_args}")

            # Set smartsim execution settings
            run = MpirunSettings(exe=self.sod_exe, exe_args=exe_args,)
            run.set_tasks(self.n_tasks_per_env)

            if i == 0:
                f_mpmd = run
            else:
                f_mpmd.make_mpmd(run)

        return f_mpmd


    def _create_mpmd_ensemble(self):
        f_mpmd = self._generate_mpmd_settings()

        return self.smartsim_experiment.create_model("ensemble", f_mpmd, batch_settings=None, path=self.cwd)


    def _stop_exp(self):

        if self.smartsim_experiment:
            # Stop ensembles
            self.smartsim_experiment.stop(self.ensemble)

            # Move ensemble log files to the "experiment_name" folder
            for f in glob.glob("ensemble-*"):
                shutil.move(f, f"{self.experiment_name}/")


    def start(self, block=True):
        # block=True keeps the Python process waiting until the simulations finish

        logger.info("Preparing ensemble of executions...")
        if not self.ensemble:
            self.ensemble = self._create_mpmd_ensemble()
            logger.info(f"New ensamble created")
        else:
            self.ensemble.run_settings = self._generate_mpmd_settings()
            logger.info(f"Updated ensamble args for SOD2D simulations")

        logger.info("Launching simulations...")
        self.smartsim_experiment.start(self.ensemble, block=block)
        logger.info("Simulations finished")

        self._stop_exp()