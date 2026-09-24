from smartsim import Experiment
from smartsim.log import get_logger
from smartsim.settings import MpirunSettings


logger = get_logger(__name__)


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
            control_parameters = [0.0, 0.0, 0.0]

        # Store configuration
        self.sod_exe = sod_exe
        self.n_tasks_per_env = n_tasks_per_env
        self.sod_JSON = sod_JSON
        self.N_CFDs = N_CFDs
        self.control_parameters = control_parameters
        self.cwd = cwd

        # List of launchers implemented/tested
        launchers = ["local", "slurm"]
        if smartsim_launcher not in launchers:
            raise ValueError(f"Launcher type '{smartsim_launcher}' not implemented. Please use one of these: {launchers}")

        # Generate SmartSim experiment
        self.smartsim_experiment = Experiment(experiment_name, launcher=smartsim_launcher)


    def _create_mpmd_ensemble(self):

        for i in range(self.N_CFDs):

            # Set execution arguments
            control_parameters_str = ",".join(str(v) for v in self.control_parameters)
            sod_args = {"N_control_parameters": len(self.control_parameters), "control_parameters": control_parameters_str}
            exe_args = [self.sod_JSON, *[f"--{k}={v}" for k, v in sod_args.items()]]

            logger.info(f"Exe arguments: {exe_args}")

            # Set smartsim execution settings
            run = MpirunSettings(exe=self.sod_exe, exe_args=exe_args,)
            run.set_tasks(self.n_tasks_per_env)

            if i == 0:
                f_mpmd = run
            else:
                f_mpmd.make_mpmd(run)

        return self.smartsim_experiment.create_model("ensemble", f_mpmd, batch_settings=None, path=self.cwd)


    def start(self, block=True):
        # block=True keeps the Python process waiting until the simulations finish

        logger.info("Preparing ensemble of executions...")
        ensemble = self._create_mpmd_ensemble()

        logger.info("Launching simulations...")
        self.smartsim_experiment.start(ensemble, block=block)