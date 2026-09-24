from smartsim.log import get_logger

from smartsim import Experiment

from smartsim.settings import MpirunSettings
from smartsim.settings.settings import create_batch_settings


logger = get_logger(__name__)


def init_smartsim(smartsim_launcher = "local"):

    logger.info("Starting SmartSim...")

    # List of launchers implemented/tested
    launchers = ["local", "slurm"]
    if smartsim_launcher not in launchers:
        raise ValueError(f"Launcher type '{smartsim_launcher}' not implemented. Please use one of these: {launchers}")

    # Generate Sod2D experiment
    exp = Experiment("sod2d_exp", launcher=smartsim_launcher)

    return exp


def _generate_mpmd_settings(N_CFDs, control_parameters, sod_exe, n_tasks_per_env, sod_JSON):
    # Set SOD2D exe arguments
    logger.info("Preparing execution arguments...")

    for i in range(N_CFDs):
            control_parameters_str = ",".join(str(v) for v in control_parameters)

            sod_args = {"N_control_parameters": len(control_parameters), "control_parameters": control_parameters_str}
            exe_args = [sod_JSON] + [f"--{k}={v}" for k,v in sod_args.items()]

            logger.info(f"Exe arguments: {exe_args}")

            run = MpirunSettings(exe=sod_exe, exe_args=exe_args)
            run.set_tasks(n_tasks_per_env)
            if i == 0:
                f_mpmd = run
            else:
                f_mpmd.make_mpmd(run)
    return f_mpmd


def create_mpmd_ensemble(exp, N_CFDs, control_parameters, sod_exe, n_tasks_per_env, sod_JSON, cwd):

    f_mpmd = _generate_mpmd_settings(N_CFDs, control_parameters, sod_exe, n_tasks_per_env, sod_JSON)
    batch_settings = None

    return exp.create_model("ensemble", f_mpmd, batch_settings=batch_settings, path=cwd)