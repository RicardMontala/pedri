import numpy as np
import os
from smartsim.log import get_logger

logger = get_logger(__name__)


class SOD2DPostprocessor:

    def __init__(self,
        N_CFDs=1,
        mesh_name="mesh20_NACA",
        surf2analyse=1,
        n_tasks_per_env=1,
        initial_avgTime=0.0,
        rho=1.0,
        Uinf=1.0,
        Lref=1.0,
        Lz=1.0,
        AoA=0.0,
        cwd="./",
    ):

        # Store configuration
        self.N_CFDs=N_CFDs
        self.mesh_name=mesh_name
        self.surf2analyse=surf2analyse
        self.n_tasks_per_env=n_tasks_per_env
        self.initial_avgTime=initial_avgTime
        self.rho=rho
        self.Uinf=Uinf
        self.Lref=Lref
        self.Lz=Lz
        self.AoA=np.deg2rad(AoA)
        self.cwd=cwd
        

    def compute_ClCd(self):

        logger.info("Computing Cl and Cd from simulations...")

        Cl_avg = np.zeros(self.N_CFDs)
        Cd_avg = np.zeros(self.N_CFDs)

        for i in range(self.N_CFDs):

            # Load surface file data [ITER, TIME, AREA, FPRE_X, FPRE_Y, FPRE_Z, FTAU_X, FTAU_Y, FTAU_Z]
            surf_path = os.path.join(self.cwd, f"output_{i}", f"surf_code_{self.surf2analyse}-{self.mesh_name}-{self.n_tasks_per_env}.dat")
            surf_data = np.loadtxt(surf_path, skiprows=1, delimiter=',')

            # Reference Surface and Dynamic pressure
            S_ref = self.Lref*self.Lz
            q_ref =  0.5*self.rho*self.Uinf**2

            # Compute instantaneous Lift Coefficient
            Cl_pres = ( -surf_data[:,3]*np.sin(self.AoA) + surf_data[:,4]*np.cos(self.AoA) ) / (q_ref*S_ref)
            Cl_visc = ( -surf_data[:,6]*np.sin(self.AoA) + surf_data[:,7]*np.cos(self.AoA) ) / (q_ref*S_ref)
            Cl = Cl_pres + Cl_visc
            
            # Compute  instantaneous Drag Coefficient
            Cd_pres = ( surf_data[:,3]*np.cos(self.AoA) + surf_data[:,4]*np.sin(self.AoA) ) / (q_ref*S_ref)
            Cd_visc = ( surf_data[:,6]*np.cos(self.AoA) + surf_data[:,7]*np.sin(self.AoA) ) / (q_ref*S_ref)
            Cd = Cd_pres + Cd_visc

            # Compute average coefficients
            time_mask = surf_data[:,1] >= self.initial_avgTime # Trim transient period

            Cl_avg[i] = np.mean(Cl[time_mask])
            Cd_avg[i] = np.mean(Cd[time_mask])

        return Cl_avg, Cd_avg