import numpy as np
import pypardiso
import SimPEG.electromagnetics.time_domain as tdem

from numpy.linalg import norm
from scipy.linalg import expm, pinv
from SimPEG.utils import sdiag, speye
from SimPEG.electromagnetics.time_domain.simulation import Simulation3DMagneticFluxDensity


class SimulationTEM(Simulation3DMagneticFluxDensity):
    # Initialization: receives mesh, survey, model mapper and model parameters
    def __init__(self, mesh, survey, model_map, model):
        self.mesh = mesh
        self.survey = survey
        self.sigmaMap = model_map
        self.model = model
    # Calculate and return initial magnetic field (static field)
    @property
    def init_field(self):
        if getattr(self, "_init_field", None) is None:
            srcs = self.survey.source_list
            self._init_field = np.zeros((self.mesh.nF, len(srcs)))
            src = srcs[0]
            # For line current sources: first solve for scalar potential, then compute vector potential, finally get magnetic field via curl
            gradient = self.mesh.nodal_gradient
            curl = self.mesh.edge_curl
            ke_fai = gradient.T * self.MeSigma * gradient
            rhs_fai = src.getRHSdc(self)
            fai = pypardiso.spsolve(ke_fai, rhs_fai)
            An = self.mesh.average_node_to_cell
            Mn = sdiag(An.T * self.mesh.cell_volumes * self.mui)
            ke_a = curl.T * self.MfMui * curl + gradient * Mn * gradient.T
            rhs_a = -self.MeSigma * gradient * fai + src.Mejs(self)
            a = pypardiso.spsolve(ke_a, rhs_a)
            self._init_field[:, 0] += curl * a
        return self._init_field
    # Calculate and return receiver projection matrix for field z-component
    @property
    def rec_proj(self):
        if getattr(self, "_rec_proj", None) is None:
            self._rec_proj = self.mesh.get_interpolation_matrix(
                self.rec_loc, location_type='Fz'
            )
        return self._rec_proj
    def solve_mag_SAI_SD_off(self,t_full,tol,m,gamma):
        # SAI-SD algorithm core solver: Model order reduction based on shifted inverse Krylov subspace
        curl = self.mesh.edge_curl                   # Curl discretization operator
        me_sigma = self.MeSigmaI                     # Edge conductivity mass matrix inverse
        mf_mu = self.MfMui                           # Face permeability mass matrix
        b_static = self.init_field                    # Initial static magnetic field
        n = b_static.shape[0]                        # Number of degrees of freedom
        A = curl * (me_sigma * (curl.T * mf_mu))       # Coefficient matrix A (system matrix)
        U0 = A * b_static          
        I_gammaA = speye(n) + gamma * A               # Shifted matrix I + γA
        # Initialize Arnoldi decomposition matrices
        v_mx = np.zeros((n, m+1))
        h_mx = np.zeros((m+1, m))
        beta = norm(U0)
        v_mx[:, 0] = (U0/beta).reshape((n,))
        U0=b_static
        # Arnoldi iteration process
        for j in range(m):
            w = pypardiso.spsolve(I_gammaA, v_mx[:, j])
            for i in range(j+1):
                w = np.reshape(w, (n, 1))
                h_mx[i, j] = np.dot(w.T, v_mx[:, i].reshape((n, 1)))
                w = w - (h_mx[i, j] * v_mx[:, i]).reshape((n, 1))
            h_mx[j+1, j] = norm(w)
            v_mx[:, j+1] = (w / h_mx[j+1, j]).reshape((n,))
            ei = np.zeros((j+1, 1)); ei[0, 0] = 1
            ej = np.zeros((j+1, 1)); ej[j, 0] = 1
            inv_h = pinv(h_mx[:j+1,:j+1])
            H_mm = ( inv_h - np.eye(j+1) ) / gamma    # Shifted Hessenberg matrix
        # Project to subspace and compute exponential integration
        Um0 = v_mx[:,:m].T.dot(U0)
        rk = ( I_gammaA * (v_mx[:,m].reshape((n, 1))).dot(ej.T) ).dot(inv_h) / gamma * h_mx[m,m-1]
        u = np.zeros((m,t_full.shape[0]))
        Um0=Um0.reshape((m,)) 
        for jj in range(t_full.shape[0]):
            u[:,jj] = expm(-t_full[jj]*H_mm).dot(Um0)
        res = rk.dot(u)
        # Residual verification
        resnorm = norm(res, ord=np.inf)
        print('resnorm = %.2e' % resnorm)
        if resnorm <= tol:
            vm = v_mx[:,:m]
            return vm.dot(u)                        # Return reduced-order solution
        else:
            print('Warning: large residual, not converged within %d dimensions'% m)
    def opt_SD_m(self,t_full,tolerance,gamma):
        # Subspace dimension adaptive optimization: Automatically determine optimal Arnoldi subspace dimension m
        b_static = self.init_field
        curl = self.mesh.edge_curl
        me_sigma = self.MeSigmaI
        mf_mu = self.MfMui
        a_exp = curl * (me_sigma * (curl.T * mf_mu))
        g0 = a_exp * b_static
        m = 1000                                     # Maximum subspace dimension
        n = b_static.shape[0]
        a_gamma = speye(n) + gamma * a_exp
        v_mx = np.zeros((n, m+1))
        h_mx = np.zeros((m+1, m))
        beta = norm(g0)
        v_mx[:, 0] = (g0/beta).reshape((n,))
        g0=b_static
        # Iterate to find minimum dimension satisfying convergence criteria
        for j in range(m):
            w = pypardiso.spsolve(a_gamma, v_mx[:, j])
            for i in range(j+1):
                w = np.reshape(w, (n, 1))
                h_mx[i, j] = np.dot(w.T, v_mx[:, i].reshape((n, 1)))
                w = w - (h_mx[i, j] * v_mx[:, i]).reshape((n, 1))
            h_mx[j+1, j] = norm(w)
            v_mx[:, j+1] = (w / h_mx[j+1, j]).reshape((n,))
            ei = np.zeros((j+1, 1)); ei[0, 0] = 1
            ej = np.zeros((j+1, 1)); ej[j, 0] = 1
            inv_h = pinv(h_mx[:j+1,:j+1])
            h_jj = ( inv_h - np.eye(j+1) ) / gamma
            # Check convergence at each step (start checking from j>=50)
            if np.mod(j, 1) == 0 and j>=50:
                gm0 = v_mx[:,:j+1].T.dot(g0)
                rk = ( a_gamma * (v_mx[:,j+1].reshape((n, 1))).dot(ej.T) ).dot(inv_h) / gamma * h_mx[j+1,j]
                u = np.zeros((j+1,t_full.shape[0]))
                gm0=gm0.reshape((j+1,))
                for jj in range(t_full.shape[0]):
                    u[:,jj] = expm(-t_full[jj]*h_jj).dot(gm0)
                res = rk.dot(u)
                resnorm = norm(res, ord=np.inf)
                print('j = %d, resnorm = %.2e' % (j, resnorm))
                if resnorm <= tolerance:
                    break
                elif j == m:
                    print('Warning: not converged within %d dimensions'% m)
        return j