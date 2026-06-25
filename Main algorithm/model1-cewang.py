print('Loading required Python modules......')
# Install required packages if missing
# Note: SimPEG version 0.18.0+ may cause Mfmui errors
import numpy as np
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain as tdem
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
import  warnings, time
import matplotlib as mpl
from matplotlib import font_manager
from discretize import TreeMesh
from discretize.utils import refine_tree_xyz,active_from_xyz
from SimPEG.utils import mkvc
from SimPEG import maps
from simulation import SimulationTEM
# from pyMorEM.simulation import SimulationTEM
warnings.filterwarnings("ignore")# Ignore minor warnings
print('Modules loaded successfully!!!')
# ==========================================================================================================
# Octree Mesh Generation: Local refinement based on octree grid
# ==========================================================================================================
stime=time.time()
print('Octree mesh generation started, timing begins, current time:',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
# Preparation: Define basic grid parameters
dh = 5                  # Minimum cell width (meters)
dom_width = 12000.0     # Total domain width (meters)
nbc = 2 ** int(np.round(np.log(dom_width/dh)/np.log(2.0)))  # Number of base cells (power of 2)
h = [(dh, nbc)]         # Grid spacing definition
mesh = TreeMesh([h, h, h], x0="CCC")  # Create octree mesh centered at origin

# Generate terrain control points (flat terrain at z=0)
xx, yy = np.meshgrid(np.linspace(-dom_width/2, dom_width/2, 5),
                            np.linspace(-dom_width/2, dom_width/2, 5))
zz = np.zeros(np.shape(xx))                  # Terrain elevation is 0
topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]  # Convert to Nx3 coordinate matrix

# Refine mesh based on terrain surface
mesh = refine_tree_xyz(
        mesh, topo_xyz, method="surface",octree_levels=[0,0,0,0,0,0,0,1], finalize=False
)
# ==========================================================================================================
print('Setting up receiver points......')
# Multiple receivers: 2D receiver grid layout
num_x=11                # Number of receivers along x-axis (11 receivers)
num_y=11                # Number of receivers along y-axis (2D grid)
xrx, yrx = np.meshgrid(np.linspace(-50, 50, num_x),
                     np.linspace(-50, 50, num_y))  # Receiver range: x,y∈[-50,50]m
zrx = np.zeros(np.shape(xrx))                  # Receivers located at surface (z=0)
number_cd=num_x*num_y                         # Total number of receivers
rec_loc = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]  # Convert to Nx3 receiver matrix
print('Receiver setup completed')
# ==========================================================================================================
# Define refinement regions: Multi-level refinement strategy
# ==========================================================================================================

# ============================================================
# Transmitter area and near-surface refinement: Gradually expanding refinement range
# ============================================================
print('Starting region refinement......')
octree_levels = []
for i in range(3):
    # Control refinement level for each layer, first layer has highest refinement
    octree_levels.append(1) if i == 0 else octree_levels.append(0)
    box_range =100 * 2 ** i       # Refinement region range (gradually expanding)
    air_range = 100*2**i          # Air layer range
    x_obj, y_obj, z_obj = np.meshgrid(
        [-box_range, box_range],
        [-box_range, box_range],
        [(-air_range), air_range]
    )
    xyz_obj = np.c_[mkvc(x_obj), mkvc(y_obj), mkvc(z_obj)]
    mesh = refine_tree_xyz(
        mesh, xyz_obj,
        octree_levels=list(reversed(octree_levels)),
        method='box',
        finalize=False,
        )
print('Region refinement completed')

# ============================================================
# Single block target definition and refinement in half-space
# ============================================================
blft=1  # Target refinement switch
if blft:
    # Target spatial range: Cubic anomaly buried at 50-130m depth
    numx1 = -40
    numx2 = 40
    numx = int((numx2 - numx1) / dh + 1)
    numy1 = -40
    numy2 = 40
    numy = int((numy2 - numy1) / dh + 1)
    numz1 = -130
    numz2 = -50
    numz = int((numz2 - numz1) / dh + 1)
    
    # Generate grid point coordinates inside target
    x0, y0, z0 = np.meshgrid(np.linspace(numx1, numx2, numx),
                             np.linspace(numy1, numy2, numy),
                             np.linspace(numz1, numz2, numz))
    xyz0 = np.c_[mkvc(x0), mkvc(y0), mkvc(z0)]  # Target coordinate matrix

    # Refinement region setup around target (boundary transition zone)
    pointx = [numx1 - 2 * dh, numx2 + 2 * dh]
    pointy = [numy1 - 2 * dh, numy2 + 2 * dh]
    pointz = [numz1 - 2 * dh, numz2 + 2 * dh]
    octree_levels = []
    for i in range(1):
        if i == 0:
            octree_levels.append(1)  # Highest refinement level
        else:
            octree_levels.append(0)
            # Expand refinement region (this branch is not executed)
            pointx = [pointx[0] - 2 ** (i + 2) * dh, pointx[1] + 2 ** (i + 2) * dh]
            pointy = [pointy[0] - 2 ** (i + 2) * dh, pointy[1] + 2 ** (i + 2) * dh]
            pointz = [pointz[0] - 2 ** (i + 2) * dh, pointz[1] + 2 ** (i + 2) * dh]
        
        # Create refinement box corner points
        x_obj, y_obj, z_obj = np.meshgrid(
            [pointx[0], pointx[1]],
            [pointy[0], pointy[1]],
            [pointz[0], pointz[1]]
        )
        xyz_obj = np.c_[mkvc(x_obj), mkvc(y_obj), mkvc(z_obj)]
        mesh = refine_tree_xyz(
            mesh, xyz_obj,
            octree_levels=list(reversed(octree_levels)),
            method='box',
            finalize=False,
        )
# ==========================================================================================================
mesh.finalize()  # Finalize mesh construction, lock topology
# ==========================================================================================================
# Output mesh information: For mesh quality verification
# ==========================================================================================================
print(mesh)                               # Mesh summary
print(mesh.n_nodes)                       # Number of physical nodes
print(mesh.n_total_nodes)                 # Total nodes (including hanging nodes)
print('Hanging faces: %d' % (mesh.n_hanging_faces))  # Number of hanging faces (octree feature)
print('Total faces: %d' % (mesh.n_total_faces))      # Total faces
# ==========================================================================================================
# Conductivity model construction
# ==========================================================================================================
# Conductivity parameter configuration (unit: S/m)
air_conductivity = 1e-8      # Air conductivity
bkg_conductivity = 1e-2      # Background medium conductivity
block_conductivity = 1e-0    # Target conductivity

# Terrain processing: Determine active cells (underground region)
ind_active = active_from_xyz(mesh, topo_xyz)
model_map = maps.InjectActiveCells(mesh, ind_active, air_conductivity)  # Model mapper
model = bkg_conductivity * np.ones(ind_active.sum())  # Initialize background model

# Anomaly identification: Determine target cells by coordinate range
ind = (
    (mesh.gridCC[ind_active, 0] <= np.amax(x0))   # x coordinate range
    & (mesh.gridCC[ind_active, 0] >= np.amin(x0))
    & (mesh.gridCC[ind_active, 1] <= np.amax(y0))   # y coordinate range
    & (mesh.gridCC[ind_active, 1] >= np.amin(y0))
    & (mesh.gridCC[ind_active, 2] <= np.amax(z0))   # z coordinate range
    & (mesh.gridCC[ind_active, 2] >= np.amin(z0))
)
print(ind)                          # Output target cell indices
model[ind] = block_conductivity     # Set target conductivity
etime=time.time()
print('Mesh generation completed, time elapsed:',etime-stime,'seconds, current time:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# ==========================================================================================================
print('Loading transmitter settings......')
# Fit square loop transmitter
source_list = []
L=100        # Half side length of square loop (loop size: 200m×200m)
source_list.append(
    tdem.sources.LineCurrent(
        location=np.array([[-L, L, 0],    # Loop vertex coordinates
                           [L, L, 0],
                           [L, -L, 0],
                           [-L, -L, 0],
                           [-L, L, 0]]),  # Closed loop
        current=10  # Transmitter current (Amperes)
    ))
survey = tdem.Survey(source_list)  # Build survey object
#END
print('Transmitter settings loaded')
# ==========================================================================================================
print('Time series preparation started, current time:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# offtime = 1e-2                                  # Offtime duration (commented alternative)
nt_off = 101  # Number of off-time sampling points
t_off = np.logspace(-5, -2, nt_off)  # Logarithmically distributed time series (10μs - 10ms)
dt_off = t_off[1:] - t_off[:-1]      # Time intervals
dt_off = np.r_[1e-5, dt_off]         # Initial time step
t_full = np.r_[0, t_off]             # Complete time series (including t=0)
np.savetxt(r'data/model1/time.dat', t_off)  # Save time series
print('Time series preparation completed')
# ==========================================================================================================
print('Computation started, current time:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# Call subroutine to start formal computation
print('Loading simulation......')
sim = SimulationTEM(mesh, survey, model_map, model)
print('Simulation loaded')
print('Loading receivers......')
sim.rec_loc = rec_loc
print('Receivers loaded')
# ==========================================================================================================
# SAI-SD Algorithm: Fixed subspace dimension
# ==========================================================================================================
print('Calculating subspace dimension')
tolerance=1e-3       # Convergence tolerance
gamma=7e-5           # Shift parameter (shifted inverse Arnoldi)
m=200                 # Fixed subspace dimension
print('m:%d'% m)
# ==========================================================================================================
# SAI-SD Algorithm: Model order reduction based on shifted inverse Krylov subspace
# ==========================================================================================================
s1time=time.time()
print('Starting total field calculation')
b = sim.solve_mag_SAI_SD_off(t_off, tolerance,  m,gamma)  # Core solution

s2time=time.time()
print('Starting bz calculation')
bz_SD = sim.rec_proj * b  # Project total field to receivers to get magnetic field z-component

s3time=time.time()
filename = r'data/model1/80lft-kuosan'
suffix1 = 'bz.dat'         # Magnetic field z-component filename
suffix2 = 'dbz-dt.dat'     # Time derivative of magnetic field filename
name1=filename+suffix1
name2=filename+suffix2
np.savetxt(name1, bz_SD.T)  # Save magnetic field data
bz_SD = np.loadtxt(name1)   # Reload

# ==========================================================================================================
# Numerical differentiation for dBz/dt: Linear interpolation refinement + central difference
# ==========================================================================================================
print('Starting dbz/dt calculation')
# Non-zero multi-receiver interpolation scheme
dt_diff = 1e-7                                   # Differential time step
t_diff = np.arange(t_off[0], t_off[-1], dt_diff)  # Extended refined time series
dbdt_SD_off = np.zeros(np.shape(bz_SD))

# Calculate time derivative for each receiver
for i in range(number_cd):
    bz_SD0=bz_SD[:,i]
    b_recdiff = np.interp(t_diff, t_off, bz_SD0)  # Linear interpolation to refine magnetic field curve
    deltx2 = t_diff[2:] - t_diff[:-2]             # Central difference interval
    delty2 = b_recdiff[2:] - b_recdiff[:-2]
    # deltx = np.r_[deltx2, (t_diff[-1] - t_diff[-2]) * 1.4]  # Alternative scheme
    deltx = np.r_[deltx2, (t_diff[-1] - t_diff[-2]) ]
    delty = np.r_[delty2, b_recdiff[-1] - b_recdiff[-2]]
    dbdt_diff = delty / deltx                     # Central difference for derivative
    dbdt_SD_off0 = np.interp(t_off, t_diff[1:], dbdt_diff)  # Interpolate back to original time points
    dbdt_SD_off[:,i]=dbdt_SD_off0

etime = time.time()
np.savetxt(name2, dbdt_SD_off)  # Save time derivative data

# Output computation time statistics
print('Total magnetic field calculation time:',s2time-s1time,'seconds')
print('Magnetic field z-component calculation time:',s3time-s1time,'seconds')
print('dbz-dt calculation time and total computation time:',(etime - s1time),'seconds')
print('Total program runtime:',etime-stime,'seconds')
print('All computation results saved, please check!')