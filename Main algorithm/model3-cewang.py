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
dh = 1.25                 # Minimum cell width (meters) - finer mesh 1.25m
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
        mesh, topo_xyz, method="surface",octree_levels=[0,0,0,0,0,0,0,0,0,1], finalize=False  # Deeper refinement level
)
# ==========================================================================================================
print('Setting up receiver points......')
# Multiple receivers: 2D receiver grid layout
num_x=11                # Number of receivers along x-axis (11 receivers)
num_y=11                # Number of receivers along y-axis (11 receivers)
xrx, yrx = np.meshgrid(np.linspace(-50, 50, num_x),
                     np.linspace(-50, 50, num_y))
zrx = np.zeros(np.shape(xrx))                  # Receivers located at surface (z=0)
number_cd=num_x*num_y                         # Total number of receivers
rec_loc = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]  # Convert to Nx3 receiver matrix
print('Receiver setup completed')
# ==========================================================================================================
# Define refinement regions: Multi-level refinement strategy
# ==========================================================================================================

# ============================================================
# Transmitter area and near-surface refinement: Gradually expanding refinement range
# Using finer refinement strategy
# ============================================================
print('Starting region refinement......')
octree_levels = [1,0]  # Initial refinement level setting
for i in range(3):
    octree_levels.append(0)
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
# [Model3 Feature] External DAT file controlled refinement region loading
# Using predefined DAT files to control mesh refinement for complex anomalies
# ============================================================

# Load main anomaly surface control points file (for radial refinement)
filename02 = r'data/model3/'
filename01 = 'octree-dc4-s-q'
suffix02 = '.DAT'
name0=filename02+filename01+suffix02
xyz0=np.loadtxt(name0)  # Load anomaly surface coordinates defined by external DAT file
octree_levels=[1]
mesh = refine_tree_xyz(
    mesh, xyz0,
    octree_levels=list(reversed(octree_levels)),
    method='radial',  # Using radial refinement method
    finalize=False,
    )

# Load left anomaly 1 surface control points file
filename01 = 'octree-dc4-s-lft1'
suffix02 = '.DAT'
name0=filename02+filename01+suffix02
xyz0=np.loadtxt(name0)
octree_levels=[1,0,0]  # Multi-level refinement
mesh = refine_tree_xyz(
    mesh, xyz0,
    octree_levels=list(reversed(octree_levels)),
    method='radial',
    # method='surface',
    finalize=False,
    )

# Load left anomaly 2 surface control points file
filename01 = 'octree-dc4-s-lft2'
suffix02 = '.DAT'
name0=filename02+filename01+suffix02
xyz0=np.loadtxt(name0)
octree_levels=[1,0,0]
mesh = refine_tree_xyz(
    mesh, xyz0,
    octree_levels=list(reversed(octree_levels)),
    method='radial',
    finalize=False,
    )

# Load main anomaly inner control points file (for box refinement)
me01 = 'octree-dc4-s-q-1'
suffix02 = '.DAT'
name0=filename02+me01+suffix02
xyz0=np.loadtxt(name0)
octree_levels=[1,0,0,0]  # Higher level inner refinement
mesh = refine_tree_xyz(
    mesh, xyz0,
    octree_levels=list(reversed(octree_levels)),
    method='box',  # Using box refinement method
    finalize=False,
    )

# Load left anomaly 1 inner control points file
filename01 = 'octree-dc4-s-lft1-1'
suffix02 = '.DAT'
name0=filename02+filename01+suffix02
xyz0=np.loadtxt(name0)
octree_levels=[1,0,0,0]
mesh = refine_tree_xyz(
    mesh, xyz0,
    octree_levels=list(reversed(octree_levels)),
    method='box',
    finalize=False,
    )

# Load left anomaly 2 inner control points file
filename01 = 'octree-dc4-s-lft2-1'
suffix02 = '.DAT'
name0=filename02+filename01+suffix02
xyz0=np.loadtxt(name0)
octree_levels=[1,0,0,0]
mesh = refine_tree_xyz(
    mesh, xyz0,
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
air_conductivity = 1e-8       # Air conductivity
bkg_conductivity = 1e-2      # Background medium conductivity (0.01 S/m)
block_conductivity = 1e-3     # Low conductivity anomaly (0.001 S/m, good conductor)
block_conductivity1=1e-0      # High conductivity anomaly (1 S/m, high conductor)

# Terrain processing: Determine active cells (underground region)
ind_active = active_from_xyz(mesh, topo_xyz)
model_map = maps.InjectActiveCells(mesh, ind_active, air_conductivity)  # Model mapper
model = bkg_conductivity * np.ones(ind_active.sum())  # Initialize background model

# ============================================================
# [Model3 Feature] Irregular complex anomaly definition
# Using DAT file defined coordinates and direct coordinate range combination to define anomalies
# ============================================================

# Load irregular anomaly 2 coordinate file and set conductivity
filename01 = 'octree-dc4-s-lft2-1'
filename02 = 'E:/陈肯/陈肯/陈肯/新模型/新模型/'  # External path
suffix02 = '.DAT'
name0=filename02+filename01+suffix02
xyz0=np.loadtxt(name0)
ind=mesh.closest_points_index(xyz0)  # Find closest grid points
ind=np.unique(ind)                   # Remove duplicates
model[ind] = block_conductivity      # Set low conductivity anomaly

# Load irregular anomaly 1 coordinate file and set conductivity
filename01 = 'octree-dc4-s-lft1-1'
filename02 = 'E:/陈肯/陈肯/陈肯/新模型/新模型/'
suffix02 = '.DAT'
name0=filename02+filename01+suffix02
xyz0=np.loadtxt(name0)
ind=mesh.closest_points_index(xyz0)
ind=np.unique(ind)
model[ind] = block_conductivity

# Load main anomaly coordinate file and set high conductivity
filename01 = 'octree-dc4-s-q-1'
filename02 = 'E:/陈肯/陈肯/陈肯/新模型/新模型/'
suffix02 = '.DAT'
name0=filename02+filename01+suffix02
xyz0=np.loadtxt(name0)
ind=mesh.closest_points_index(xyz0)
ind=np.unique(ind)
model[ind] = block_conductivity1  # High conductivity

# ============================================================
# [Model3 Feature] Background medium correction regions
# Clear some anomaly covered regions and restore background conductivity
# ============================================================

# Correction region 1: Background region below left high conductivity anomaly
ind1 = (
    (mesh.gridCC[ind_active, 0] <-430)     # x-direction range
    & (mesh.gridCC[ind_active, 0] >= -460)  # x-direction range
    & (mesh.gridCC[ind_active, 1] <= 110)   # y-direction range
    & (mesh.gridCC[ind_active, 1] >= -110)  # y-direction range
    & (mesh.gridCC[ind_active, 2] <= -90)   # z-direction upper limit
    & (mesh.gridCC[ind_active, 2] >= -210)  # z-direction lower limit
)
model[ind1] = bkg_conductivity  # Restore background conductivity

# Correction region 2: Middle region background layer
ind1 = (
        (mesh.gridCC[ind_active, 0] < 100)   # x-direction range
        & (mesh.gridCC[ind_active, 0] >= -460)
        & (mesh.gridCC[ind_active, 1] <= 110)
        & (mesh.gridCC[ind_active, 1] >= -110)
        & (mesh.gridCC[ind_active, 2] < -200)   # z-direction specific layer
        & (mesh.gridCC[ind_active, 2] >= -220)
)
model[ind1] = bkg_conductivity  # Restore background conductivity

# Correction region 3: Left region background layer
ind1 = (
        (mesh.gridCC[ind_active, 0] < -200)  # x-direction range
        & (mesh.gridCC[ind_active, 0] >= -460)
        & (mesh.gridCC[ind_active, 1] <= 110)
        & (mesh.gridCC[ind_active, 1] >= -110)
        & (mesh.gridCC[ind_active, 2] <= -50)   # z-direction shallow layer
        & (mesh.gridCC[ind_active, 2] > -100)
)
model[ind1] = bkg_conductivity  # Restore background conductivity

# Correction region 4: Right region background layer
ind1 = (
        (mesh.gridCC[ind_active, 0] < 500)   # x-direction range
        & (mesh.gridCC[ind_active, 0] > 27)  # Exclude main anomaly
        & (mesh.gridCC[ind_active, 1] <= 110)
        & (mesh.gridCC[ind_active, 1] >= -110)
        & (mesh.gridCC[ind_active, 2] < -150)  # z-direction specific layer
        & (mesh.gridCC[ind_active, 2] >= -155)
)
model[ind1] = bkg_conductivity  # Restore background conductivity

etime=time.time()
print('Mesh generation completed, time elapsed:',etime-stime,'seconds, current time:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# ==========================================================================================================
# Mesh visualization (optional)
# ==========================================================================================================
plot_mesh0 = 1  # Visualization enabled by default
if plot_mesh0:
    # Define Chinese font (SimSun) and English font (Times New Roman)
    chinese_font = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=12)  # Chinese SimSun
    english_font = {'family': 'Times New Roman', 'size': 12}  # English Times New Roman
    print('Mesh visualization')
    mpl.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(19.2, 10.8))
    
    # Build complete model (including air layer) for plotting
    model0 = air_conductivity * np.ones(model_map.indInactive.sum())
    model1 = model
    model2 = np.hstack([model1,model0])

    log_model = np.log10(model2)  # Logarithmic transformation for contrast enhancement
    ind_active1 = ind_active
    ind_active1[:] = True
    plotting_map = maps.InjectActiveCells(mesh, ind_active1, np.nan)
    
    ax1 = fig.add_axes([0.16, 0.18, 0.55, 0.75])
    # Set axis tick labels to Times New Roman
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(28)
    
    pcolor_opts = {'cmap': 'jet'}  # Jet color map
    
    # Set different axis limits
    x_min, x_max = -200, 200  # x-axis range
    y_min, y_max = -250, 150  # y-axis range
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    # Axis label font configuration
    xlabel_font = {'family': 'Times New Roman', 'size': 28}
    ylabel_font = {'family': 'Times New Roman', 'size': 28}
    colorbar_title_font = {'family': 'Times New Roman','size': 28}
    
    # Plot conductivity model slice at Y=0
    mesh_plot=mesh.plotSlice(
        plotting_map * log_model,
        normal="Y",
        ax=ax1,
        ind=int(mesh.h[0].size / 2),
        grid=True,
        clim=(np.min(log_model), np.max(log_model)),
        pcolor_opts=pcolor_opts
    )
    
    ax1.set_title("Conductivity Model at  Y = 0 m", fontdict=english_font, y=1.1)
    
    # Add colorbar
    for collection in ax1.collections:
        if hasattr(collection, 'get_array'):
            cax = fig.add_axes([0.72, 0.18, 0.03, 0.75])
            cbar = plt.colorbar(collection, cax=cax)  # Create colorbar
            break
    
    # Set colorbar tick labels to Times New Roman
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(28)
    
    # Calculate new color range (extend by one order of magnitude)
    vmin = np.min(log_model) - 1  # Decrease minimum
    vmax = np.max(log_model) + 1  # Increase maximum

    # Set color range for plot objects
    for collection in ax1.collections:
        if hasattr(collection, 'set_clim'):
            collection.set_clim(vmin, vmax)  # Set color range
            break

    # Custom tick label format as scientific notation
    cbar.formatter = FuncFormatter(lambda x, pos: f"$10^{{{x:.1f}}}$")
    cbar.update_ticks()
    
    ax1.set_xlabel('Y(m)', fontdict=xlabel_font)
    ax1.set_ylabel('Z(m)', fontdict=ylabel_font)
    cbar.set_label('Conductivity (S/m)', fontdict=colorbar_title_font)  # Set colorbar label
    
    # Save image
    plt.savefig(r'data/model3/dc4_mesh.png', dpi=600, pad_inches=0)  # Save path
    plt.show()
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
print('Transmitter settings loaded')
# ==========================================================================================================
print('Time series preparation started, current time:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# offtime = 1e-2                                  # Offtime duration (commented alternative)
nt_off = 101  # Number of off-time sampling points
t_off = np.logspace(-5, -2, nt_off)  # Logarithmically distributed time series (10μs - 10ms)
dt_off = t_off[1:] - t_off[:-1]      # Time intervals
dt_off = np.r_[1e-5, dt_off]         # Initial time step
t_full = np.r_[0, t_off]             # Complete time series (including t=0)
np.savetxt(r'data/model3/time.dat', t_off)  # Save to model3 directory
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
gamma=10e-5           # Shift parameter (10e-5)
m=sim.opt_SD_m(t_off, tolerance,  gamma)  # Solve for optimal subspace dimension
m=m+1                # Dimension compensation
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
filename = r'data/model3/dc4-kuosan'  # Output filename
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
    deltx = np.r_[deltx2, (t_diff[-1] - t_diff[-2]) ]  # Boundary handling
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