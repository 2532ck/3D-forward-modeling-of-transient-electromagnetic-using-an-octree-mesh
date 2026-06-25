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
# Multiple receivers: Single line along y-axis
num_x=1                 # Number of receivers along x-axis (single point)
num_y=101               # Number of receivers along y-axis (101 receivers)
xrx, yrx = np.meshgrid(np.linspace(-0, 0, num_x),    # Single line along y-axis: x=0
                     np.linspace(-50, 50, num_y))    # y∈[-50,50]m
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
# Layered anomaly target region refinement (Second refinement layer)
# ============================================================
print('Starting layered anomaly region refinement......')
octree_levels = []
for i in range(1):
    octree_levels.append(1) if i == 0 else octree_levels.append(0)
    box_range =-30               # Lateral range ±30m
    air_range1 =-190             # Lower boundary depth 190m
    air_range=-50                # Upper boundary depth 50m
    x_obj, y_obj, z_obj = np.meshgrid(
        [-box_range, box_range],
        [-box_range, box_range],
        [air_range1, air_range]  # Target layered anomaly depth range: -190m to -50m
    )
    xyz_obj = np.c_[mkvc(x_obj), mkvc(y_obj), mkvc(z_obj)]
    mesh = refine_tree_xyz(
        mesh, xyz_obj,
        octree_levels=list(reversed(octree_levels)),
        method='box',
        finalize=False,
        )

# ============================================================
# Layer boundary refinement: Refine layered structure interfaces
# ============================================================
print('Starting layer boundary refinement......')
# Layered anomaly interface position parameters
numxc1=-dom_width/2             # x-direction boundary start
numxc2=dom_width/2             # x-direction boundary end
numc=151                        # Number of interface grid points
numyc1=-dom_width/2            # y-direction boundary start
numyc2=dom_width/2             # y-direction boundary end
cz1=-80                         # Interface 1 depth (upper interface)
cz2=-160                        # Interface 2 depth (lower interface)

# Generate two horizontal layer interface grids
cengx,cengy = np.meshgrid(np.linspace(numxc1, numxc2, numc),
                            np.linspace(numyc1, numyc2, numc))
cengz1=np.full((numc,numc),cz1)  # Upper interface depth array
cengz2=np.full((numc,numc),cz2)  # Lower interface depth array
topo_xyz1 = np.c_[mkvc(cengx), mkvc(cengy), mkvc(cengz1)]
topo_xyz2 = np.c_[mkvc(cengx), mkvc(cengy), mkvc(cengz2)]

# Apply local refinement for both layer interfaces (Gaussian refinement strategy)
mesh = refine_tree_xyz(
        mesh, topo_xyz1, method="surface",octree_levels=[0,0,0,0,1,0,0,0,0,0,0,0], finalize=False
)
mesh = refine_tree_xyz(
        mesh, topo_xyz2, method="surface",octree_levels=[0,0,0,0,1,0,0,0,0,0,0,0], finalize=False
)
print('Layer boundary refinement completed')
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
bkg_conductivity = 1/3*1e-2   # Background medium conductivity (approx. 0.33×10⁻² S/m)
block_conductivity = 1e-2     # Layered anomaly conductivity (0.01 S/m)
block_conductivity1=1e-0      # Prismatic target conductivity (1 S/m, high conductor)

# Terrain processing: Determine active cells (underground region)
ind_active = active_from_xyz(mesh, topo_xyz)
model_map = maps.InjectActiveCells(mesh, ind_active, air_conductivity)  # Model mapper
model = bkg_conductivity * np.ones(ind_active.sum())  # Initialize background model
np.savetxt(r'data/model2/model_bf.dat',model)         # Save background model

# ============================================================
# Layered anomaly definition: Horizontal conductive layer (80m-160m depth)
# ============================================================
# Layered anomaly
ind = (
     (mesh.gridCC[ind_active, 2] <= cz1)   # z-coordinate less than upper interface depth
    & (mesh.gridCC[ind_active, 2] >= cz2)  # z-coordinate greater than lower interface depth
)
model[ind] = block_conductivity            # Set layered anomaly conductivity

# ============================================================
# Stepped prismatic target definition: Stepped prismatic anomaly (6 decreasing blocks)
# ============================================================
# Stepped target
py1=30                                    # Prism y-direction upper limit (initial)
py2=20                                    # Prism y-direction lower limit (initial)
pz1=-50                                   # Prism z-direction upper limit (initial)
pz2=-140                                  # Prism z-direction lower limit (initial)

# Loop to create 6 decreasing prisms forming inclined structure
for i in range(6):
    ind1 = (
            (mesh.gridCC[ind_active, 0] <= 30)                # x-direction range ±30m
            & (mesh.gridCC[ind_active, 0] >= -30)
            & (mesh.gridCC[ind_active, 1] <= py1 - i * 10)   # y-direction decreases with i
            & (mesh.gridCC[ind_active, 1] >= py2 - i * 10)    # y-direction decreases with i
            & (mesh.gridCC[ind_active, 2] <= pz1 - i * 10)   # z-direction decreases with i (upward shift)
            & (mesh.gridCC[ind_active, 2] >= pz2 - i * 10)   # z-direction decreases with i
    )
    print(ind1)                        # Output each stepped target cell indices
    model[ind1] = block_conductivity1   # Set stepped target conductivity (high conductor)
np.savetxt(r'data/model2/model_af.dat',model)  # Save model with anomalies
etime=time.time()
print('Mesh generation completed, time elapsed:',etime-stime,'seconds, current time:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# ==========================================================================================================
# Mesh visualization (optional)
# ==========================================================================================================
plot_mesh0 = 0  # Visualization switch (0=off, 1=on)
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
    
    # Set axis limits (focus on target region)
    x_min, x_max = -400, 400  # x-axis range
    y_min, y_max = -400, 400  # y-axis range
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    # Axis label font configuration
    xlabel_font = {'family': 'Times New Roman', 'size': 28}
    ylabel_font = {'family': 'Times New Roman', 'size': 28}
    colorbar_title_font = {'family': 'Times New Roman','size': 28}
    
    # Plot conductivity model slice at X=0
    mesh_plot=mesh.plotSlice(
        plotting_map * log_model,
        normal="X",
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
    plt.savefig(r'data/model2/3F6CZ_mesh.png', dpi=600, pad_inches=0)
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
np.savetxt(r'data/model2/time.dat', t_off)  # Save time series
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
# SAI-SD Algorithm: Adaptive optimization of subspace dimension
# ==========================================================================================================
print('Calculating subspace dimension')
tolerance=1e-3       # Convergence tolerance
gamma=10e-5          # Shift parameter (shifted inverse Arnoldi)
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
filename = r'data/model2/3F6CZ-cexian'
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
