import asyncio
import numpy as np
import plotly.graph_objects as go
import cloudvolume
from cloudvolume import Bbox
import navis
import os
import sys
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from scipy.spatial import cKDTree
import caveclient
from caveclient import CAVEclient
from datetime import datetime, timezone
import math
import octarine as oc
from PIL import Image
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# This looks for .env in the project root directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # If no .env file, try to load from current directory or parent directories
    load_dotenv()


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout (e.g., CloudVolume warnings)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class ConnectomeVisualizer:
    """
    A class for visualizing connectomics neurons and EM data.

    Supports multiple species:
    - mouse (MICrONS dataset) - Public access, no authentication required
    - fly (FlyWire dataset) - Public access, no authentication required
    - human (H01 dataset) - Requires authentication (see README)
    - zebrafish (Fish1 dataset) - Requires authentication (see README)

    The CAVEclient automatically retrieves the correct EM and segmentation paths via its
    InfoService API. Authentication tokens are passed to both CAVEclient and CloudVolume
    to enable access to authenticated graphene:// backend endpoints.

    Authentication is managed via a CAVECLIENT_TOKEN in a .env file.
    See README for complete authentication setup instructions.
    """
    
    # Default paths for data sources
    # Mouse (MICrONS) - Public access, no authentication required
    MICRONS_EM_PATH = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
    MICRONS_SEG_PATH = "graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public"

    # FlyWire (Drosophila) - Public access, no authentication required
    FLYWIRE_EM_PATH = "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14"
    FLYWIRE_SEG_PATH =  "graphene://https://prod.flywire-daf.com/segmentation/1.0/flywire_public" 
    
    # H01 (Human Cortex) - Requires authentication via https://h01-release.storage.googleapis.com/proofreading.html
    H01_EM_PATH = "precomputed://gs://h01-release/data/20210601/4nm_raw"
    H01_SEG_PATH = "graphene://https://local.brain-wire-test.org/segmentation/table/h01_full0_v2"
    # # TODO: These might be the 104 proofread neurons for H01 (https://h01-release.storage.googleapis.com/data.html)
    # H01_EM_PATH = "precomputed://gs://h01-release/data/20210601/proofread_104"
    # H01_SEG_PATH = "graphene://https://local.brain-wire-test.org/segmentation/table/h01_full0_v2"

    # Fish1 (Zebrafish) - Requires authentication via https://fish1-release.storage.googleapis.com/tutorials.html
    FISH1_EM_PATH = "precomputed://gs://fish1-public/clahe_231218"
    FISH1_SEG_PATH = "graphene://https://pcgv3local.brain-wire-test.org/segmentation/table/fish1_v250915"

    # Default colors for neurons
    NEURON_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#FF4500",  # Orange-red
        "#32CD32",  # Lime green
        "#FFD700",  # Gold
        "#9370DB",  # Medium purple
        "#FF69B4",  # Hot pink
        "#00CED1",  # Dark turquoise
        "#FF8C00",  # Dark orange
        "#8A2BE2",  # Blue violet
        "#20B2AA",  # Light sea green
        "#1E90FF",  # Dodger blue
    ]

    datastacks = ['minnie65_public', 'flywire_fafb_public', 'h01_c3_flat', 'fish1_full']

    data_parameters = {
        "mouse": {
            "em_path": MICRONS_EM_PATH,
            "seg_path": MICRONS_SEG_PATH,
            "datastack_name": "minnie65_public",
            "em_mip": 2,
            "seg_mip": 2
        },
        "fly": {
            "em_path": FLYWIRE_EM_PATH,
            "seg_path": FLYWIRE_SEG_PATH,
            "datastack_name": "flywire_fafb_public",
            "em_mip": 2,
            "seg_mip": 0
        },
        "human": {
            "em_path": H01_EM_PATH,
            "seg_path": H01_SEG_PATH,
            "datastack_name": "h01_c3_flat",  # Requires authentication - see README
            "em_mip": 2,  # Adjusted for H01 scale
            "seg_mip": 0
        },
        "zebrafish": {
            "em_path": FISH1_EM_PATH,
            "seg_path": FISH1_SEG_PATH,
            "datastack_name": "fish1_full",  # Requires authentication - see README
            "em_mip": 2,
            "seg_mip": 0
        }
    }
    def __init__(self,
                 species: str = "fly",
                 output_dir: str = "./output",
                 dataset: str = "public",
                 timestamp: int = None,
                 verbose: bool = True
                 ):
        """
        Initialize the ConnectomeVisualizer.

        Args:
            em_path: Path to the EM data
            seg_path: Path to the segmentation data
            output_dir: Directory to save output files
            dataset: FlyWire dataset to use ("sandbox", "public", "production")
            verbose: Whether to print status messages (default: True)
        """
        self.species = species
        self.verbose = verbose
        # Set up data paths
        self.em_path = self.data_parameters[species]["em_path"]
        self.seg_path = self.data_parameters[species]["seg_path"]
        self.datastack_name = self.data_parameters[species]["datastack_name"]
        self.timestamp = timestamp

        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data containers
        self.neurons = []
        self.neuron_ids = []
        self.neuron_color_map = {}

        self.color_idx = 0

        # Initialize data volumes
        self.cv_em = None
        self.cv_seg = None
        self.vol_em = None
        self.vol_seg = None
        self.vol_supervoxels = None

        self.em_mip = self.data_parameters[species]["em_mip"]
        self.seg_mip = self.data_parameters[species]["seg_mip"]
        self.datastack_name = self.data_parameters[species]["datastack_name"]

        # Get CAVECLIENT_TOKEN for authenticated datasets (human, zebrafish)
        self.auth_token = None
        if species in ["human", "zebrafish"]:
            self.auth_token = os.getenv("CAVECLIENT_TOKEN")
            if not self.auth_token:
                raise ValueError(
                    f"\n{'='*70}\n"
                    f"ERROR: CAVECLIENT_TOKEN not found!\n"
                    f"{'='*70}\n"
                    f"The {species} dataset requires authentication.\n\n"
                    f"To fix this:\n"
                    f"1. Make sure you have a .env file in the project root with:\n"
                    f"   CAVECLIENT_TOKEN=your_token_here\n\n"
                    f"2. If you don't have a token yet:\n"
                    f"   a) Request access: https://forms.gle/tpbndoL1J6xB47KQ9\n"
                    f"   b) After approval, get your token from:\n"
                    f"      https://global.brain-wire-test.org/auth/api/v1/create_token\n"
                    f"   c) Run: python scripts/setup_cave_auth.py\n\n"
                    f"3. See .env.example for template\n"
                    f"{'='*70}\n"
                )

        # Initialize CAVEclient
        if self.datastack_name is not None:
            if species in ["human", "zebrafish"]:
                server_address = "https://global.brain-wire-test.org/"
                self.client = CAVEclient(self.datastack_name, server_address=server_address, auth_token=self.auth_token)
            else:
                self.client = CAVEclient(self.datastack_name)

            # Get paths from CAVEclient InfoService (overrides hardcoded defaults)
            try:
                client_em_path = self.client.info.image_source()
                client_seg_path = self.client.info.segmentation_source()
                
                if client_em_path.startswith("precomputed://") or client_em_path.startswith("graphene://"):
                    self.em_path = client_em_path
                if client_seg_path.startswith("precomputed://") or client_seg_path.startswith("graphene://"):
                    self.seg_path = client_seg_path

                if self.verbose:
                    print(f"Retrieved paths from CAVEclient InfoService vs Hardcoded paths:")
                    print(f"  EM\n\tretrieved: {client_em_path}\n\thardcoded: {self.em_path}")
                    print(f"  Segmentation\n\tretrieved: {client_seg_path}\n\thardcoded: {self.seg_path}")

            except Exception as path_error:
                if self.verbose:
                    print(f"Note: Could not retrieve paths from CAVEclient InfoService, using hardcoded paths: {path_error}")

            if self.verbose:
                print(f"Successfully initialized CAVEclient for {species} (datastack: {self.datastack_name})")
        else:
            self.client = None
        
        # Initialize coordinates
        self.x = None
        self.y = None
        self.z = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.min_z = None
        self.max_z = None
        
        # Initialize grid coordinates
        self.x_2d = None
        self.y_2d = None
        self.X_3d = None
        self.Y_3d = None
        self.Z_3d = None
        
        # Initialize figures
        self.neuron_fig = None
        self.em_seg_fig = None
        self.interactive_fig = None
        
        # Initialize segmentation data
        self.dataset = dataset
        self.segmentation_data = None
        self.nonzero_points = None
        
        # Initialize root_ids_grid cache as a dictionary mapping locations to grids
        self.root_ids_grids = {}  # Dictionary to store multiple root_ids_grids
        self.current_location = None  # Current location (x, y, z, window_size, window_z)
        
        # Force plotly as the backend for navis
        navis.config.default_backend = 'octarine'
        
        # Connect to data sources
        self._connect_to_data_sources()
    
    def _connect_to_data_sources(self):
        """Connect to the EM and segmentation data sources."""
        try:
            # Connect to EM data
            self.cv_em = cloudvolume.CloudVolume(self.em_path, use_https=True, mip=self.em_mip, timestamp=self.timestamp)
            self.em_resolution = self.cv_em.resolution

            # Connect to segmentation data
            # Pass auth token to CloudVolume if available (for authenticated graphene:// endpoints)
            if self.auth_token:
                self.cv_seg = cloudvolume.CloudVolume(
                    self.seg_path,
                    use_https=True,
                    fill_missing=True,
                    mip=self.seg_mip,
                    timestamp=self.timestamp,
                    secrets={'token': self.auth_token}
                )
            else:
                self.cv_seg = cloudvolume.CloudVolume(
                    self.seg_path,
                    use_https=True,
                    fill_missing=True,
                    mip=self.seg_mip,
                    timestamp=self.timestamp
                )

            self.seg_resolution = self.cv_seg.resolution

            if self.verbose:
                print("Successfully connected to data sources.")
        except Exception as e:
            if self.verbose:
                print(f"Error connecting to data sources: {e}")

    async def _fetch_neuron_mesh(self, neuron_id: int, timeout: Optional[float] = None) -> Tuple[int, Optional[Any]]:
        """Helper function to fetch a single neuron mesh asynchronously."""
        
        # Create a future to manage the execution
        loop = asyncio.get_event_loop()
        
        def fetch_mesh_quietly():
            with suppress_stdout():
                return self.cv_seg.mesh.get(neuron_id)[neuron_id]
        
        try:
            if timeout is not None:
                # Run the blocking operation in a thread pool with timeout
                mesh_future = loop.run_in_executor(None, fetch_mesh_quietly)
                mesh = await asyncio.wait_for(mesh_future, timeout)
            else:
                # No timeout, but still run in executor to avoid blocking
                mesh = await loop.run_in_executor(None, fetch_mesh_quietly)
                
            if self.verbose:
                print(f"Successfully fetched mesh for neuron {neuron_id}")
            return neuron_id, mesh

        except asyncio.TimeoutError:
            if self.verbose:
                print(f"Timeout after {timeout}s while fetching neuron {neuron_id}")
            # Make sure we don't leave the task hanging
            return neuron_id, None
        except Exception as e:
            if self.verbose:
                print(f"Error loading neuron {neuron_id}: {e}")
            return neuron_id, None

    def load_neurons(self, neuron_ids: List[int]):
        """
        Load neuron meshes for the specified neuron IDs.
        
        Args:
            neuron_ids: List of neuron IDs to load
        """
        self.neurons = []
        self.neuron_ids = neuron_ids
        self.neuron_color_map = {}
        
        for i, neuron_id in enumerate(neuron_ids):
            color = self.NEURON_COLORS[self.color_idx % len(self.NEURON_COLORS)]
            self.color_idx += 1

            self.neuron_color_map[neuron_id] = color
            try:

                # neuron = flywire.get_mesh_neuron(neuron_id, dataset=self.dataset)
                with suppress_stdout():
                    neuron = self.cv_seg.mesh.get(neuron_id)[neuron_id]

                self.neurons.append(neuron)
                if self.verbose:
                    print(f"Loaded neuron {neuron_id}")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading neuron {neuron_id}: {e}")
        
        return self.neurons
    
    def get_neuron_skeletons(self, neuron_ids: List[int]):
        """
        Load neuron skeletons for the specified neuron IDs.
        """
        if self.client is None:
            auth_msg = ""
            if self.species == "human":
                auth_msg = " Please authenticate via https://h01-release.storage.googleapis.com/proofreading.html"
            elif self.species == "zebrafish":
                auth_msg = " Please authenticate via https://fish1-release.storage.googleapis.com/tutorials.html"
            raise ValueError(f"Skeleton retrieval requires CAVEclient, which is not initialized for {self.species}.{auth_msg}")
        return self.client.skeleton.get_skeleton(neuron_ids)
        
    def reset_colors(self):
        self.color_idx = 0
        self.neuron_color_map = {}
        for i, neuron_id in enumerate(self.neuron_ids):
            color = self.NEURON_COLORS[self.color_idx % len(self.NEURON_COLORS)]
            self.color_idx += 1

            self.neuron_color_map[neuron_id] = color

    async def load_neurons_parallel(self, neuron_ids: List[int], timeout: Optional[float] = None, batch_size: int = 40):
        """
        Load neuron meshes for the specified neuron IDs in parallel batches.
        
        Args:
            neuron_ids: List of neuron IDs to load
            timeout: Optional timeout in seconds for each neuron fetch
            batch_size: Number of neurons to request in parallel per batch
        """
        loaded_neurons_map = {}
        
        # Process neuron IDs in batches
        for i in range(0, len(neuron_ids), batch_size):
            batch = neuron_ids[i:i+batch_size]
            if self.verbose:
                print(f"Processing batch {i//batch_size + 1}/{math.ceil(len(neuron_ids)/batch_size)}: {batch}")

            # Create tasks for this batch
            tasks = [self._fetch_neuron_mesh(nid, timeout=timeout) for nid in batch]

            # Run batch tasks concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    if self.verbose:
                        print(f"An unexpected error occurred during mesh fetching: {result}")
                    continue
                elif isinstance(result, tuple) and len(result) == 2:
                    neuron_id, mesh = result
                    loaded_neurons_map[neuron_id] = mesh
                    if mesh is None and self.verbose:
                        print(f"Skipping neuron {neuron_id} due to loading error.")
                else:
                    if self.verbose:
                        print(f"Unexpected result format: {result}")
        
        return loaded_neurons_map

    def load_em_data(self, x: int, y: int, z: int, window_size_nm: int = 512, window_z: int = 1, segmentation_method: str = "api", recalculate: Optional[bool] = None):
        """
        Load EM data around the specified coordinates.
        
        Args:
            x: X coordinate in the original space
            y: Y coordinate in the original space
            z: Z coordinate in the original space
            window_size: Size of the window to load (in pixels)
            window_z: Number of z-slices to load (default: 1, for a single slice)
            segmentation_method: Method to use for segmentation data:
                - "direct": Use direct segmentation volume (fastest)
                - "mesh": Use mesh-based segmentation (for neurons not in segmentation)
                - "api": Use FlyWire API to get segmentation (most accurate but slower)
        """
        # Store the current location
        self.current_location = (x, y, z, window_size_nm, window_z)
        
        # Store coordinates
        self.x = x # in nm // 4
        self.y = y # in nm // 4
        self.z = z # in nm
        
        # Calculate window boundaries
        self.min_x = self.x - window_size_nm
        self.max_x = self.x + window_size_nm
        self.min_y = self.y - window_size_nm
        self.max_y = self.y + window_size_nm
        
        # Calculate z window boundaries
        half_window_z = window_z // 2
        self.min_z = self.z - half_window_z * self.em_resolution[2]
        self.max_z = self.z + (window_z - half_window_z) * self.em_resolution[2]

        if self.verbose:
            print(f"Loading EM data at ({x}, {y}, {z}) with window size {window_size_nm} nm and z-window {window_z}")
        # Load EM data
        try:
            self.vol_em = self.cv_em[self.min_x//self.em_resolution[0]:self.max_x//self.em_resolution[0], 
                                     self.min_y//self.em_resolution[1]:self.max_y//self.em_resolution[1], 
                                     self.min_z//self.em_resolution[2]:self.max_z//self.em_resolution[2]][:,:,:,0]

            # Load segmentation data if using direct method
            # if segmentation_method == "direct":
            self.vol_supervoxels = self.cv_seg[self.min_x//self.seg_resolution[0]:self.max_x//self.seg_resolution[0], 
                                        self.min_y//self.seg_resolution[1]:self.max_y//self.seg_resolution[1], 
                                        self.min_z//self.seg_resolution[2]:self.max_z//self.seg_resolution[2]][:,:,:,0]
            
            if not np.all(np.array(self.em_resolution) == np.array(self.seg_resolution)):
                em_res = np.array(self.em_resolution)
                seg_res = np.array(self.seg_resolution)
                # Check if segmentation needs upsampling (seg is lower res than EM)
                if np.all(seg_res >= em_res):
                    if self.verbose:
                        print(f"Upsampling segmentation from {self.seg_resolution} to match EM {self.em_resolution}")
                    self.vol_supervoxels = self.upsample_segmentation(self.vol_supervoxels, self.em_resolution, self.seg_resolution)
                # Check if segmentation needs downsampling (seg is higher res than EM)
                elif np.all(seg_res <= em_res):
                    if self.verbose:
                        print(f"Downsampling segmentation from {self.seg_resolution} to match EM {self.em_resolution}")
                    self.vol_supervoxels = self.downsample_segmentation(self.vol_supervoxels, self.em_resolution, self.seg_resolution)
                else:
                    if self.verbose:
                        print(f"Warning: Mixed resolution relationship between EM {self.em_resolution} and seg {self.seg_resolution}")
            if self.verbose:
                print(f"Loaded EM data at ({x}, {y}, {z}) with window size {window_size_nm} nm and z-window {window_z}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading EM data: {e}")
            return False

        if self.verbose:
            print(f"EM data shape: {self.vol_em.shape}")
        
        # Create coordinate grids
        self._create_coordinate_grids()
        
        neurons_in_vol = None
        # Process segmentation data if neurons are loaded
        if self.neurons:
            if segmentation_method == "mesh":
                self._process_mesh_segmentation()
            elif segmentation_method == "api":
                # Check if we already have the root_ids_grid for this location
                if recalculate is None:
                    if self.current_location in self.root_ids_grids:
                        neurons_in_vol = self._process_segmentation_from_api(recalculate=False)
                    else:
                        neurons_in_vol = self._process_segmentation_from_api(recalculate=True)
                else:
                    neurons_in_vol = self._process_segmentation_from_api(recalculate=recalculate)
            else:  # direct
                self._process_segmentation_from_volume()
        
        return neurons_in_vol
    
    def upsample_segmentation(self, vol_supervoxels: np.ndarray, em_resolution: Tuple[int, int, int], seg_resolution: Tuple[int, int, int]):
        """
        Upsamples the segmentation data to match EM resolution by tiling.

        Args:
            vol_supervoxels: The segmentation volume (supervoxel IDs).
            em_resolution: The resolution of the EM data (e.g., (4, 4, 40)).
            seg_resolution: The resolution of the segmentation data (e.g., (16, 16, 40)).

        Returns:
            np.ndarray: The upsampled segmentation volume.
        """

        ratio_x = seg_resolution[0] // em_resolution[0]
        ratio_y = seg_resolution[1] // em_resolution[1]
        ratio_z = seg_resolution[2] // em_resolution[2]

        if not (seg_resolution[0] % em_resolution[0] == 0 and \
                seg_resolution[1] % em_resolution[1] == 0 and \
                seg_resolution[2] % em_resolution[2] == 0):
            if self.verbose:
                print("Warning: Upsampling ratios are not integers. This might lead to unexpected results.")
            # Fallback or raise error if non-integer ratios are not supported
            # For now, we'll proceed with integer division, which might be incorrect

        # Target shape based on the ratios.
        # If vol_supervoxels has shape (sx, sy, sz)
        # The new shape will be (sx * ratio_x, sy * ratio_y, sz * ratio_z)
        
        # The CloudVolume download gives (x,y,z,channel), so we squeeze it to (x,y,z)
        # So vol_supervoxels.shape is (nx_seg, ny_seg, nz_seg)
        nx_seg, ny_seg, nz_seg = vol_supervoxels.shape
        
        nx_em = nx_seg * ratio_x
        ny_em = ny_seg * ratio_y
        nz_em = nz_seg * ratio_z
        
        upsampled_vol = np.zeros((nx_em, ny_em, nz_em), dtype=vol_supervoxels.dtype)
        
        for i in range(nx_seg):
            for j in range(ny_seg):
                for k in range(nz_seg):
                    upsampled_vol[i*ratio_x:(i+1)*ratio_x, 
                                  j*ratio_y:(j+1)*ratio_y, 
                                  k*ratio_z:(k+1)*ratio_z] = vol_supervoxels[i, j, k]
                                  
        return upsampled_vol

    def downsample_segmentation(self, vol_supervoxels: np.ndarray, em_resolution: Tuple[int, int, int], seg_resolution: Tuple[int, int, int]):
        """
        Downsamples the segmentation data to match EM resolution by subsampling.
        Uses nearest-neighbor (takes every Nth voxel) to preserve segment IDs.

        Args:
            vol_supervoxels: The segmentation volume (supervoxel IDs).
            em_resolution: The resolution of the EM data (e.g., (32, 32, 30)).
            seg_resolution: The resolution of the segmentation data (e.g., (16, 16, 30)).

        Returns:
            np.ndarray: The downsampled segmentation volume.
        """
        ratio_x = int(em_resolution[0] // seg_resolution[0])
        ratio_y = int(em_resolution[1] // seg_resolution[1])
        ratio_z = int(em_resolution[2] // seg_resolution[2])

        # Ensure ratios are at least 1
        ratio_x = max(1, ratio_x)
        ratio_y = max(1, ratio_y)
        ratio_z = max(1, ratio_z)

        if self.verbose:
            print(f"Downsample ratios: x={ratio_x}, y={ratio_y}, z={ratio_z}")

        # Subsample by taking every Nth voxel
        downsampled_vol = vol_supervoxels[::ratio_x, ::ratio_y, ::ratio_z]
        
        return downsampled_vol

    def _create_coordinate_grids(self):
        """Create coordinate grids for 2D and 3D visualizations."""
        if self.vol_em is None:
            if self.verbose:
                print("EM data not loaded. Call load_em_data first.")
            return
        
        # Create coordinates for 2D plot in original space
        self.x_2d = np.linspace(self.min_x, self.max_x, self.vol_em.shape[1])
        self.y_2d = np.linspace(self.min_y, self.max_y, self.vol_em.shape[0])
        
        # Create coordinates for 3D plot with EM slice
        self.X_3d, self.Y_3d = np.meshgrid(self.x_2d, self.y_2d)
        
        # For visualization, we'll use the central z-slice
        central_z_idx = self.vol_em.shape[2] // 2
        self.central_z = self.min_z + central_z_idx * self.em_resolution[2]
        self.Z_3d = np.ones_like(self.X_3d) * self.central_z
    
    def _process_mesh_segmentation(self):
        """Process neuron meshes to create segmentation data for the current EM slice."""
        if not self.neurons:
            if self.verbose:
                print("No neurons loaded. Call load_neurons first.")
            return
        
        # Create empty segmentation data dictionary
        self.segmentation_data = {}
        self.nonzero_points = {}
        
        # Get the z-plane in original coordinates
        z_plane = self.central_z
        z_tolerance = 20  # Tolerance in nm to consider vertices part of the slice
        
        # Process each neuron
        for i, neuron in enumerate(self.neurons):
            neuron_id = self.neuron_ids[i]
            
            # Create empty segmentation array for this neuron
            seg_array = np.zeros_like(self.vol_em[:,:,0], dtype=np.int64)
            
            # Get vertices from the neuron mesh
            vertices = neuron.vertices
            
            # Scale vertices to match EM resolution (16,16,40)
            scaled_vertices = vertices # / np.array([16, 16, 40])
            
            # Find vertices that are in the current z-plane
            z_mask = np.abs(vertices[:, 2] - z_plane) <= z_tolerance
            slice_vertices = scaled_vertices[z_mask]
            
            # Find vertices that are within the current x,y window
            x_mask = (slice_vertices[:, 0] >= self.min_x) & (slice_vertices[:, 0] < self.max_x)
            y_mask = (slice_vertices[:, 1] >= self.min_y) & (slice_vertices[:, 1] < self.max_y)
            window_mask = x_mask & y_mask
            
            window_vertices = slice_vertices[window_mask]
            
            # Convert to pixel coordinates within the segmentation array
            pixel_x = window_vertices[:, 0] - self.min_x
            pixel_y = window_vertices[:, 1] - self.min_y
            
            # Round to integers
            pixel_x = np.round(pixel_x).astype(int)
            pixel_y = np.round(pixel_y).astype(int)
            
            # Ensure coordinates are within bounds
            valid_mask = (pixel_x >= 0) & (pixel_x < seg_array.shape[1]) & (pixel_y >= 0) & (pixel_y < seg_array.shape[0])
            pixel_x = pixel_x[valid_mask]
            pixel_y = pixel_y[valid_mask]
            
            # Set segmentation values
            if len(pixel_x) > 0:
                seg_array[pixel_y, pixel_x] = neuron_id
                
                # Map back to original coordinates for visualization
                x_coords = self.x_2d[pixel_x]
                y_coords = self.y_2d[pixel_y]
                
                self.nonzero_points[neuron_id] = (x_coords, y_coords)
                self.segmentation_data[neuron_id] = seg_array

                if self.verbose:
                    print(f"Found {len(pixel_x)} points for neuron {neuron_id} in the current slice")
            else:
                if self.verbose:
                    print(f"No points found for neuron {neuron_id} in the current slice")
                self.nonzero_points[neuron_id] = (np.array([]), np.array([]))
                self.segmentation_data[neuron_id] = seg_array
    
    def _process_segmentation_from_volume(self):
        """Process segmentation data from the segmentation volume."""
        if self.vol_seg is None or not self.neuron_ids:
            if self.verbose:
                print("Segmentation data or neuron IDs not available.")
            return
        
        # Create empty segmentation data dictionary
        self.segmentation_data = {}
        self.nonzero_points = {}
        
        # Process each neuron ID
        for neuron_id in self.neuron_ids:
            # Create a mask for this neuron ID across all z-slices
            mask = self.vol_seg == neuron_id
            
            # Create segmentation array for this neuron
            seg_array = np.zeros_like(self.vol_em, dtype=np.int64)
            seg_array[mask] = neuron_id
            
            # Get nonzero points
            nonzero_indices = np.where(mask)
            
            if len(nonzero_indices[0]) > 0:
                # Store the full 3D indices
                self.nonzero_points[neuron_id] = nonzero_indices
                self.segmentation_data[neuron_id] = seg_array

                if self.verbose:
                    print(f"Found {len(nonzero_indices[0])} points for neuron {neuron_id} across all z-slices")
            else:
                if self.verbose:
                    print(f"No points found for neuron {neuron_id} in any z-slice")
                self.nonzero_points[neuron_id] = np.array([])
                self.segmentation_data[neuron_id] = seg_array

    def _process_segmentation_from_api(self, dataset: Optional[str] = None, timestamp: Optional[str] = None, recalculate: bool = True):
        """
        Process segmentation data using the FlyWire API.
        
        Args:
            dataset: Optional dataset to use instead of the default
            timestamp: Optional timestamp for the segmentation data
            recalculate: Whether to recalculate the root_ids_grid or reuse the cached one
        """
        if self.client is None:
            auth_msg = ""
            if self.species == "human":
                auth_msg = " Please authenticate via https://h01-release.storage.googleapis.com/proofreading.html"
            elif self.species == "zebrafish":
                auth_msg = " Please authenticate via https://fish1-release.storage.googleapis.com/tutorials.html"
            raise ValueError(f"API-based segmentation processing requires CAVEclient, which is not initialized for {self.species}.{auth_msg}")
        
        # if not self.neuron_ids:
        #     print("No neuron IDs available.")
        #     return
        
        # Create empty segmentation data dictionary
        self.segmentation_data = {}
        self.nonzero_points = {}
        
        # If we need to recalculate the root_ids_grid or it doesn't exist for this location
        if recalculate or self.current_location not in self.root_ids_grids:
            if self.verbose:
                print(f"Calculating root IDs grid for location {self.current_location} (this may take a while)...")

            # Create a grid of x,y coordinates for the EM slice
            x_coords, y_coords = np.meshgrid(self.x_2d, self.y_2d)
            
            # Create z coordinates for the full slab
            z_coords = np.linspace(self.min_z, self.max_z, self.vol_em.shape[2])
            
            # Create a 3D grid of coordinates
            X, Y, Z = np.meshgrid(self.x_2d, self.y_2d, z_coords)
            
            # Stack the coordinates into a (N, 3) array
            locs = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
            

            if self.timestamp is None:
                if self.verbose:
                    print("No timestamp provided, using latest timestamp")
                root_ids = self.client.chunkedgraph.get_roots(self.vol_supervoxels)
            else:
                root_ids = self.client.chunkedgraph.get_roots(self.vol_supervoxels, timestamp=datetime.fromtimestamp(self.timestamp, timezone.utc))

            # Reshape back to the original grid shape
            root_ids_grid = root_ids.reshape(X.shape)

            # Store in our dictionary
            self.root_ids_grids[self.current_location] = root_ids_grid
            if self.verbose:
                print("Root IDs grid calculation complete.")
        else:
            if self.verbose:
                print(f"Reusing cached root IDs grid for location {self.current_location}.")
            root_ids_grid = self.root_ids_grids[self.current_location]
        
        # Get the central z-slice index for visualization
        central_z_idx = self.vol_em.shape[2] // 2
        
        neurons_in_vol = []
        # Process each neuron ID
        for neuron_id in self.neuron_ids:
            # Create a mask for this neuron ID in the central z-slice for visualization
            mask = root_ids_grid[:, :, :] == neuron_id
            
            # Create segmentation array for this neuron
            seg_array = np.zeros_like(self.vol_em, dtype=np.int64)
            seg_array[mask] = neuron_id
            
            # Get nonzero points
            nonzero_indices = np.where(mask)
            
            if len(nonzero_indices[0]) > 0:


                self.nonzero_points[neuron_id] = nonzero_indices #(x_coords_nonzero, y_coords_nonzero)
                self.segmentation_data[neuron_id] = seg_array
                neurons_in_vol.append(neuron_id)
                if self.verbose:
                    print(f"Found {len(nonzero_indices[0])} points for neuron {neuron_id} in the current slice")
            else:
                if self.verbose:
                    print(f"No points found for neuron {neuron_id} in the current slice")
                self.nonzero_points[neuron_id] = np.array([])
                self.segmentation_data[neuron_id] = seg_array
        return neurons_in_vol
    
    def get_adjacent_neurons(self, neuron_id: int):
        """
        Get the adjacent neurons to a given neuron by finding its border pixels
        within the current root_ids_grid.
        """
        if self.current_location not in self.root_ids_grids:
            if self.verbose:
                print("Root IDs grid not available for the current location. Load EM data first.")
            return None, None, None

        root_ids_grid = self.root_ids_grids[self.current_location]

        if neuron_id not in self.nonzero_points or len(self.nonzero_points[neuron_id]) == 0:
            if self.verbose:
                print(f"No points found for neuron {neuron_id} in the current volume.")
            return [], [], []

        # nonzero_points stores indices as (dim0_indices, dim1_indices, dim2_indices)
        # which correspond to (x_indices, y_indices, z_indices) after meshgrid and flattening.
        # The root_ids_grid is typically (y_dim, x_dim, z_dim) or (x_dim, y_dim, z_dim)
        # Let's assume root_ids_grid is (y_grid_dim, x_grid_dim, z_grid_dim)
        # and self.nonzero_points[neuron_id] are indices into this grid.
        # If self.X_3d, self.Y_3d, self.Z_3d were created with meshgrid(self.x_2d, self.y_2d, z_coords_em)
        # and locs = np.column_stack((X.flatten(), Y.flatten(), Z.flatten())),
        # then root_ids_grid = root_ids.reshape(X.shape) means root_ids_grid has shape (y_len, x_len, z_len)
        # So, nonzero_points[neuron_id] will be (y_indices, x_indices, z_indices) if derived from this.

        neuron_voxels_indices = self.nonzero_points[neuron_id] # This is a tuple of arrays (ys, xs, zs)
        
        # Convert tuple of arrays to a list of (y, x, z) tuples
        neuron_voxel_coords = list(zip(neuron_voxels_indices[0], neuron_voxels_indices[1], neuron_voxels_indices[2]))

        border_pixels = []
        adjacent_supervoxels = set()

        shape = root_ids_grid.shape # (y_dim, x_dim, z_dim)

        for y, x, z in neuron_voxel_coords:
            is_border_pixel = False
            # Check 6 neighbors
            for dy, dx, dz in [(0,1,0), (0,-1,0), (1,0,0), (-1,0,0)]:
                ny, nx, nz = y + dy, x + dx, z + dz

                # Check bounds
                if 0 <= ny < shape[0] and 0 <= nx < shape[1] and 0 <= nz < shape[2]:
                    neighbor_root_id = root_ids_grid[ny, nx, nz]
                    if neighbor_root_id != neuron_id:
                        is_border_pixel = True
                        if neighbor_root_id != 0: # Assuming 0 is background or unassigned
                            adjacent_supervoxels.add(neighbor_root_id)
                        break # Found a differing neighbor, this is a border pixel
                else:
                    # Neighbor is out of bounds, so this is a border pixel
                    is_border_pixel = True
                    break 
            
            if is_border_pixel:
                # Convert grid indices back to world coordinates for consistency if needed,
                # or return grid indices. For now, returning grid indices.
                # world_x = self.x_2d[x]
                # world_y = self.y_2d[y]
                # world_z = self.min_z + z * self.em_resolution[2] # Assuming z is index in EM z-slab
                border_pixels.append((y, x, z)) # Storing as (y,x,z) grid indices

        if self.verbose:
            print(f"Found {len(border_pixels)} border pixels for neuron {neuron_id}.")
            print(f"Adjacent supervoxels: {list(adjacent_supervoxels)}")

        if self.client is None:
            if self.verbose:
                print(f"Note: CAVEclient not available for {self.species}. Returning supervoxel IDs instead of root IDs.")
            # Return supervoxel IDs as root IDs when CAVEclient is unavailable
            return border_pixels, list(adjacent_supervoxels), list(adjacent_supervoxels)
        
        root_ids = self.client.chunkedgraph.get_roots(list(adjacent_supervoxels))
        return border_pixels, list(adjacent_supervoxels), list(root_ids)

    def create_3d_neuron_figure(self, selected_neuron_ids: List[int] = None, bbox: Bbox = None, add_em_slice: bool = True):
        """
        Create a 3D figure with neurons and EM slice.

        Returns:
            Plotly figure object
        """
        if not self.neurons:
            if self.verbose:
                print("No neurons loaded. Call load_neurons first.")
            return None

        if add_em_slice and self.vol_em is None:
            if self.verbose:
                print("No EM data loaded. Call load_em_data first.")
            return None
        
        # Create figure
        self.neuron_fig = oc.Viewer(offscreen=True) # go.Figure()
        # self.neuron_fig.show_bounds = True
        

        # Add neurons to the figure
        for i, neuron in enumerate(self.neurons):
            neuron_id = self.neuron_ids[i]
            if selected_neuron_ids is not None and neuron_id not in selected_neuron_ids:
                continue
            color = self.neuron_color_map[neuron_id]
            # neuron_fig = navis.plot3d(neuron, color=color, alpha=1, backend='octarine')

            
            # for trace in neuron_fig.data:
            #     # Remove neuron from legend
            #     trace.showlegend = False
            #     self.neuron_fig.add_trace(trace)
            if bbox is not None:
                cropped_neuron = neuron.crop(bbox)
                self.neuron_fig.add(cropped_neuron, color = color)
            else:
                self.neuron_fig.add(neuron, color = color)

        return self.neuron_fig
    
    def create_em_segmentation_figure(self, selected_neuron_ids: List[int] = None, z_slice: int = None, marker_size: int = 1, marker_opacity: float = 0.1, pointer_xyz: Tuple[int, int, int] = None, minimal_output: bool = False):
        """
        Create a 2D figure with EM slice and segmentation overlay.
        
Args:
            selected_neuron_ids: List of neuron IDs to include in the visualization
            z_slice: Specific z-slice to visualize (if None, uses the central slice)
            marker_size: Size of markers for segmentation overlay
            marker_opacity: Opacity of markers for segmentation overlay
            pointer_xyz: Tuple of (x,y,z) to mark a point on the EM slice.
            minimal_output: If True, produces an image with no axes, titles, legend, or background.
            
        Returns:
            octarine figure object
        """
        if self.vol_em is None:
            if self.verbose:
                print("EM data not loaded.")
            return None
        
        # Get the central z-slice index
        if z_slice is None:
            central_z_idx = self.vol_em.shape[2] // 2
        else:
            central_z_idx = z_slice
        
        # Create figure
        self.em_seg_fig = go.Figure()
        
        # Add EM slice
        self.em_seg_fig.add_trace(
            go.Heatmap(
                z=self.vol_em[:,:,central_z_idx].T,
                x=self.x_2d,
                y=self.y_2d,
                colorscale='Gray',
                showscale=False,
                name='EM Slice'
            )
        )
        if self.nonzero_points is not None:
            # Add segmentation overlay
            for neuron_id, nonzero_indices in self.nonzero_points.items():
                if selected_neuron_ids is not None and neuron_id not in selected_neuron_ids:
                    continue
                if len(nonzero_indices) > 0:
                    # Filter points to only include those in the current z-slice

                    z_mask = nonzero_indices[2] == central_z_idx
                    x_coords = self.x_2d[nonzero_indices[0][z_mask]]
                    y_coords = self.y_2d[nonzero_indices[1][z_mask]]
                    
                    if len(x_coords) > 0:
                        self.em_seg_fig.add_trace(
                            go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode='markers',
                                marker=dict(
                                    color=self.neuron_color_map[neuron_id],
                                    size=marker_size,
                                    opacity=marker_opacity
                                ),
                                name=f'Neuron {neuron_id}'
                            )
                        )
        
        # Clear previous annotations and shapes if any
        if hasattr(self.em_seg_fig.layout, 'annotations'):
            self.em_seg_fig.layout.annotations = []
        if hasattr(self.em_seg_fig.layout, 'shapes'):
            self.em_seg_fig.layout.shapes = []

        if pointer_xyz is not None:
            # Add a red 'X' marker at the pointer_xyz location
            self.em_seg_fig.add_trace(
                go.Scatter(
                    x=[pointer_xyz[0]],
                    y=[pointer_xyz[1]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='x'
                    ),
                    name='Pointer',
                    showlegend=False
                )
            )

        # Update layout with increased margins to ensure labels are visible
        self.em_seg_fig.update_layout(
            title="EM Slice with Segmentation",
            xaxis=dict(
                scaleanchor="y", 
                scaleratio=1, 
                title_text="X (nm)",
                title_standoff=15  # Add more space for the title
            ),
            yaxis=dict(
                title_text="Y (nm)",
                title_standoff=15  # Add more space for the title
            ),
            margin=dict(l=80, r=80, b=80, t=100),
            height=700,
            width=900
        )
        
        if minimal_output:
            self.em_seg_fig.update_layout(
                xaxis_visible=False,
                yaxis_visible=False,
                title_text=None,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, b=0, t=0)
            )
            # Ensure heatmap and scatter traces are still visible
            for trace in self.em_seg_fig.data:
                if trace.type == 'heatmap':
                    trace.showscale = False # Also hide colorscale for heatmap
        
        return self.em_seg_fig

    def crop_images(self, im: Image.Image, base_filename: str, suffix: str, width: int, height: int):
        image_width, image_height = im.size
                  
      
        arr = np.asarray(im)
        alpha = arr[:, :, 3]
        ys, xs = np.nonzero(alpha)          # row indices first, then column indices
        non_empty_coords = list(zip(xs, ys))
        max_pixels = np.max(np.array(non_empty_coords))
        min_pixels = np.min(np.array(non_empty_coords))
        distance_from_center = max(np.abs(len(alpha)//2-max_pixels), np.abs(len(alpha)//2-min_pixels))
        ratio = len(alpha)//2/distance_from_center
        self.neuron_fig.screenshot(os.path.join(self.output_dir, f"{base_filename}_{suffix}_1.png"), alpha=True, size=(int(ratio*width), int(ratio*height)))
        im = Image.open(os.path.join(self.output_dir, f"{base_filename}_{suffix}_1.png"))
        image_width, image_height = im.size
        im1 = im.crop((image_width//2-width, image_height//2-height, image_width//2+width, image_height//2+height))
        im1.resize((width, height))
        im1.save(os.path.join(self.output_dir, f"{base_filename}_{suffix}.png"))
        os.remove(os.path.join(self.output_dir, f"{base_filename}_{suffix}_1.png"))

    def save_3d_views(self, bbox: Bbox = None, base_filename: str = "3d_neuron_mesh", width: int = 512, height: int = 512, scale: float = 1.0, mode: str = "3d_view", crop=True):
        """
        Save 3D views from different angles.
        
        Args:
            base_filename: Base filename for the output files
            width: Width of the saved image in pixels
            height: Height of the saved image in pixels
            scale: Scale factor for the image resolution (higher values produce sharper images)
        """

        if self.neuron_fig is None:
            self.create_3d_neuron_figure(add_em_slice=False)
        
        minpt = None
        maxpt = None
        if bbox is None:
            for neuron in self.neurons:
                if minpt is None:
                    minpt = np.min(neuron.vertices, axis=0)
                    maxpt = np.max(neuron.vertices, axis=0)
                else:
                    minpt = np.minimum(minpt, np.min(neuron.vertices, axis=0))
                    maxpt = np.maximum(maxpt, np.max(neuron.vertices, axis=0))
            bbox = Bbox(minpt, maxpt)

        center = (bbox.minpt + bbox.maxpt) / 2

        self.create_3d_neuron_figure(bbox=bbox, add_em_slice=False)
        front_path = os.path.join(self.output_dir, f"{base_filename}_front.png")
        cam = self.neuron_fig.camera
        cam.local.position = (bbox.minpt[0], center[1], center[2])
        cam.look_at(center)
        # Set the camera position from the front view
        self.neuron_fig.screenshot(front_path, alpha=True, size=(width, height))
        im = Image.open(front_path)
        if crop:
            self.crop_images(im, base_filename, "front", width, height)
        else:
            im.save(front_path)
        if self.verbose:
            print(f"Saved front view to {front_path} (size: {width}x{height}, scale: {scale})")


        self.create_3d_neuron_figure(bbox=bbox, add_em_slice=False)
        side_path = os.path.join(self.output_dir, f"{base_filename}_side.png")
        cam = self.neuron_fig.camera
        cam.local.position = (center[0], bbox.minpt[1], center[2])
        cam.look_at(center)
        self.neuron_fig.screenshot(side_path, alpha=True, size=(width, height))
        im = Image.open(side_path)
        if crop:
            self.crop_images(im, base_filename, "side", width, height)
        else:
            im.save(side_path)
        if self.verbose:
            print(f"Saved side view to {side_path} (size: {width}x{height}, scale: {scale})")


        self.create_3d_neuron_figure(bbox=bbox, add_em_slice=False)
        top_path = os.path.join(self.output_dir, f"{base_filename}_top.png")
        cam = self.neuron_fig.camera
        cam.local.position = (center[0], center[1], bbox.maxpt[2])
        cam.look_at(center)
        self.neuron_fig.screenshot(top_path, alpha=True, size=(width, height))
        im = Image.open(top_path)
        if crop:
            self.crop_images(im, base_filename, "top", width, height)
        else:
            im.save(top_path)
        if self.verbose:
            print(f"Saved top view to {top_path} (size: {width}x{height}, scale: {scale})")

        return {
            'front': front_path,
            'top': top_path,
            'side': side_path
        }

    def save_em_segmentation(self, filename: str = "em.png", width: int = 1200, height: int = 900, scale: float = 2.0):
        """
        Save EM slice with segmentation overlay.
        
        Args:
            filename: Filename for the output file
            width: Width of the saved image in pixels
            height: Height of the saved image in pixels
            scale: Scale factor for the image resolution (higher values produce sharper images)
        """
        if self.em_seg_fig is None:
            self.create_em_segmentation_figure()

        # Update the figure size for saving
        self.em_seg_fig.update_layout(
            width=width,
            height=height
        )

        output_path = os.path.join(self.output_dir, filename)
        self.em_seg_fig.write_image(output_path, scale=scale)
        if self.verbose:
            print(f"Saved EM slice with segmentation to {output_path} (size: {width}x{height}, scale: {scale})")
        return output_path

    def set_output_directory(self, output_dir: str):
        """
        Set the output directory for saved files.
        
        Args:
            output_dir: Path to the output directory
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if self.verbose:
            print(f"Output directory set to {self.output_dir}")

    def add_neurons(self, neuron_ids: List[int]):
        """
        Add new neurons to the visualization without reloading existing ones.
        
        Args:
            neuron_ids: List of neuron IDs to add
        
        Returns:
            List of newly added neurons
        """
        # Filter out neuron IDs that are already loaded
        new_neuron_ids = [nid for nid in neuron_ids if nid not in self.neuron_ids]

        if not new_neuron_ids:
            if self.verbose:
                print("All specified neurons are already loaded.")
            return []
        
        new_neurons = []
        
        # Load new neurons
        for i, neuron_id in enumerate(new_neuron_ids):
            # Assign a color from the color palette
            color = self.NEURON_COLORS[self.color_idx % len(self.NEURON_COLORS)]
            self.color_idx += 1

            self.neuron_color_map[neuron_id] = color
            
            try:
                # neuron = flywire.get_mesh_neuron(neuron_id, dataset=self.dataset)
                with suppress_stdout():
                    neuron = self.cv_seg.mesh.get(neuron_id)[neuron_id]
                self.neurons.append(neuron)
                self.neuron_ids.append(neuron_id)
                new_neurons.append(neuron)
                if self.verbose:
                    print(f"Loaded neuron {neuron_id}")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading neuron {neuron_id}: {e}")
        
        # Process segmentation for new neurons if we have EM data and current location
        if self.vol_em is not None and self.current_location in self.root_ids_grids:
            # Get the grid for the current location
            root_ids_grid = self.root_ids_grids[self.current_location]
            
            # Process only the new neurons
            for neuron_id in new_neuron_ids:
                # Create a mask for this neuron ID
                mask = root_ids_grid == neuron_id
                
                # Create segmentation array for this neuron
                seg_array = np.zeros_like(self.vol_em, dtype=np.int64)
                seg_array[mask] = neuron_id
                
                # Get nonzero points
                nonzero_indices = np.where(mask)
                
                if len(nonzero_indices[0]) > 0:
                    
                    self.nonzero_points[neuron_id] = nonzero_indices #(x_coords_nonzero, y_coords_nonzero)
                    self.segmentation_data[neuron_id] = seg_array

                    if self.verbose:
                        print(f"Found {len(nonzero_indices[0])} points for neuron {neuron_id} in the current slice")
                else:
                    if self.verbose:
                        print(f"No points found for neuron {neuron_id} in the current slice")
                    self.nonzero_points[neuron_id] = np.array([])
                    self.segmentation_data[neuron_id] = seg_array

        # Update existing figures if they exist
        if new_neurons and self.interactive_fig is not None:
            if self.verbose:
                print("Updating existing interactive figure with new neurons...")
            
            # Add new neurons to the 3D subplot
            for neuron in new_neurons:
                neuron_id = self.neuron_ids[self.neurons.index(neuron)]
                color = self.neuron_color_map[neuron_id]
                neuron_fig = navis.plot3d(neuron, color=color, alpha=1, backend='octarine')
                
                for trace in neuron_fig.data:
                    self.interactive_fig.add_trace(trace, row=1, col=1)
            
            # Add segmentation overlay for new neurons to the 2D subplot
            for neuron_id in new_neuron_ids:
                nonzero_indices = self.nonzero_points[neuron_id]
                if len(nonzero_indices) > 0:
                    color = self.neuron_color_map[neuron_id]
                    self.interactive_fig.add_trace(
                        go.Scatter(
                            x=nonzero_indices[0],
                            y=nonzero_indices[1],
                            mode='markers',
                            marker=dict(
                                color=color,
                                size=1,
                                opacity=0.1
                            ),
                            name=f'Neuron {neuron_id}'
                        ),
                        row=1, col=2
                    )
            
            # Also update other figures if they exist
            if self.neuron_fig is not None:
                self.create_3d_neuron_figure()
            
            if self.em_seg_fig is not None:
                self.create_em_segmentation_figure()
            
            # Update the display
            # self.update_interactive_display()
        
        return new_neurons

    def find_nearest_mesh_point(self, neuron_id: int, location: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Find the nearest point on a neuron's mesh to a given 3D location.
        
        Args:
            neuron_id: ID of the neuron to search
            location: Tuple of (x, y, z) coordinates in nm
            
        Returns:
            Tuple of (x, y, z) coordinates in nm of the nearest point on the mesh
            
        Raises:
            ValueError: If the neuron is not loaded
        """
        # Check if the neuron is loaded
        if neuron_id not in self.neuron_ids:
            # Try to load the neuron
            try:
                if self.verbose:
                    print(f"Neuron {neuron_id} not loaded. Attempting to load it now...")
                self.add_neurons([neuron_id])
            except Exception as e:
                raise ValueError(f"Neuron {neuron_id} is not loaded and could not be loaded: {e}")

        # Get the index of the neuron
        neuron_index = self.neuron_ids.index(neuron_id)
        neuron = self.neurons[neuron_index]

        # Get the vertices from the neuron mesh
        vertices = neuron.vertices
        
        # Convert input location to numpy array
        location_array = np.array(location)
        
        # Calculate distances from the location to all vertices
        distances = np.linalg.norm(vertices - location_array, axis=1)
        
        # Find the index of the minimum distance
        min_index = np.argmin(distances)
        
        # Get the nearest vertex
        nearest_point = vertices[min_index]
        
        # Print information about the search
        min_distance = distances[min_index]
        if self.verbose:
            print(f"Nearest point on neuron {neuron_id} to location {location}:")
            print(f"  Coordinates: ({nearest_point[0]:.2f}, {nearest_point[1]:.2f}, {nearest_point[2]:.2f})")
            print(f"  Distance: {min_distance:.2f} nm")
        
        return tuple(nearest_point)

    def visualize_nearest_point(self, neuron_id: int, location: Tuple[float, float, float], 
                               marker_size: int = 10, window_size: int = 256):
        """
        Visualize the nearest point on a neuron's mesh to a given 3D location.
        
        This method finds the nearest point, loads the EM data around it,
        and creates a visualization highlighting the point.
        
        Args:
            neuron_id: ID of the neuron to search
            location: Tuple of (x, y, z) coordinates in nm
            marker_size: Size of the marker for the nearest point
            window_size: Size of the EM data window to load
            
        Returns:
            The interactive figure object
        """
        # Find the nearest point
        nearest_point = self.find_nearest_mesh_point(neuron_id, location)
       
        raise NotImplementedError("Visualizing nearest point is not implemented yet")

    def find_nearest_skeleton_point(self, neuron_id: int, location: Tuple[float, float, float], 
                                 skeletonize_if_needed: bool = True) -> Tuple[float, float, float]:
        """
        Find the nearest point on a neuron's skeleton to a given 3D location.
        
        This method is more useful for navigating along the neuron's structure
        than just using mesh vertices, as it finds points along the centerline
        of the neuron.
        
        Args:
            neuron_id: ID of the neuron to search
            location: Tuple of (x, y, z) coordinates in nm
            skeletonize_if_needed: Whether to generate a skeleton if the neuron doesn't have one
            
        Returns:
            Tuple of (x, y, z) coordinates in nm of the nearest point on the skeleton
            
        Raises:
            ValueError: If the neuron is not loaded or cannot be skeletonized
        """
        # Check if the neuron is loaded
        if neuron_id not in self.neuron_ids:
            # Try to load the neuron
            try:
                if self.verbose:
                    print(f"Neuron {neuron_id} not loaded. Attempting to load it now...")
                self.add_neurons([neuron_id])
            except Exception as e:
                raise ValueError(f"Neuron {neuron_id} is not loaded and could not be loaded: {e}")

        # Get the index of the neuron
        neuron_index = self.neuron_ids.index(neuron_id)
        neuron = self.neurons[neuron_index]

        # Check if the neuron has a skeleton
        if not hasattr(neuron, 'nodes') or neuron.nodes is None or len(neuron.nodes) == 0:
            if skeletonize_if_needed:
                try:
                    if self.verbose:
                        print(f"Neuron {neuron_id} does not have a skeleton. Generating one now...")
                    # Use navis to skeletonize the neuron
                    skeleton = navis.skeletonize(neuron)
                    
                    # If skeletonization returns a list, take the first element
                    if isinstance(skeleton, list):
                        skeleton = skeleton[0]
                    
                    # Replace the neuron with the skeletonized version
                    self.neurons[neuron_index] = skeleton
                    neuron = skeleton
                except Exception as e:
                    raise ValueError(f"Failed to skeletonize neuron {neuron_id}: {e}")
            else:
                raise ValueError(f"Neuron {neuron_id} does not have a skeleton and skeletonize_if_needed is False")
        
        # Get the nodes from the skeleton
        nodes = neuron.nodes
        
        # Extract the x, y, z coordinates from the nodes
        if isinstance(nodes, pd.DataFrame):
            # If nodes is a DataFrame, extract the coordinates
            node_coords = nodes[['x', 'y', 'z']].values
        else:
            # If nodes is already a numpy array
            node_coords = nodes[:, :3]
        
        # Convert input location to numpy array
        location_array = np.array(location)
        
        # Calculate distances from the location to all nodes
        distances = np.linalg.norm(node_coords - location_array, axis=1)
        
        # Find the index of the minimum distance
        min_index = np.argmin(distances)
        
        # Get the nearest node
        nearest_point = node_coords[min_index]
        
        # Print information about the search
        min_distance = distances[min_index]
        if self.verbose:
            print(f"Nearest skeleton point on neuron {neuron_id} to location {location}:")
            print(f"  Coordinates: ({nearest_point[0]:.2f}, {nearest_point[1]:.2f}, {nearest_point[2]:.2f})")
            print(f"  Distance: {min_distance:.2f} nm")

        # If nodes is a DataFrame, we can get additional information
        if isinstance(nodes, pd.DataFrame) and self.verbose:
            node_id = nodes.iloc[min_index].name
            node_type = nodes.iloc[min_index].get('type', 'unknown')
            radius = nodes.iloc[min_index].get('radius', 0)
            print(f"  Node ID: {node_id}")
            print(f"  Node type: {node_type}")
            print(f"  Radius: {radius:.2f} nm")
        
        return tuple(nearest_point)

    def visualize_nearest_skeleton_point(self, neuron_id: int, location: Tuple[float, float, float], 
                                        marker_size: int = 10, window_size: int = 256):
        """
        Visualize the nearest point on a neuron's skeleton to a given 3D location.
        
        This method finds the nearest skeleton point, loads the EM data around it,
        and creates a visualization highlighting the point.
        
        Args:
            neuron_id: ID of the neuron to search
            location: Tuple of (x, y, z) coordinates in nm
            marker_size: Size of the marker for the nearest point
            window_size: Size of the EM data window to load
            
        Returns:
            The interactive figure object
        """
        # Find the nearest skeleton point
        nearest_point = self.find_nearest_skeleton_point(neuron_id, location)

        raise NotImplementedError("Visualizing nearest skeleton point is not implemented yet")

    def get_root_id_at_location(self, x: float, y: float) -> int:
        """
        Get the root ID at a specific 2D location in the current EM slice.
        
        This function takes coordinates from the 2D EM plot and returns the neuron ID
        (root ID) at that location. This is useful for identifying neurons directly
        from the visualization.
        
        Args:
            x: X coordinate in nm
            y: Y coordinate in nm
            
        Returns:
            The root ID at the specified location, or 0 if no neuron is present
            
        Raises:
            ValueError: If no EM data is loaded or the location is outside the current view
        """
        if self.current_location is None or self.current_location not in self.root_ids_grids:
            raise ValueError("No EM data loaded or no segmentation data available. Load EM data first.")
        
        # Get the root_ids_grid for the current location
        root_ids_grid = self.root_ids_grids[self.current_location]
        
        # Check if the coordinates are within the current view
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            raise ValueError(f"Coordinates ({x}, {y}) are outside the current view bounds: "
                            f"X: [{self.min_x}, {self.max_x}], Y: [{self.min_y}, {self.max_y}]")
        
        # Convert the coordinates to indices in the grid
        x_idx = np.argmin(np.abs(self.x_2d - x))
        y_idx = np.argmin(np.abs(self.y_2d - y))
        
        # Get the root ID at that location
        root_id = root_ids_grid[y_idx, x_idx]
        
        return root_id

    def find_neuron_interface(self, neuron_id1: int, neuron_id2: int, max_distance: float = 1000.0) -> Dict[str, Any]:
        """
        Find the barrier/interface between two loaded neuron meshes.
        
        This method identifies points on both neurons that are close to each other,
        effectively finding the interface or barrier between them. It calculates
        the minimum distance between vertices of the two neurons and returns
        information about the interface region.
        
        Args:
            neuron_id1: ID of the first neuron
            neuron_id2: ID of the second neuron
            max_distance: Maximum distance (in nm) to consider vertices as part of the interface
            
        Returns:
            Dictionary containing:
                - 'interface_points1': Points on neuron1 that are part of the interface
                - 'interface_points2': Points on neuron2 that are part of the interface
                - 'distances': Distances between corresponding interface points
                - 'min_distance': Minimum distance between any points on the two neurons
                - 'mean_distance': Mean distance between interface points
                - 'median_distance': Median distance between interface points
                - 'interface_area': Approximate area of the interface (in nm²)
                
        Raises:
            ValueError: If either neuron is not loaded
        """
        # Check if both neurons are loaded
        if neuron_id1 not in self.neuron_ids or neuron_id2 not in self.neuron_ids:
            raise ValueError(f"Both neurons must be loaded. Neuron1: {neuron_id1 in self.neuron_ids}, Neuron2: {neuron_id2 in self.neuron_ids}")
        
        # Get the indices of the neurons
        neuron_index1 = self.neuron_ids.index(neuron_id1)
        neuron_index2 = self.neuron_ids.index(neuron_id2)
        
        # Get the neurons
        neuron1 = self.neurons[neuron_index1]
        neuron2 = self.neurons[neuron_index2]
        
        # Get the vertices from both neuron meshes
        vertices1 = neuron1.vertices
        vertices2 = neuron2.vertices
        
        # Initialize lists to store interface points
        interface_points1 = []
        interface_points2 = []
        distances = []
        
        # Find the minimum distance between any points on the two neurons
        min_overall_distance = float('inf')
        
        # For efficiency, we'll use a KD-tree for the second neuron's vertices
        tree = cKDTree(vertices2)
        
        # Find points on neuron1 that are close to neuron2
        for i, point1 in enumerate(vertices1):
            # Find the nearest point on neuron2
            distance, index = tree.query(point1)
            
            # Update the minimum overall distance
            min_overall_distance = min(min_overall_distance, distance)
            
            # If the distance is less than the maximum distance, add to interface points
            if distance <= max_distance:
                interface_points1.append(point1)
                interface_points2.append(vertices2[index])
                distances.append(distance)
        
        # Convert lists to numpy arrays
        interface_points1 = np.array(interface_points1)
        interface_points2 = np.array(interface_points2)
        distances = np.array(distances)
        
        # Calculate statistics
        mean_distance = np.mean(distances) if len(distances) > 0 else float('inf')
        median_distance = np.median(distances) if len(distances) > 0 else float('inf')
        
        interface_point = (np.mean(interface_points2[distances<=min_overall_distance], axis=0)+np.mean(interface_points1[distances<=min_overall_distance], axis=0))/2
        # Print information about the interface
        if self.verbose:
            print(f"Interface between neurons {neuron_id1} and {neuron_id2}:")
            print(f"  Number of interface points: {len(interface_points1)}")
            print(f"  Minimum distance: {min_overall_distance:.2f} nm")
            print(f"  Mean distance: {mean_distance:.2f} nm")
            print(f"  Median distance: {median_distance:.2f} nm")

        # Return the interface information
        return {
            'interface_points1': interface_points1,
            'interface_points2': interface_points2,
            'distances': distances,
            'min_distance': min_overall_distance,
            'mean_distance': mean_distance,
            'median_distance': median_distance,
            'interface_point': interface_point
        }

    def find_normal_and_parallel_vector_to_interface(self, interface_points1: np.ndarray, interface_points2: np.ndarray, number_of_parallel_vectors: int = 3) -> np.ndarray:
        """
        Find the normal vector to the interface between two sets of points.
        
        This function calculates the normal vector to the plane defined by the interface points."""
        interface_point1 = np.mean(interface_points1, axis=0)
        interface_point2 = np.mean(interface_points2, axis=0)
        normal_vector = interface_point2 - interface_point1
        normal_vector /= np.linalg.norm(normal_vector)

        parallel_vectors = []

        parallel_vector = np.random.rand(len(normal_vector))
        parallel_vector -= np.dot(parallel_vector, normal_vector) * normal_vector
        parallel_vector /= np.linalg.norm(parallel_vector)

        for theta in np.linspace(0, 2*np.pi, number_of_parallel_vectors+1):
            rotated_vector = parallel_vector * np.cos(theta) + np.cross(normal_vector, parallel_vector) * np.sin(theta) + normal_vector * np.dot(normal_vector, parallel_vector) * (1-np.cos(theta))
            rotated_vector /= np.linalg.norm(rotated_vector)
            parallel_vectors.append(rotated_vector)
        parallel_vectors = parallel_vectors[:number_of_parallel_vectors]

        return normal_vector, parallel_vectors

    def get_neuron_vertex_count(self, neuron_id: int) -> Optional[int]:
        """
        Get the number of vertices for a given neuron ID at the current timestamp.
        
        Args:
            neuron_id: The ID of the neuron to query
            
        Returns:
            The number of vertices, or None if the neuron cannot be loaded.
        """
        try:
            with suppress_stdout():
                neuron = self.cv_seg.mesh.get(neuron_id)[neuron_id]
            if neuron and hasattr(neuron, 'vertices') and neuron.vertices is not None:
                return neuron.vertices.shape[0]
            else:
                if self.verbose:
                    print(f"Warning: Neuron {neuron_id} loaded but has no vertices.")
                return 0 # Or None, depending on desired behavior for empty meshes
        except Exception as e:
            if self.verbose:
                print(f"Error getting vertex count for neuron {neuron_id} at timestamp {self.timestamp}: {e}")
            return None

    def get_edit_history(self, neuron_id):
        """
        Get the edit history for a given neuron ID
        
        Args:
            neuron_id (str): The ID of the neuron to get the edit history for
        """
        if self.client is None:
            auth_msg = ""
            if self.species == "human":
                auth_msg = " Please authenticate via https://h01-release.storage.googleapis.com/proofreading.html"
            elif self.species == "zebrafish":
                auth_msg = " Please authenticate via https://fish1-release.storage.googleapis.com/tutorials.html"
            raise ValueError(f"Edit history retrieval requires CAVEclient, which is not initialized for {self.species}.{auth_msg}")
        
        return self.client.chunkedgraph.get_tabular_change_log(neuron_id, filtered=True)

    def is_neuron_id_valid(self, neuron_id: int) -> bool:
        """
        Check if a neuron ID is valid and can be loaded for the current species.
        
        This method attempts to load the neuron mesh to verify it exists and is accessible.
        
        Args:
            neuron_id: The neuron ID to validate
            
        Returns:
            True if the neuron ID is valid and can be loaded, False otherwise
        """
        try:
            with suppress_stdout():
                # Try to get the mesh - if it succeeds, the neuron ID is valid
                mesh = self.cv_seg.mesh.get(neuron_id)
                if neuron_id in mesh and mesh[neuron_id] is not None:
                    return True
                return False
        except Exception as e:
            if self.verbose:
                print(f"Neuron ID {neuron_id} is not valid: {e}")
            return False

    def clear_neurons(self):
        """Reset the state related to loaded neurons and clear figures."""
        self.neurons = []
        self.neuron_ids = []
        self.neuron_color_map = {}
        self.segmentation_data = {} # Consider if this should always be cleared or only if neurons change
        self.nonzero_points = {}   # Consider if this should always be cleared
        self.color_idx = 0
        
        # Clear existing figures
        self.neuron_fig = None
        self.em_seg_fig = None
        self.interactive_fig = None

        if self.verbose:
            print("Cleared loaded neurons and associated data/figures.")

    def remove_neurons(self, neuron_ids_to_remove: List[int]):
        """
        Remove specified neurons from the visualization.

        Args:
            neuron_ids_to_remove: List of neuron IDs to remove.
        """
        removed_count = 0
        indices_to_remove = []
        ids_actually_removed = []

        for i, neuron_id in enumerate(self.neuron_ids):
            if neuron_id in neuron_ids_to_remove:
                indices_to_remove.append(i)
                ids_actually_removed.append(neuron_id)

        # Remove elements in reverse order of index to avoid shifting issues
        indices_to_remove.sort(reverse=True)
        for index in indices_to_remove:
            del self.neurons[index]
            del self.neuron_ids[index]

        # Remove associated data
        for neuron_id in ids_actually_removed:
            self.neuron_color_map.pop(neuron_id, None)
            self.color_idx -= 1
            self.segmentation_data.pop(neuron_id, None)
            self.nonzero_points.pop(neuron_id, None)
            removed_count += 1
        
        self.reset_colors()

        if removed_count > 0:
            if self.verbose:
                print(f"Removed {removed_count} specified neurons: {ids_actually_removed}")
            # Clear figures as they are now potentially outdated
            self.neuron_fig = None
            self.em_seg_fig = None
            self.interactive_fig = None
            if self.verbose:
                print("Cleared existing figures as they may be outdated.")
        else:
            if self.verbose:
                print("No specified neurons found among loaded neurons.")                      # just re-centre the orbit :contentReference[oaicite:1]{index=1}

# Example usage
if __name__ == "__main__":
    # Create visualizer
    visualizer = ConnectomeVisualizer(output_dir="./connectome_output")
    
    # Load neurons
    visualizer.load_neurons([720575940625431866])
    visualizer.save_3d_views(base_filename="neurons_with_em_high_res")