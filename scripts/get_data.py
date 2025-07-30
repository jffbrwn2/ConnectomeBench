import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
import time
import asyncio
from tqdm import tqdm
import asyncio
from connectome_visualizer import ConnectomeVisualizer
import random
import caveclient

class TrainingDataGatherer:
    """
    A class for gathering training data at scale from FlyWire edit histories.
    
    This class processes edit histories to identify merge and split error corrections,
    and finds the locations of these edits using the neuron interface method.
    """
    
    def __init__(self, output_dir: str = "./training_data", species: str = "fly", vertices_threshold: int = 1000, valid_segment_vertices_threshold: int = 1000):
        """
        Initialize the TrainingDataGatherer.
        
        Args:
            output_dir: Directory to save output files
            vertices_threshold: Minimum number of vertices for a segment to be considered significant
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.vertices_threshold = vertices_threshold
        self.valid_segment_vertices_threshold = valid_segment_vertices_threshold
        # Initialize the FlyWireVisualizer
        self.visualizer = ConnectomeVisualizer(output_dir=output_dir, species=species)
         
        # Initialize data containers
        self.training_data = []
        
    def subtract_time_from_timestamp(self, timestamp_str: str, minutes: int = 1) -> str:
        """
        Subtract a specified amount of time from a timestamp string.
        
        Args:
            timestamp_str: The timestamp string to modify
            minutes: Number of minutes to subtract (default: 1)
            
        Returns:
            Updated timestamp string
        """
        try:
            # Parse the timestamp string to a datetime object
            # Assuming the timestamp is in a format that datetime can parse
            # If not, you may need to adjust the parsing logic
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Subtract the specified time
            new_dt = dt - timedelta(minutes=minutes)
            
            # Convert back to string in the same format
            return new_dt.isoformat()
        except Exception as e:
            print(f"Error subtracting time from timestamp: {e}")
            return timestamp_str
    async def process_neuron_edit(self, neuron_id: int, edit_info: Dict[str, Any], split_only: bool = False, merge_only: bool = False, edit_history: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Process a single edit for a given neuron ID.
        
        Args:
            neuron_id: ID of the neuron to process
        """
        # Extract information from the row
        timestamp = edit_info.get('timestamp')
        if isinstance(timestamp, int):
            timestamp = timestamp//1000
        elif isinstance(timestamp, datetime):
            timestamp = int(timestamp.timestamp())
        else:
            print(f"Warning: timestamp is not an int or datetime: {edit_info.get('timestamp')}")
            return None
        
        try:
            prev_timestamp = timestamp - 1
        except Exception as e:
            print(f"Error calculating previous timestamp: {e}")
            prev_timestamp = timestamp  # Fallback to current timestamp if error

        all_before_root_ids = edit_history['before_root_ids']
        all_before_root_ids = [x for y in all_before_root_ids for x in y]
        all_after_root_ids = edit_history['after_root_ids']
        all_after_root_ids = [x for y in all_after_root_ids for x in y]


        is_merge = edit_info.get('is_merge', False)
        if is_merge and split_only:
            print("Skipping merge operation because split_only is True")
            return None
        if not is_merge and merge_only:
            print("Skipping split operation because merge_only is True")
            return None

        operation_id = edit_info.get('operation_id')
        if is_merge:
            visualizer = ConnectomeVisualizer(output_dir=self.output_dir, species=self.visualizer.species, timestamp=prev_timestamp) 
        else:
            visualizer = ConnectomeVisualizer(output_dir=self.output_dir, species=self.visualizer.species, timestamp=timestamp) 
        visualizer.timestamp = timestamp
        visualizer._connect_to_data_sources()
        
        if is_merge and not split_only:

            # Get root IDs involved in the merge
            after_root_ids = edit_info.get('after_root_ids', [])
            before_root_ids = edit_info.get('before_root_ids', [])

            if not after_root_ids or not before_root_ids or len(before_root_ids) < 2:
                return None

            merged_neuron_id = after_root_ids[0]

            # --- Load Neurons Once and Filter/Sort ---
            vertex_counts = None
            loaded_neurons_map = {}
            try:

                # skeletons = visualizer.get_neuron_skelettons(before_root_ids)
                # if not all(len(skeleton['vertices']) >= self.skeleton_vertices_threshold for skeleton in skeletons):
                #     print(f"Skipping merge operation {operation_id}: No skeleton meets vertex threshold {self.skeleton_vertices_threshold}")
                #     return None
                # Load all neurons involved before the merge
                visualizer.load_neurons(before_root_ids)

                # Get vertex counts directly from loaded neurons
                # Map neuron ID to its vertex count, handling load failures (count=0)
                vertex_counts = {}
                for neuron in visualizer.neurons:
                    # Find the ID corresponding to the loaded neuron object
                    # This assumes self.visualizer.neuron_ids is correctly populated by load_neurons
                    neuron_idx = visualizer.neurons.index(neuron)
                    current_neuron_id = visualizer.neuron_ids[neuron_idx]
                    vertex_counts[current_neuron_id] = neuron.vertices.shape[0] if hasattr(neuron, 'vertices') and neuron.vertices is not None else 0
                    loaded_neurons_map[current_neuron_id] = neuron # Keep track of loaded neuron objects

                # Ensure all requested IDs have a count (even if 0 due to load failure)
                for rid in before_root_ids:
                    if rid not in vertex_counts:
                        vertex_counts[rid] = 0

                # Check if at least one segment meets the threshold
                if not all(count >= self.vertices_threshold for count in vertex_counts.values()):
                    print(f"Skipping merge operation {operation_id}: No segment meets vertex threshold {self.vertices_threshold}")
                    return None

                # Sort before_root_ids by vertex count (descending) using the obtained counts
                before_root_ids.sort(key=lambda rid: vertex_counts.get(rid, 0), reverse=True)

            except Exception as e:
                print(f"Error loading neurons or getting vertex counts for merge {operation_id}: {e}. Skipping.")
                return None # Skip this edit if loading/counting fails
            # --- End Load/Filter/Sort Logic ---

            # Store the edit information
            edit_info = {
                'neuron_id': neuron_id,
                'timestamp': timestamp,
                'prev_timestamp': prev_timestamp,
                'is_merge': True,
                'operation_id': operation_id,
                'merged_neuron_id': merged_neuron_id,
                'interface_point': None,
                'before_root_ids': before_root_ids, # Use sorted list
                'before_vertex_counts': vertex_counts, # Store counts
                'after_root_ids': after_root_ids,
                'species': visualizer.species
            }

            # Try to find the interface using the already loaded neurons
            try:
                # Ensure we have at least two loaded neurons corresponding to the largest IDs
                if before_root_ids[0] in loaded_neurons_map and before_root_ids[1] in loaded_neurons_map:
                    interface = visualizer.find_neuron_interface(
                        before_root_ids[0],
                        before_root_ids[1]
                    )
                    # Store the interface information
                    edit_info['interface_point'] = interface['interface_point'].tolist() if isinstance(interface['interface_point'], np.ndarray) else interface['interface_point']
                    edit_info['min_distance'] = interface['min_distance']
                else:
                        print(f"Could not find interface for merge {operation_id}: One or both largest neurons failed to load earlier.")

            except Exception as e:
                print(f"Error finding interface for merge operation {operation_id}: {e}")



        elif not is_merge and not merge_only: # Split operation
            print("Found split operation")
            # Get root IDs involved in the split
            before_root_ids = edit_info.get('before_root_ids', [])
            after_root_ids = edit_info.get('after_root_ids', [])

            if not before_root_ids or not after_root_ids or len(after_root_ids) < 2:
                return None

            split_neuron_id = before_root_ids[0]

            # --- Load Neurons Once and Filter/Sort ---
            vertex_counts = None
            loaded_neurons_map = {}
            try:
                # Load all neurons created by the split
                # IMPORTANT: This clears existing neurons in the visualizer
                visualizer.load_neurons(after_root_ids)

                # Get vertex counts directly from loaded neurons
                vertex_counts = {}
                for neuron in visualizer.neurons:
                    neuron_idx = visualizer.neurons.index(neuron)
                    current_neuron_id = visualizer.neuron_ids[neuron_idx]
                    vertex_counts[current_neuron_id] = neuron.vertices.shape[0] if hasattr(neuron, 'vertices') and neuron.vertices is not None else 0
                    loaded_neurons_map[current_neuron_id] = neuron

                # Ensure all requested IDs have a count
                for rid in after_root_ids:
                    if rid not in vertex_counts:
                        vertex_counts[rid] = 0

                # # Check if at least one segment meets the threshold
                # if not all(count >= self.vertices_threshold for count in vertex_counts.values()):
                #     print(f"Skipping split operation {operation_id}: No segment meets vertex threshold {self.vertices_threshold}")
                #     return None

                # Sort after_root_ids by vertex count (descending)
                after_root_ids.sort(key=lambda rid: vertex_counts.get(rid, 0), reverse=True)

            except Exception as e:
                print(f"Error loading neurons or getting vertex counts for split {operation_id}: {e}. Skipping.")
                return None # Skip this edit if loading/counting fails
            # --- End Load/Filter/Sort Logic ---

            # Store the edit information
            edit_info = {
                'neuron_id': neuron_id,
                'timestamp': timestamp,
                'prev_timestamp': prev_timestamp,
                'is_merge': False,
                'operation_id': operation_id,
                'split_neuron_id': split_neuron_id,
                'before_root_ids': before_root_ids,
                'after_root_ids_used': {after_root_id: after_root_id in all_before_root_ids for after_root_id in after_root_ids},
                'after_root_ids': after_root_ids, # Use sorted list
                'after_vertex_counts': vertex_counts, # Store counts
                'interface_point': None,
                'species': visualizer.species
            }

            # Try to find the interface using the already loaded neurons
            try:
                    # Ensure we have at least two loaded neurons corresponding to the largest IDs
                if after_root_ids[0] in loaded_neurons_map and after_root_ids[1] in loaded_neurons_map:
                    interface = visualizer.find_neuron_interface(
                        after_root_ids[0],
                        after_root_ids[1]
                    )
                    edit_info['interface_point'] = interface['interface_point'].tolist() if isinstance(interface['interface_point'], np.ndarray) else interface['interface_point']
                else:
                    print(f"Could not find interface for split {operation_id}: One or both largest neurons failed to load earlier.")

            except Exception as e:
                print(f"Error finding interface for split operation {operation_id}: {e}")

        return edit_info

        

    async def process_neuron_edit_history(self, neuron_id: int, edit_history: pd.DataFrame, split_only: bool = False, merge_only: bool = False, K: int = 50) -> List[Dict[str, Any]]:
        """
        Process the edit history for a given neuron ID.
        
        Args:
            neuron_id: ID of the neuron to process
            
        Returns:
            List of dictionaries containing edit information
        """
        print(f"Processing edit history for neuron {neuron_id}...")
        
        # Get the edit history
        # visualizer = FlyWireVisualizer(output_dir=self.output_dir, species=self.visualizer.species)
        # edit_history = self.visualizer.get_edit_history(neuron_id)
        
        if edit_history is None or len(edit_history) == 0:
            print(f"No edit history found for neuron {neuron_id}")
            return []
        

        # Convert to DataFrame if it's not already
        if isinstance(edit_history, dict):
            # Assuming values are DataFrames, concatenate them
            if all(isinstance(df, pd.DataFrame) for df in edit_history.values()):
                edit_history = pd.concat(edit_history.values(), ignore_index=True)
            else:
                # Handle cases where values are not DataFrames or dict is structured differently
                # This part might need adjustment based on the actual structure of the dict
                print(f"Warning: edit_history is a dict but values are not all DataFrames. Converting dict keys/structure to DataFrame.")
                edit_history = pd.DataFrame(edit_history) 
        elif not isinstance(edit_history, pd.DataFrame):
            # If it's not a dict and not a DataFrame, try converting it
            try:
                edit_history = pd.DataFrame(edit_history)
            except ValueError as e:
                 print(f"Error converting edit_history to DataFrame: {e}")
                 return []
        # Initialize list to store edit information
        edits = []

        # Randomly sample K rows from the edit history
        if len(edit_history) > K:
            edit_history_sampled = edit_history.sample(n=K, random_state=42) # Use a fixed random state for reproducibility
        else:
            edit_history_sampled = edit_history # If fewer than K edits, use all
        
        # Process each sampled edit
        edits = [self.process_neuron_edit(neuron_id, row, split_only=split_only, merge_only=merge_only, edit_history=edit_history) for i, (_, row) in enumerate(edit_history_sampled.iterrows()) ]
        results = await asyncio.gather(*edits)
        edits = [result for result in results if result is not None]



        
        return edits
    
    async def process_neuron_list(self, neuron_ids: List[int], split_only: bool = False, merge_only: bool = False, K: int = 50, save_interval: int = 10) -> List[Dict[str, Any]]:
        """
        Process a list of neuron IDs.
        
        Args:
            neuron_ids: List of neuron IDs to process
            save_interval: Interval at which to save the training data
            
        Returns:
            List of dictionaries containing edit information
        """
        print("Getting edit history")
        edit_history = self.visualizer.get_edit_history(neuron_ids)
        print("Processing edit history")
        edits = await asyncio.gather(*[self.process_neuron_edit_history(neuron_id, edit_history[neuron_id], split_only=split_only, merge_only=merge_only, K=K) for neuron_id in neuron_ids])
        
            
        # Save the training data at intervals
        self.save_training_data(edits)
        print(f"Saved training data after processing {len(neuron_ids)} neurons")
        
        return edits
    
    def save_training_data(self, edits: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Save the training data to a file.
        
        Args:
            edits: List of dictionaries containing edit information
            filename: Optional filename to save to
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            # Create a default filename based on the current time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.json"
        
        # Ensure the filename has the correct extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Create the full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the data
        with open(filepath, 'w') as f:
            json.dump(edits, f, indent=2)
        
        print(f"Saved training data to {filepath}")
        
        return filepath
    
    def load_training_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load training data from a file.
        
        Args:
            filepath: Path to the training data file
            
        Returns:
            List of dictionaries containing edit information
        """
        with open(filepath, 'r') as f:
            edits = json.load(f)
        
        print(f"Loaded training data from {filepath}")
        
        return edits
    
    async def generate_em_data_for_edits(self, edits: List[Dict[str, Any]], window_size_nm: int = 512, window_z: int = 3) -> List[Dict[str, Any]]:
        """
        Generate EM data for each edit.
        
        Args:
            edits: List of dictionaries containing edit information
            window_size: Size of the EM data window
            
        Returns:
            List of dictionaries containing edit information with EM data
        """
        em_data = []

        async def single_edit_em_data(edit: Dict[str, Any], window_size_nm: int = 128, window_z: int = 3):
            # Skip edits without interface points
            if edit.get('interface_point') is None:
                return None
 
            if edit['is_merge']:
                timestamp = edit['prev_timestamp']
            else:
                timestamp = edit['timestamp']
            visualizer = ConnectomeVisualizer(output_dir=self.output_dir, species=self.visualizer.species, timestamp=timestamp)
            visualizer.timestamp = timestamp
            visualizer._connect_to_data_sources()
            print(f"Previous timestamp: {edit['prev_timestamp']}, Current timestamp: {edit['timestamp']}")

            # Get the interface point
            interface_point = edit['interface_point']
            
            window_size = int(window_size_nm)
            window_z += int(edit.get('min_distance', 0)//visualizer.em_resolution[2])
            # Load EM data around the interface point
            try:
                neurons_in_vol = visualizer.load_em_data(
                    interface_point[0], 
                    interface_point[1], 
                    interface_point[2], 
                    window_size_nm=window_size,
                    window_z=window_z,
                )  

                neurons_in_vol = visualizer._process_segmentation_from_api()

                # Get unique root IDs from the matrix
                root_ids_matrix = visualizer.root_ids_grids[visualizer.current_location]
                unique_root_ids = np.unique(root_ids_matrix)
                # Filter out any zero or negative values that might represent background or invalid IDs
                unique_root_ids = unique_root_ids[unique_root_ids > 0]

                
                # Get the EM data
                em_volume = visualizer.vol_em
                # Store the EM data
                edit_with_em = edit.copy()
                edit_with_em['em_data'] = {
                    'shape': em_volume.shape,
                    'all_unique_root_ids': unique_root_ids.tolist(),
                    'location': visualizer.current_location,
                    'neurons_in_vol': neurons_in_vol,
                    'valid_segment_vertices_threshold': self.valid_segment_vertices_threshold
                }
            except Exception as e:
                print(f"Error loading EM data for edit {edit.get('operation_id')}: {e}")
                return None
            return edit_with_em
        
        results = await asyncio.gather(*[single_edit_em_data(edit) for edit in edits])
        em_data = [result for result in results if result is not None]

        all_unique_neuron_ids = [x['em_data']['all_unique_root_ids'] for x in em_data]
        all_unique_neuron_ids = [x for y in all_unique_neuron_ids for x in y]


        # print("Loading all of the unique neurons")
        # all_unique_neurons = await self.visualizer.load_neurons_parallel(all_unique_neuron_ids, timeout=5*60.0)
        # for x in em_data:
        #     x['unique_root_ids'] = [int(root_id) for root_id in x['em_data']['all_unique_root_ids'] if (all_unique_neurons[root_id] is not None) and (len(all_unique_neurons[root_id].vertices) > self.valid_segment_vertices_threshold)]


        return em_data
    
    def save_em_data(self, em_data: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Save the EM data to a file.
        
        Args:
            em_data: List of dictionaries containing edit information with EM data
            filename: Optional filename to save to
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            # Create a default filename based on the current time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"em_data_{timestamp}.json"
        
        # Ensure the filename has the correct extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Create the full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the data
        with open(filepath, 'w') as f:
            json.dump(em_data, f, indent=2)
        
        print(f"Saved EM data to {filepath}")
        
        return filepath


# Example usage
if __name__ == "__main__":
    try:
        species = "mouse"
        num_neurons = 200
        if species == "fly":
            client = caveclient.CAVEclient("flywire_fafb_public")
            neuron_ids = list(client.materialize.query_table('proofread_neurons')['pt_root_id'])[:num_neurons]
        elif species == "mouse":
            client = caveclient.CAVEclient("minnie65_public")
            neuron_ids = list(client.materialize.query_table('proofreading_status_and_strategy')['valid_id'])
            # random.seed(42)
            random.seed(84)
            neuron_ids = random.sample(neuron_ids, num_neurons)
        elif species == "human":
            url = "https://global.brain-wire-test.org/"
            client = caveclient.CAVEclient(datastack_name='h01_c3_flat', server_address=url)
            neuron_ids = list(client.materialize.query_table('proofreading_status_test')['pt_root_id'])[:num_neurons]
        elif species == "fish":
            url = "https://global.brain-wire-test.org/"


        gatherer = TrainingDataGatherer(output_dir="./training_data", species=species)

        
        # neuron_ids = [
        #     864691135441799752
        # ]
        
        # # Process the neurons
        edits = asyncio.run(gatherer.process_neuron_list(neuron_ids, split_only=True, K = 1000))
        
        # edits = asyncio.run(gatherer.process_neuron_list(neuron_ids, split_only=False, merge_only=True, K = 200))

        edits_flat = [x for y in edits for x in y]
        breakpoint()
        
        edits_flat = [x for x in edits_flat if x['interface_point'] is not None]

        # Generate EM data for the edits
        em_data = asyncio.run(gatherer.generate_em_data_for_edits(edits_flat))
        
        # Save the EM data
        gatherer.save_em_data(em_data) 
    except:
        import pdb; pdb.post_mortem()