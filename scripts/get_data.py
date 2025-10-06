import argparse
import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import caveclient
import numpy as np
import pandas as pd

from connectome_visualizer import ConnectomeVisualizer

class TrainingDataGatherer:
    """
    A class for gathering training data at scale from FlyWire edit histories.
    
    This class processes edit histories to identify merge and split error corrections,
    and finds the locations of these edits using the neuron interface method.
    """
    
    def __init__(
        self,
        output_dir: str = "./training_data",
        species: str = "fly",
        vertices_threshold: int = 1000,
        valid_segment_vertices_threshold: int = 1000,
        logger: Optional[logging.Logger] = None,
    ):
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
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
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
        except Exception:
            self.logger.exception("Error subtracting time from timestamp")
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
            self.logger.warning(
                "Skipping edit for neuron %s: unsupported timestamp %s", neuron_id, edit_info.get("timestamp")
            )
            return None
        
        try:
            prev_timestamp = timestamp - 1
        except Exception:
            self.logger.exception("Error calculating previous timestamp")
            prev_timestamp = timestamp  # Fallback to current timestamp if error

        all_before_root_ids = edit_history['before_root_ids']
        all_before_root_ids = [x for y in all_before_root_ids for x in y]
        all_after_root_ids = edit_history['after_root_ids']
        all_after_root_ids = [x for y in all_after_root_ids for x in y]


        is_merge = edit_info.get('is_merge', False)
        if is_merge and split_only:
            self.logger.debug("Skipping merge operation because split_only is True")
            return None
        if not is_merge and merge_only:
            self.logger.debug("Skipping split operation because merge_only is True")
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
                    self.logger.debug(
                        "Skipping merge operation %s: no segment meets vertex threshold %s",
                        operation_id,
                        self.vertices_threshold,
                    )
                    return None

                # Sort before_root_ids by vertex count (descending) using the obtained counts
                before_root_ids.sort(key=lambda rid: vertex_counts.get(rid, 0), reverse=True)

            except Exception as e:
                self.logger.warning(
                    "Error loading neurons for merge %s: %s", operation_id, e, exc_info=True
                )
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
                    self.logger.debug(
                        "Could not find interface for merge %s: one or both largest neurons failed to load earlier",
                        operation_id,
                    )

            except Exception as e:
                self.logger.warning(
                    "Error finding interface for merge operation %s: %s", operation_id, e, exc_info=True
                )



        elif not is_merge and not merge_only: # Split operation
            self.logger.debug("Found split operation")
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
                self.logger.warning(
                    "Error loading neurons for split %s: %s", operation_id, e, exc_info=True
                )
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
                    self.logger.debug(
                        "Could not find interface for split %s: one or both largest neurons failed to load earlier",
                        operation_id,
                    )

            except Exception as e:
                self.logger.warning(
                    "Error finding interface for split operation %s: %s", operation_id, e, exc_info=True
                )

        return edit_info

        

    async def process_neuron_edit_history(self, neuron_id: int, edit_history: pd.DataFrame, split_only: bool = False, merge_only: bool = False, K: int = 50) -> List[Dict[str, Any]]:
        """
        Process the edit history for a given neuron ID.
        
        Args:
            neuron_id: ID of the neuron to process
            
        Returns:
            List of dictionaries containing edit information
        """
        self.logger.info("Processing edit history for neuron %s", neuron_id)
        
        # Get the edit history
        # visualizer = FlyWireVisualizer(output_dir=self.output_dir, species=self.visualizer.species)
        # edit_history = self.visualizer.get_edit_history(neuron_id)
        
        if edit_history is None or len(edit_history) == 0:
            self.logger.debug("No edit history found for neuron %s", neuron_id)
            return []
        

        # Convert to DataFrame if it's not already
        if isinstance(edit_history, dict):
            # Assuming values are DataFrames, concatenate them
            if all(isinstance(df, pd.DataFrame) for df in edit_history.values()):
                edit_history = pd.concat(edit_history.values(), ignore_index=True)
            else:
                # Handle cases where values are not DataFrames or dict is structured differently
                # This part might need adjustment based on the actual structure of the dict
                self.logger.warning(
                    "edit_history for neuron %s is a dict with non-DataFrame values; attempting conversion",
                    neuron_id,
                )
                edit_history = pd.DataFrame(edit_history) 
        elif not isinstance(edit_history, pd.DataFrame):
            # If it's not a dict and not a DataFrame, try converting it
            try:
                edit_history = pd.DataFrame(edit_history)
            except ValueError as e:
                 self.logger.error("Error converting edit_history to DataFrame for neuron %s: %s", neuron_id, e)
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
    
    async def process_neuron_list(
        self,
        neuron_ids: List[int],
        split_only: bool = False,
        merge_only: bool = False,
        K: int = 50,
        save_interval: int = 10,
        save: bool = True,
        filename: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Process a list of neuron IDs.
        
        Args:
            neuron_ids: List of neuron IDs to process
            save_interval: Interval at which to save the training data
            
        Returns:
            List of dictionaries containing edit information
        """
        self.logger.info("Fetching edit history for %s neurons", len(neuron_ids))
        edit_history = self.visualizer.get_edit_history(neuron_ids)
        self.logger.info("Processing edit history")
        edits_nested = await asyncio.gather(*[
            self.process_neuron_edit_history(
                neuron_id,
                edit_history[neuron_id],
                split_only=split_only,
                merge_only=merge_only,
                K=K,
            )
            for neuron_id in neuron_ids
        ])
        
        training_file: Optional[str] = None
        if save:
            training_file = self.save_training_data(edits_nested, filename=filename)
            self.logger.info("Saved training data after processing %s neurons", len(neuron_ids))

        flattened: List[Dict[str, Any]] = [
            item for sublist in edits_nested for item in (sublist or []) if item is not None
        ]
        return flattened, training_file
    
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
        
        # Flatten nested lists (one list per neuron) before saving
        flattened: List[Dict[str, Any]] = []
        for entry in edits:
            if isinstance(entry, list):
                flattened.extend(item for item in entry if item is not None)
            elif isinstance(entry, dict):
                flattened.append(entry)
            else:
                self.logger.debug("Skipping unexpected entry of type %s while saving", type(entry))

        # Save the data
        with open(filepath, 'w') as f:
            json.dump(flattened, f, indent=2)

        self.logger.info("Saved %s training edits to %s", len(flattened), filepath)

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
        
        self.logger.info("Loaded training data from %s", filepath)
        
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
                self.logger.warning(
                    "Error loading EM data for edit %s: %s", edit.get("operation_id"), e, exc_info=True
                )
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
        
        self.logger.info("Saved EM data to %s", filepath)
        
        return filepath


def _load_neuron_ids_from_file(path: Path) -> List[int]:
    """Load neuron IDs from a JSON list or newline-delimited text file."""

    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Neuron ID file {path} is empty")

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [int(x) for x in data]
    except json.JSONDecodeError:
        pass

    ids: List[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(int(line))
    return ids


def _fetch_neuron_ids(species: str, num_neurons: int, seed: int) -> List[int]:
    """Fetch candidate neuron IDs for the requested species."""

    if species == "mouse":
        client = caveclient.CAVEclient("minnie65_public")
        ids = list(client.materialize.query_table('proofreading_status_and_strategy')['valid_id'])
    elif species == "fly":
        client = caveclient.CAVEclient("flywire_fafb_public")
        ids = list(client.materialize.query_table('proofread_neurons')['pt_root_id'])
    elif species == "human":
        client = caveclient.CAVEclient(datastack_name='h01_c3_flat', server_address="https://global.brain-wire-test.org/")
        ids = list(client.materialize.query_table('proofreading_status_test')['pt_root_id'])
    elif species == "fish":
        client = caveclient.CAVEclient(datastack_name='zebrafish_flat', server_address="https://global.brain-wire-test.org/")
        ids = list(client.materialize.query_table('proofreading_status_test')['pt_root_id'])
    else:
        raise ValueError(f"Unsupported species: {species}")

    if not ids:
        raise RuntimeError(f"No neuron IDs available for species '{species}'")

    if num_neurons <= 0 or num_neurons >= len(ids):
        return ids

    random.seed(seed)
    return random.sample(ids, num_neurons)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gather split/merge training data from CAVE edit histories."
    )
    parser.add_argument("--species", choices=["mouse", "fly", "human", "fish"], default="mouse")
    parser.add_argument("--output-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--output-filename", type=str, help="Optional override for the training data JSON filename.")
    parser.add_argument("--em-output-filename", type=str, help="Optional override when saving EM data JSON.")
    parser.add_argument("--num-neurons", type=int, default=50, help="Number of neurons to sample when neuron IDs are not provided.")
    parser.add_argument("--neuron-ids-file", type=Path, help="Path to a newline-delimited or JSON list of neuron IDs to process.")
    parser.add_argument("--split-only", action="store_true", help="Collect only split operations.")
    parser.add_argument("--merge-only", action="store_true", help="Collect only merge operations.")
    parser.add_argument("--edits-per-neuron", type=int, default=50, help="Maximum edit events sampled per neuron.")
    parser.add_argument("--vertices-threshold", type=int, default=1000, help="Minimum vertex count for segments considered in edits.")
    parser.add_argument("--valid-segment-threshold", type=int, default=1000, help="Minimum vertices when filtering EM segmentation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for neuron sampling.")
    parser.add_argument("--extract-em-volumes", action="store_true", help="Extract EM volumes around edit interface points.")
    parser.add_argument("--window-size-nm", type=int, default=512, help="XY window size in nanometres for EM extraction.")
    parser.add_argument("--window-z", type=int, default=3, help="Z slices for EM extraction.")
    parser.add_argument("--no-save", action="store_true", help="Do not emit the training data JSON (useful for dry runs).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split_only and args.merge_only:
        raise SystemExit("--split-only and --merge-only cannot both be set")

    _configure_logging(args.verbose)
    logger = logging.getLogger("get_data")

    if args.neuron_ids_file:
        neuron_ids = _load_neuron_ids_from_file(args.neuron_ids_file)
        logger.info("Loaded %s neuron IDs from %s", len(neuron_ids), args.neuron_ids_file)
    else:
        neuron_ids = _fetch_neuron_ids(args.species, args.num_neurons, args.seed)
        logger.info("Sampled %s neuron IDs for species %s", len(neuron_ids), args.species)

    gatherer = TrainingDataGatherer(
        output_dir=str(args.output_dir),
        species=args.species,
        vertices_threshold=args.vertices_threshold,
        valid_segment_vertices_threshold=args.valid_segment_threshold,
        logger=logging.getLogger("TrainingDataGatherer"),
    )

    training_filename = args.output_filename
    if training_filename is not None and not training_filename.endswith(".json"):
        training_filename = f"{training_filename}.json"

    edits_flat, training_path = asyncio.run(
        gatherer.process_neuron_list(
            neuron_ids,
            split_only=args.split_only,
            merge_only=args.merge_only,
            K=args.edits_per_neuron,
            save=not args.no_save,
            filename=training_filename,
        )
    )

    if args.no_save:
        logger.info("Processed %s edits (no training data written)", len(edits_flat))
    else:
        logger.info(
            "Wrote %s edits to %s",
            len(edits_flat),
            training_path,
        )

    if args.extract_em_volumes:
        if not edits_flat:
            logger.warning("No edits available for EM generation; skipping")
        else:
            em_data = asyncio.run(
                gatherer.generate_em_data_for_edits(
                    edits_flat,
                    window_size_nm=args.window_size_nm,
                    window_z=args.window_z,
                )
            )
            em_filename = args.em_output_filename
            if em_filename is not None and not em_filename.endswith(".json"):
                em_filename = f"{em_filename}.json"
            em_path = gatherer.save_em_data(em_data, filename=em_filename)
            logger.info("Saved EM volumes with %s entries to %s", len(em_data), em_path)


if __name__ == "__main__":
    main()
