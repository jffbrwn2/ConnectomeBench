#!/usr/bin/env python3
"""
Function to load the first N entries from merge_error_only_mouse.json
and visualize them using ConnectomeVisualizer.
"""

import json
import os
import asyncio
from typing import List, Dict, Any
from connectome_visualizer import ConnectomeVisualizer


def load_and_plot_segments(
    json_path: str = "/Users/jbrown/Documents/boyden_lab/ai-proofreading/connectomebench/scripts/training_data/merge_error_only_mouse.json",
    num_entries: int = 5,
    output_dir: str = "./segment_visualizations",
    save_images: bool = True,
    use_parallel_loading: bool = True
) -> List[Dict[str, Any]]:
    """
    Load the first N entries from the merge error JSON file and plot the 3D meshes using ConnectomeVisualizer.

    Args:
        json_path: Path to the JSON file containing merge error data
        num_entries: Number of entries to load and visualize (default: 5)
        output_dir: Directory to save visualization outputs
        save_images: Whether to save visualization images (default: True)
        use_parallel_loading: Whether to use parallel loading for neurons (faster, default: True)

    Returns:
        List of dictionaries containing visualization information for each entry
    """
    # Load the JSON file
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Take only the first num_entries
    entries = data[:num_entries]

    print(f"Loaded {len(entries)} entries from {json_path}\n")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Process each entry
    for i, entry in enumerate(entries):
        print(f"\n{'='*80}")
        print(f"Processing entry {i+1}/{len(entries)}")
        print(f"{'='*80}")

        # Extract relevant information
        neuron_id = entry.get('neuron_id')
        timestamp = entry.get('timestamp')
        prev_timestamp = entry.get('prev_timestamp')
        is_merge = entry.get('is_merge', False)
        species = entry.get('species', 'mouse')
        interface_point = entry.get('interface_point')
        after_root_ids = entry.get('after_root_ids', [])
        before_root_ids = entry.get('before_root_ids', [])
        split_neuron_id = entry.get('split_neuron_id')

        print(f"Neuron ID: {neuron_id}")
        print(f"Timestamp: {timestamp}")
        print(f"Previous Timestamp: {prev_timestamp}")
        print(f"Is Merge: {is_merge}")
        print(f"Species: {species}")
        print(f"Interface Point: {interface_point}")
        print(f"Split Neuron ID: {split_neuron_id}")
        print(f"Before Root IDs: {before_root_ids}")
        print(f"After Root IDs: {after_root_ids}")

        # Create visualizer for this entry
        entry_output_dir = os.path.join(output_dir, f"entry_{i+1}_neuron_{neuron_id}")
        os.makedirs(entry_output_dir, exist_ok=True)

        # Initialize visualizer with the appropriate timestamp and species
        visualizer = ConnectomeVisualizer(
            species=species,
            output_dir=entry_output_dir,
            timestamp=timestamp
        )

        # Determine which neuron IDs to visualize
        # For split errors (is_merge=False), we want to see the segments after the split
        neuron_ids_to_visualize = after_root_ids if after_root_ids else before_root_ids

        if not neuron_ids_to_visualize:
            print(f"Warning: No neuron IDs found for entry {i+1}. Skipping.")
            continue

        print(f"Loading neurons: {neuron_ids_to_visualize}")

        try:
            # Load neurons - use parallel loading if requested
            if use_parallel_loading and len(neuron_ids_to_visualize) > 1:
                print("Using parallel loading for faster performance...")
                neurons_map = asyncio.run(visualizer.load_neurons_parallel(
                    neuron_ids_to_visualize,
                    timeout=30.0,
                    batch_size=min(10, len(neuron_ids_to_visualize))
                ))

                # Update visualizer state with loaded neurons
                visualizer.neurons = []
                visualizer.neuron_ids = []
                for nid in neuron_ids_to_visualize:
                    if nid in neurons_map and neurons_map[nid] is not None:
                        visualizer.neurons.append(neurons_map[nid])
                        visualizer.neuron_ids.append(nid)
                        # Assign color
                        if nid not in visualizer.neuron_color_map:
                            color = visualizer.NEURON_COLORS[visualizer.color_idx % len(visualizer.NEURON_COLORS)]
                            visualizer.color_idx += 1
                            visualizer.neuron_color_map[nid] = color

                print(f"Successfully loaded {len(visualizer.neurons)}/{len(neuron_ids_to_visualize)} neurons")
            else:
                visualizer.load_neurons(neuron_ids_to_visualize)

            # Create and save 3D views (no EM data loading needed)
            if save_images:
                print("\nCreating 3D neuron views...")
                view_paths = visualizer.save_3d_views(
                    base_filename="neuron_mesh",
                    width=512,
                    height=512
                )

                results.append({
                    'entry_index': i + 1,
                    'neuron_id': neuron_id,
                    'timestamp': timestamp,
                    'species': species,
                    'is_merge': is_merge,
                    'split_neuron_id': split_neuron_id,
                    'after_root_ids': after_root_ids,
                    'before_root_ids': before_root_ids,
                    'interface_point': interface_point,
                    'view_paths': view_paths,
                    'output_dir': entry_output_dir
                })
            else:
                # Just store metadata without saving
                results.append({
                    'entry_index': i + 1,
                    'neuron_id': neuron_id,
                    'timestamp': timestamp,
                    'species': species,
                    'is_merge': is_merge,
                    'split_neuron_id': split_neuron_id,
                    'after_root_ids': after_root_ids,
                    'before_root_ids': before_root_ids,
                    'interface_point': interface_point,
                    'output_dir': entry_output_dir
                })

        except Exception as e:
            print(f"Error processing entry {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"Completed processing {len(results)}/{len(entries)} entries")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of the visualization results."""
    print("\n" + "="*80)
    print("SUMMARY OF VISUALIZED ENTRIES")
    print("="*80)

    for result in results:
        print(f"\nEntry {result['entry_index']}:")
        print(f"  Neuron ID: {result['neuron_id']}")
        print(f"  Species: {result['species']}")
        print(f"  Is Merge: {result['is_merge']}")
        print(f"  Timestamp: {result['timestamp']}")
        print(f"  Split Neuron ID: {result.get('split_neuron_id', 'N/A')}")
        print(f"  Before Root IDs: {result['before_root_ids']}")
        print(f"  After Root IDs: {result['after_root_ids']}")
        print(f"  Output Directory: {result['output_dir']}")

        if 'view_paths' in result:
            print(f"  3D Views:")
            for view_name, view_path in result['view_paths'].items():
                print(f"    - {view_name}: {view_path}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage
    print("Loading and plotting first 5 segments from merge_error_only_mouse.json...\n")

    results = load_and_plot_segments(
        num_entries=5,
        save_images=True,
        use_parallel_loading=True
    )

    # Print summary
    print_summary(results)
