import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import argparse
import random
import multiprocessing
import logging
import asyncio

from cloudvolume import Bbox
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from connectome_visualizer import ConnectomeVisualizer
from util import LLMProcessor, evaluate_response, create_unified_result_structure
from prompts import create_split_identification_prompt, create_split_comparison_prompt

logging.basicConfig(level=logging.INFO)


def generate_neuron_images(
    base_neuron_ids: List[str],
    bbox_neuron_ids: List[str],
    merge_coords: List[float],
    output_dir: str,
    timestamp: Optional[int] = None,
    species: str = "fly",
    zoom_margin: int = 5000,
    model: str = "gpt-4o-mini"
) -> Dict[str, Dict[str, Any]]:
    """
    Generate images for a list of base neurons near merge coordinates.

    Args:
        base_neuron_ids: List of string IDs for the primary neurons
        bbox_neuron_ids: List of neuron IDs to compute bounding box
        merge_coords: Coordinates near the neurons of interest [x, y, z]
        output_dir: Directory to save images
        timestamp: Optional timestamp for CAVEclient state
        species: Species for the visualizer ('fly' or 'h01')

    Returns:
        Dictionary mapping each base_neuron_id (str) to its dictionary of generated image paths.
    """

    # Use coordinates directly (same for all neurons in the list)
    merge_x, merge_y, merge_z = merge_coords
    merge_x_nm = merge_x
    merge_y_nm = merge_y
    merge_z_nm = merge_z

    coords_suffix = f"{int(merge_coords[0])}_{int(merge_coords[1])}_{int(merge_coords[2])}"
    bbox_visualizer = ConnectomeVisualizer(dataset="public", timestamp=timestamp, species=species)
    bbox_visualizer.load_neurons([int(x) for x in bbox_neuron_ids])
    segment_dimensions = []
    for bbox_neuron in bbox_visualizer.neurons:
        max_bbox_dimension = np.max(np.max(bbox_neuron.vertices, axis =0)-np.min(bbox_neuron.vertices, axis = 0))
        segment_dimensions.append(max_bbox_dimension)
    zoom_margin = int(max([4096, 2* np.min(segment_dimensions)]))

    # Create output directory specific to this base neuron and coords
    # Use a different naming scheme to avoid collision with option-based dirs
    base_neuron_ids_str = "_".join([str(id_) for id_ in sorted(base_neuron_ids)])
    neuron_dir = os.path.join(output_dir, f"base_{base_neuron_ids_str}_{coords_suffix}")
    os.makedirs(neuron_dir, exist_ok=True)


    # Initialize visualizer for each neuron to ensure clean state? Or reuse?
    # Reusing might be faster but could have side effects. Let's re-initialize for safety.
    visualizer = ConnectomeVisualizer(output_dir=neuron_dir, dataset="public", timestamp=timestamp, species=species)
    image_paths = {}

    for base_neuron_id in base_neuron_ids: # Loop through the list of IDs
        print(f"--- Processing images for base neuron {base_neuron_id} ---")

        current_neuron_image_paths = {'default': {}, 'zoomed': {}, 'em': None}

        # Define expected filenames for the current base neuron
        zoomed_base_filename = f"base_{base_neuron_id}_zoomed"
        expected_zoomed_paths = {
            view: os.path.join(neuron_dir, f"{zoomed_base_filename}_{view}.png")
            for view in ['front', 'side', 'top'] # Standard views
        }
        base_em_filename = f"base_{base_neuron_id}_em_slice_with_segmentation.png"
        expected_em_path = os.path.join(neuron_dir, base_em_filename)

        # Check if zoomed images already exist
        zoomed_exist = all(os.path.exists(p) for p in expected_zoomed_paths.values())

        if zoomed_exist:
            print(f"Found existing zoomed images for base neuron {base_neuron_id}. Skipping generation.")
            current_neuron_image_paths['zoomed'] = expected_zoomed_paths
        else:
            print(f"Generating zoomed images for base neuron {base_neuron_id}...")
            visualizer.clear_neurons() # Ensure clean start
            try:
                visualizer.load_neurons([int(base_neuron_id)])
                # Optionally load EM data if needed for context, even without segmentation overlay
                # if visualizer.vol_em is not None:
                #     visualizer.load_em_data(merge_x_nm, merge_y_nm, merge_z_nm) # Load EM around the point

            except Exception as e:
                print(f"ERROR: Failed setup for base neuron {base_neuron_id}: {e}. Skipping image generation for this neuron.")
                # Skip to the next neuron in the list if setup fails
                image_paths[base_neuron_id] = current_neuron_image_paths # Store partial/empty paths
                continue # Go to next base_neuron_id

            # Save zoomed views
            try:

                bbox = Bbox((merge_x_nm - zoom_margin, merge_y_nm - zoom_margin, merge_z_nm - zoom_margin),
                            (merge_x_nm + zoom_margin, merge_y_nm + zoom_margin, merge_z_nm + zoom_margin), unit="nm")

                visualizer.create_3d_neuron_figure(bbox=bbox, add_em_slice=False) # Only show the neuron mesh
                save_3d_views_result = visualizer.save_3d_views(bbox=bbox, base_filename=zoomed_base_filename, crop=True) # Uses visualizer.output_dir

                if save_3d_views_result is None:
                    print(f"WARNING: Zoomed image generation potentially failed or timed out for base neuron {base_neuron_id}.")
                else:
                    # Verify expected files were created and store paths
                    saved_paths = {}
                    all_saved = True
                    for view, expected_path in expected_zoomed_paths.items():
                        if os.path.exists(expected_path):
                            saved_paths[view] = expected_path
                        else:
                            print(f"Warning: Expected zoomed view file missing after save: {expected_path}")
                            all_saved = False
                    if all_saved:
                        current_neuron_image_paths['zoomed'] = saved_paths
                    else:
                        print(f"Warning: Not all expected zoomed views were saved for base neuron {base_neuron_id}.")
            except Exception as e:
                logging.error(f"ERROR: Failed to save zoomed images for base neuron {base_neuron_id}: {str(e)}")

        # Check and generate EM segmentation slice (only for the base neuron)
        base_em_path = None
        if visualizer.vol_em is not None:
            if os.path.exists(expected_em_path):
                print(f"Found existing EM slice for base neuron {base_neuron_id}. Skipping generation.")
                base_em_path = expected_em_path
            else:
                print(f"Generating EM slice for base neuron {base_neuron_id}...")
                # Ensure base neuron is loaded if we skipped zoomed generation
                if zoomed_exist:
                    visualizer.clear_neurons()
                    try:
                        visualizer.load_neurons([int(base_neuron_id)])
                        # EM data should still be loaded
                    except Exception as e:
                        print(f"ERROR: Failed setup for base neuron {base_neuron_id} EM slice: {e}. Skipping EM generation.")

                # Generate EM slice only if setup succeeded and neurons are loaded
                if visualizer.neurons:
                    try:
                        # Make sure EM data is loaded around the desired coords
                        visualizer.load_em_data(merge_x_nm, merge_y_nm, merge_z_nm)
                        save_em_segmentation_result = visualizer.save_em_segmentation(filename=base_em_filename) # Uses visualizer.output_dir
                        if save_em_segmentation_result is None:
                            print(f"WARNING: Base neuron EM segmentation generation potentially failed or timed out.")
                        elif os.path.exists(expected_em_path):
                            base_em_path = expected_em_path
                        else:
                             print(f"Warning: Expected EM slice file missing after save: {expected_em_path}")
                    except Exception as e:
                        print(f"ERROR: Failed to save base neuron EM segmentation: {str(e)}")
        else:
            print(f"Skipping EM slice generation for base neuron {base_neuron_id} as EM volume was not loaded.")

        current_neuron_image_paths['em'] = base_em_path
        current_neuron_image_paths['zoom_margin'] = zoom_margin

        image_paths[base_neuron_id] = current_neuron_image_paths
    # Save metadata JSON (specific to this neuron)
    metadata = {
        'neuron_ids': base_neuron_ids,
        'coords_used': merge_coords, # Note the coords used for centering/EM
        'timestamp': timestamp,
        'image_paths': image_paths, # Paths for this specific neuron
        'model': model, # Keep model info if relevant for context
        'zoom_margin': zoom_margin
    }
    metadata_path = os.path.join(neuron_dir, "generation_metadata.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved generation metadata to {metadata_path}")
    except Exception as e:
        print(f"ERROR: Failed to save generation metadata to {metadata_path}: {e}")

    # Return paths to the generated images for all base neurons
    return image_paths


def _process_single_split_event(item, output_dir, task, force_regenerate, use_zoomed_images, views, species, zoom_margin, model):
    """Processes a single split event: generates images and evaluates options."""
    operation_id = item.get('operation_id', 'N/A')
    try:
        base_neuron_id = str(item['before_root_ids'][0]) # Use the first pre-merge ID as base (ensure string)
        first_split_neuron_id = str(item['after_root_ids'][0])
        other_split_neuron_id = str(item['after_root_ids'][1])
        final_neuron_id = str(item['neuron_id'])
        merge_coords = item['interface_point']


        # Timestamp just before the merge occurred
        timestamp_after_split = item.get('timestamp')

        # Define neuron directory and metadata path
        coords_suffix = f"{int(merge_coords[0])}_{int(merge_coords[1])}_{int(merge_coords[2])}"
        first_neuron_dir = os.path.join(output_dir, f"base_{base_neuron_id}_{coords_suffix}")
        first_neuron_metadata_path = os.path.join(first_neuron_dir, "generation_metadata.json")
        last_neuron_dir = os.path.join(output_dir, f"base_{final_neuron_id}_{coords_suffix}")
        last_neuron_metadata_path = os.path.join(last_neuron_dir, "generation_metadata.json")

        # Try to load existing metadata and image paths
        image_paths = None
        if not force_regenerate and os.path.exists(first_neuron_metadata_path) and os.path.exists(last_neuron_metadata_path):
            image_paths = {}
            try:
                with open(first_neuron_metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Check if the parameters match
                if (
                    metadata.get('base_neuron_id') == base_neuron_id
                    and metadata.get('coords_used') == merge_coords
                    and metadata.get('timestamp') == timestamp_after_split
                ):
                    print(f"Found matching metadata for operation {operation_id}. Loading existing image paths.")
                    image_paths[base_neuron_id] = metadata.get('image_paths')
                else:
                    print(f"Metadata found but parameters mismatch for operation {operation_id}. Will regenerate images.")

                with open(last_neuron_metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Check if the parameters match
                if (
                    metadata.get('base_neuron_id') == final_neuron_id
                    and metadata.get('coords_used') == merge_coords
                    and metadata.get('timestamp') == timestamp_after_split
                ):
                    print(f"Found matching metadata for operation {operation_id}. Loading existing image paths.")
                    image_paths[final_neuron_id] = metadata.get('image_paths')
                else:
                    print(f"Metadata found but parameters mismatch for operation {operation_id}. Will regenerate images.")
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"Error reading metadata file {first_neuron_metadata_path} or {last_neuron_metadata_path}: {e}. Will regenerate images.")


        # Image generation (only if needed)
        if force_regenerate or (not os.path.exists(first_neuron_metadata_path)) or (not os.path.exists(last_neuron_metadata_path)) or (image_paths is None):
            image_paths = generate_neuron_images(
                [base_neuron_id, final_neuron_id], # Pass string
                [first_split_neuron_id, other_split_neuron_id],
                merge_coords,
                output_dir,
                timestamp=timestamp_after_split,
                species=item.get('species', 'fly') # Pass species
            )


        # Check if essential base image was generated/found (critical after generation or loading)
        if not image_paths: # Check if image_paths is None (generation failed or metadata load failed)
            print(f"Error: Image paths unavailable for merge op {operation_id}. Skipping evaluation.")
            return None

        image_set_key = 'zoomed' if use_zoomed_images else 'default'

        # Prepare options for the prompt (ID and image path)
        prompt_options = []
        option_index_to_id = {} # Will map index (int) to option ID (string)
        current_index = 1
        options_with_paths = image_paths # Keys are string IDs

        # for opt_id in [base_neuron_id, first_split_neuron_id, other_split_neuron_id]: # option_ids is list of strings
        for opt_id in [base_neuron_id, final_neuron_id]: # option_ids is list of strings
            option_paths_dict = options_with_paths.get(opt_id) # Get using string ID

            # Check if images were generated/found for this option
            if option_paths_dict and ((option_paths_dict.get('default') or option_paths_dict.get('zoomed'))):
                 # Basic check: Ensure the required image set (zoomed/default) and front view exist
                 img_check_path = option_paths_dict.get(image_set_key, {}).get('front')
                 if img_check_path and os.path.exists(img_check_path):
                    prompt_options.append({
                        'id': opt_id, # String ID
                        'paths': option_paths_dict,
                        "merge_coords": coords_suffix,
                        "zoom_margin": option_paths_dict.get('zoom_margin', zoom_margin)
                    })
                    option_index_to_id[current_index] = opt_id # Value is string ID
                    current_index += 1
                 else:
                    print(f"Warning: Required image ({image_set_key}/front) not found at {img_check_path} for option {opt_id}. Excluding from prompt.")
            else:
                print(f"Warning: Image paths dictionary for option {opt_id} not found or generation failed. Excluding from prompt.")


        if not prompt_options:
            print(f"Warning: No valid option images found for merge op {operation_id}. Skipping evaluation.")
            return None
        return {
            'operation_id': operation_id,
            'base_neuron_id': base_neuron_id, # String
            'use_zoomed_images': use_zoomed_images,
            'image_paths': image_paths,
            'prompt_options': prompt_options,
            'views': views,
            'before_root_ids': item['before_root_ids'],
            'after_root_ids': item['after_root_ids'],
            'proofread_root_id': final_neuron_id,
            'merge_coords': merge_coords,
            'interface_point': item.get('interface_point', None),
            'timestamp': timestamp_after_split
        }
    except Exception as e:
        print(f"Error processing split event for operation {operation_id}: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging worker errors
        return None # Indicate failure for this item


async def process_split_data(json_path: str, output_dir: str, force_regenerate=False, num_samples: Optional[int] = None, use_zoomed_images=True, max_workers: Optional[int] = None, views=['front', 'side', 'top'], task='split_comparison', prompt_mode = 'informative',llm_processor: LLMProcessor = None, zoom_margin: int = 5000, species: str = "fly", model: str = "gpt-4o-mini", K: int = 10):
    """Process split data from JSON and evaluate options in parallel using multiprocessing."""
    try:
        with open(json_path, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {json_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return pd.DataFrame()

    # Filter for split operations with necessary data
    split_data = [
        item for item in all_data
        if item.get('is_merge') is False
        and item.get('after_root_ids')
        and len(item['after_root_ids']) >= 2
        and item.get('interface_point')
        and item.get('em_data')
        and item['em_data'].get('all_unique_root_ids')
        and np.any(list(item['after_root_ids_used'].values()))
    ]

    total_events_found = len(split_data)
    print(f"Found {total_events_found} split events in the input file.")

    # Limit number of samples if specified
    if num_samples is not None and num_samples > 0:
        if num_samples < total_events_found:
            print(f"Processing only the first {num_samples} split events.")
            split_data = split_data[:num_samples]
        else:
            print(f"Requested {num_samples} samples, but only {total_events_found} available. Processing all available.")
    else:
         print(f"Processing all {total_events_found} split events.")

    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine number of workers (processes for multiprocessing)
    if max_workers is None:
        max_workers = os.cpu_count() or 4 # Default to CPU count or 4

    print(f"Using up to {max_workers} processes for parallel processing.")

    # Prepare arguments for starmap
    args_list = [
        (item, output_dir, task, force_regenerate, use_zoomed_images, views, species, zoom_margin, model)
        for item in split_data
    ]

    # Use multiprocessing.Pool for parallel execution
    results_raw = []
    print(f"Running image generation/option preparation in parallel for {len(args_list)} events...")
    try:
        # The parallel part (_process_single_split_event) generates images and prepares data
        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
            results_raw = pool.starmap(_process_single_split_event, args_list)

    except Exception as e:
         # Catch potential errors during pool creation or execution
         print(f"An error occurred during parallel processing: {e}")
         import traceback
         traceback.print_exc()

    # Filter out None results from parallel processing (indicating errors in individual tasks)
    processed_events = [res for res in results_raw if res is not None]
    if len(processed_events) < len(results_raw):
        print(f"Warning: {len(results_raw) - len(processed_events)} split events failed during image/option processing.")

    prompts = []
    indices = []
    correct_answers = []
    total_options_processed = 0  # Add counter for total options processed

    if task == 'split_identification':
        for event_result in processed_events:
            for option_data in event_result['prompt_options']:
                prompt = create_split_identification_prompt(
                    option_data,
                    use_zoomed_images=event_result['use_zoomed_images'],
                    views=event_result['views'],
                    llm_processor=llm_processor,
                    zoom_margin=option_data.get('zoom_margin', zoom_margin),
                    prompt_mode=prompt_mode
                )
                prompts.extend([prompt] * K)
                indices.extend([j for j in range(K)])

                total_options_processed += 1

    elif task == "split_comparison":
        split_root_ids = [x['base_neuron_id'] for x in processed_events]
        all_prompt_options = [y for x in processed_events for y in x['prompt_options']]


        split_examples = [x for x in all_prompt_options if x['id'] in split_root_ids]
        no_split_examples = [x for x in all_prompt_options if x['id'] not in split_root_ids]
        # random.shuffle(no_split_examples)

        split_examples = split_examples[:min(len(split_examples), len(no_split_examples))]
        no_split_examples = no_split_examples[:min(len(split_examples), len(no_split_examples))]

        for positive_example, negative_example in zip(split_examples, no_split_examples):

            prompt = create_split_comparison_prompt(
                positive_example,
                negative_example,
                use_zoomed_images,
                views,
                llm_processor,
                zoom_margin,
                prompt_mode
            )
            prompts.extend([prompt] * K)
            indices.extend([j for j in range(K)])
            correct_answers.extend(["1"] * K)

            prompt = create_split_comparison_prompt(
                negative_example,
                positive_example,
                use_zoomed_images,
                views,
                llm_processor,
                zoom_margin,
                prompt_mode
            )
            prompts.extend([prompt] * K)
            indices.extend([j for j in range(K)])
            correct_answers.extend(["2"] * K)

            total_options_processed += 1

    llm_analysis = await llm_processor.process_batch(prompts)

    final_results = []
    total_options_processed = 0  # Reset counter for results processing

    if task == 'split_identification':
        for event_result in processed_events:
            for option_data in event_result['prompt_options']:
                for k in range(K):
                    response = llm_analysis[total_options_processed * K + k]
                    answer_analysis = evaluate_response(response)
                    index = indices[total_options_processed * K + k]

                    # Create unified result structure
                    unified_result = create_unified_result_structure(
                        task=task,
                        event_result=event_result,
                        option_data=option_data,
                        response=response,
                        answer_analysis=answer_analysis,
                        index=index,
                        model=model,
                        zoom_margin=zoom_margin,
                        prompt_mode=prompt_mode
                    )

                    final_results.append(unified_result)
                total_options_processed += 1

    elif task == "split_comparison":
        for i, (positive_example, negative_example) in enumerate(zip(split_examples, no_split_examples)):
            for k in range(2*K):
                response = llm_analysis[i * 2*K + k]
                answer_analysis = evaluate_response(response)
                index = indices[i * 2*K + k]
                correct_answer = correct_answers[i * 2*K + k]

                # Create event result structure for the unified format
                event_result = {
                    'operation_id': f'split_comparison_{i}_{k}',
                    'root_id_requires_split': positive_example['id'],
                    'root_id_does_not_require_split': negative_example['id'],
                    'merge_coords': positive_example['merge_coords'],
                    'views': views,
                    'use_zoomed_images': use_zoomed_images,
                    'image_paths': {},
                    'prompt_options': [positive_example, negative_example]
                }

                # Create unified result structure
                unified_result = create_unified_result_structure(
                    task=task,
                    event_result=event_result,
                    response=response,
                    answer_analysis=answer_analysis,
                    index=index,
                    model=model,
                    zoom_margin=zoom_margin,
                    prompt_mode=prompt_mode,
                    correct_answer=correct_answer
                )

                final_results.append(unified_result)

    print(f"LLM evaluation complete. Generated {len(final_results)} result rows.")
    final_results = pd.DataFrame(final_results)

    if K > 1:
        final_results.to_csv(f"{output_dir}/{model}_{task}_{prompt_mode}_analysis_results_K{K}.csv", index=False)
    else:
        final_results.to_csv(f"{output_dir}/{model}_{task}_{prompt_mode}_analysis_results.csv", index=False)
    return final_results


async def process_existing_split_results(
    results_file_path: str,
    output_dir: str,
    model: str,
    prompt_mode: str = 'informative',
    llm_processor: LLMProcessor = None,
    K: int = 10
) -> pd.DataFrame:
    """
    Load existing split results file and re-evaluate with a new LLM.

    Args:
        results_file_path: Path to existing results JSON file
        output_dir: Directory to save new results
        model: Model name to use for re-evaluation
        prompt_mode: Prompt mode to use
        llm_processor: LLM processor instance
        K: Number of repeated evaluations per prompt

    Returns:
        DataFrame with new evaluation results
    """
    print(f"Loading existing results from: {results_file_path}")

    try:
        with open(results_file_path, 'r') as f:
            existing_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {results_file_path}")
        return pd.DataFrame()

    if not existing_results:
        print("No results found in the file.")
        return pd.DataFrame()

    # Determine task type from existing results
    first_result = existing_results[0]
    task = first_result.get('task', 'unknown')

    if task not in ['split_comparison', 'split_identification']:
        print(f"Error: Task type '{task}' is not a split task. Use merge_resolution.py for merge tasks.")
        return pd.DataFrame()

    print(f"Detected task type: {task}")
    print(f"Found {len(existing_results)} existing results to re-evaluate")

    # Group results by operation_id to reconstruct prompt options
    results_by_operation = {}
    for result in existing_results[:100]:
        op_id = result.get('operation_id', 'unknown')
        if op_id not in results_by_operation:
            results_by_operation[op_id] = []
        results_by_operation[op_id].append(result)

    prompts = []
    indices = []
    operation_mapping = []  # Track which prompt belongs to which operation

    # Reconstruct prompts based on task type
    if task == 'split_identification':
        for op_id, op_results in results_by_operation.items():
            # Group by individual options
            options_by_id = {}
            for result in op_results:
                option_id = result.get('id')
                if option_id and option_id not in options_by_id:
                    options_by_id[option_id] = result

            for option_id, result in options_by_id.items():
                # Reconstruct option data
                option_data = {
                    'id': option_id,
                    'paths': result.get('image_paths', {}),
                    'merge_coords': result.get('merge_coords', []),
                    'zoom_margin': result.get('zoom_margin', 5000)
                }

                use_zoomed_images = result.get('use_zoomed_images', True)
                views = result.get('views', ['front', 'side', 'top'])
                zoom_margin = result.get('zoom_margin', 5000)

                prompt = create_split_identification_prompt(
                    option_data,
                    use_zoomed_images=use_zoomed_images,
                    views=views,
                    llm_processor=llm_processor,
                    zoom_margin=zoom_margin,
                    prompt_mode=prompt_mode
                )

                prompts.extend([prompt] * K)
                indices.extend([j for j in range(K)])
                operation_mapping.extend([f"{op_id}_{option_id}"] * K)

    elif task == 'split_comparison':
        # For split comparison, need to reconstruct positive/negative pairs
        for op_id, op_results in results_by_operation.items():
            first_result = op_results[0]

            # Extract pair information
            positive_id = first_result.get('root_id_requires_split')
            negative_id = first_result.get('root_id_does_not_require_split')
            use_zoomed_images = first_result.get('use_zoomed_images', True)
            views = first_result.get('views', ['front', 'side', 'top'])
            zoom_margin = first_result.get('zoom_margin', 5000)

            # Reconstruct option data for positive and negative examples
            positive_example = {
                'id': positive_id,
                'paths': first_result.get('image_paths', {}),
                'merge_coords': first_result.get('merge_coords', [])
            }
            negative_example = {
                'id': negative_id,
                'paths': first_result.get('image_paths', {}),
                'merge_coords': first_result.get('merge_coords', [])
            }

            # Create both orderings
            prompt1 = create_split_comparison_prompt(
                positive_example, negative_example,
                use_zoomed_images, views, llm_processor,
                zoom_margin, prompt_mode
            )
            prompt2 = create_split_comparison_prompt(
                negative_example, positive_example,
                use_zoomed_images, views, llm_processor,
                zoom_margin, prompt_mode
            )

            prompts.extend([prompt1] * K + [prompt2] * K)
            indices.extend([j for j in range(K)] + [j for j in range(K)])
            operation_mapping.extend([f"{op_id}_pos_first"] * K + [f"{op_id}_neg_first"] * K)

    if not prompts:
        print("No prompts could be reconstructed from existing results.")
        return pd.DataFrame()

    print(f"Re-evaluating {len(prompts)} prompts with model: {model}")

    # Process with new LLM
    llm_analysis = await llm_processor.process_batch(prompts)

    # Create new results with updated model responses
    final_results = []

    for i, (prompt_result, original_mapping) in enumerate(zip(llm_analysis, operation_mapping)):
        # Find the original result to copy metadata from
        original_result = None
        if '_' in original_mapping:
            parts = original_mapping.split('_')
            op_id = '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
            root_id = parts[-1] if 'pos_first' not in parts[-1] and 'neg_first' not in parts[-1] else None
        else:
            op_id = original_mapping

        for result in existing_results:
            if task == 'split_comparison':
                if str(result.get('operation_id')) == str(op_id):
                    original_result = result
                    break
            elif task == 'split_identification':
                if root_id and (str(result.get('operation_id')) == str(op_id)) and (str(result.get('id'))==str(root_id)):
                    original_result = result
                    break

        if not original_result:
            continue

        # Parse new response
        answer_analysis = evaluate_response(prompt_result)

        # Create new result based on original but with new model response
        new_result = original_result.copy()
        new_result.update({
            'model': model,
            'model_raw_answer': prompt_result,
            'model_analysis': answer_analysis.get('analysis', None),
            'model_prediction': answer_analysis.get('answer', None),
            'prompt_mode': prompt_mode,
            'index': indices[i] if i < len(indices) else 0
        })

        # Update task-specific fields based on new response
        if task == 'split_identification':
            new_result['model_answer'] = answer_analysis.get('answer', None)

        final_results.append(new_result)

    print(f"Re-evaluation complete. Generated {len(final_results)} result rows.")

    # Save new results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_df = pd.DataFrame(final_results)

    # Save to CSV
    if K > 1:
        csv_filename = f"{output_dir}/{model}_{task}_{prompt_mode}_analysis_results_K{K}.csv"
    else:
        csv_filename = f"{output_dir}/{model}_{task}_{prompt_mode}_analysis_results.csv"

    results_df.to_csv(csv_filename, index=False)

    # Save to JSON
    json_filename = os.path.join(output_dir, f"{task}_results_{timestamp}.json")
    results_df.to_json(json_filename, orient='records', indent=2)

    print(f"Saved re-evaluation results to {csv_filename}")
    print(f"Saved detailed results to {json_filename}")

    return results_df


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process split data from FlyWire EM data JSON and evaluate using an LLM.")
    parser.add_argument("--input-json", required=False, help="Path to the input em_data_*.json file (not required when using --results-file)")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regeneration of images even if they seem to exist.")
    parser.add_argument("--num-samples", type=int, default=None, help="Process only the first N split events found in the JSON file.")
    parser.add_argument("--use-zoomed-images", action=argparse.BooleanOptionalAction, default=True, help="Use zoomed images in the prompt instead of default.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers (threads). Defaults to CPU count or 4.")
    parser.add_argument("--views", nargs='+', choices=['front', 'side', 'top'], default=['front', 'side', 'top'], help="Specify which views to include (e.g., --views front side). Defaults to all.")
    parser.add_argument("--task", type=str, choices=['split_comparison', 'split_identification'], default='split_comparison', help="Specify the evaluation task to perform.")
    parser.add_argument("--species", type=str, choices=['mouse', 'fly'], default='mouse', help="Specify the species to use for the output directory.")
    parser.add_argument("--zoom-margin", type=int, default=1024, help="Specify the zoom margin to use for the output directory.")
    parser.add_argument("--models", nargs='+', default=["claude-3-7-sonnet-20250219"], help="Specify one or more models to use for evaluation.")
    parser.add_argument("--prompt-modes", nargs='+', default=['informative'], help="Specify one or more prompt modes to use for evaluation.")
    parser.add_argument("--results-file", type=str, help="Path to existing results JSON file to re-evaluate with new LLM (skips image generation).")
    parser.add_argument("--K", type=int, default=10, help="Number of repeated evaluations per prompt (default: 10).")
    args = parser.parse_args()

    json_path = args.input_json
    results_file = args.results_file

    # Validate that either input-json or results-file is provided
    if not json_path and not results_file:
        parser.error("Either --input-json or --results-file must be provided")

    if results_file and json_path:
        print("Warning: Both --input-json and --results-file provided. --results-file will take precedence.")

    force_regenerate = args.force_regenerate
    num_samples = args.num_samples
    use_zoomed = args.use_zoomed_images
    max_workers = args.max_workers
    selected_views = args.views
    task = args.task
    species = args.species
    zoom_margin = args.zoom_margin
    models = args.models
    prompt_modes = args.prompt_modes
    K = args.K

    # Process each combination of model and prompt mode
    for model in models:
        for prompt_mode in prompt_modes:
            print(f"\nProcessing with model: {model} and prompt mode: {prompt_mode}")

            # Check if we should use existing results file workflow
            if results_file:
                if not os.path.exists(results_file):
                    print(f"Error: Results file not found at {results_file}")
                    continue

                print(f"Using existing results file: {results_file}")
                print("Skipping image generation and re-evaluating with new LLM")

                # Determine output directory based on results file location or use default
                if os.path.dirname(results_file):
                    current_output_dir = os.path.dirname(results_file)
                else:
                    current_output_dir = f"output/{species}_split"

                llm_processor = LLMProcessor(model=model, max_tokens=4096, max_concurrent=25)

                # Process existing results
                results_df = asyncio.run(process_existing_split_results(
                    results_file,
                    current_output_dir,
                    model,
                    prompt_mode,
                    llm_processor,
                    K
                ))

                if results_df.empty:
                    print("No results generated from existing file.")
                    continue
                else:
                    print(f"Successfully re-evaluated existing results with {model}")

                continue  # Skip the regular processing workflow

            # Regular processing workflow (when no results file is provided)
            current_output_dir = f"output/{species}_split"

            llm_processor = LLMProcessor(model=model, max_tokens=4096, max_concurrent=10)

            # Validate input path
            if not os.path.exists(json_path):
                print(f"Error: Input JSON file not found at {json_path}")
                continue

            print(f"Selected Task: {task}")
            print(f"Using input JSON: {json_path}")
            print(f"Output directory: {current_output_dir}")
            print(f"Force regenerate images: {force_regenerate}")
            print(f"Using zoomed images: {use_zoomed}")
            print(f"Selected views: {selected_views}")
            print(f"K (repetitions): {K}")
            if num_samples is not None:
                print(f"Number of samples to process: {num_samples}")
            if max_workers is not None:
                print(f"Max workers specified: {max_workers}")

            # Create output directory
            os.makedirs(current_output_dir, exist_ok=True)

            # Process all split events
            results_df = asyncio.run(process_split_data(
                json_path,
                current_output_dir,
                task=task,
                force_regenerate=force_regenerate,
                num_samples=num_samples,
                use_zoomed_images=use_zoomed,
                max_workers=max_workers,
                views=selected_views,
                llm_processor=llm_processor,
                zoom_margin=zoom_margin,
                species=species,
                model=model,
                prompt_mode=prompt_mode,
                K=K
            ))

            if results_df.empty:
                print("No results generated.")
                continue

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save raw results DataFrame
            results_filename = os.path.join(current_output_dir, f"{task}_results_{timestamp}.json")
            try:
                results_df.to_json(results_filename, orient='records', indent=2)
                print(f"Saved detailed results to {results_filename}")
            except Exception as e:
                print(f"Error saving results DataFrame: {e}")


if __name__ == "__main__":
    main()
