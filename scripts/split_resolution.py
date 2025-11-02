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
    zoom_margin: int = 5000
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


def _load_split_event_metadata(
    first_metadata_path: str,
    last_metadata_path: str,
    base_neuron_id: str,
    final_neuron_id: str,
    merge_coords: List[float],
    timestamp: int,
    operation_id: str
) -> Optional[Dict[str, Any]]:
    """
    Load and validate existing metadata for a split event.

    Returns:
        Dictionary mapping neuron IDs to their image paths if valid, None otherwise.
    """
    if not os.path.exists(first_metadata_path) or not os.path.exists(last_metadata_path):
        return None

    image_paths = {}
    try:
        # Load first neuron metadata
        with open(first_metadata_path, 'r') as f:
            first_metadata = json.load(f)

        if (
            base_neuron_id in first_metadata.get('neuron_ids', [])
            and first_metadata.get('coords_used') == merge_coords
            and first_metadata.get('timestamp') == timestamp
        ):
            print(f"Found matching metadata for base neuron {base_neuron_id}. Loading existing image paths.")
            image_paths[base_neuron_id] = first_metadata.get('image_paths', {}).get(base_neuron_id)
        else:
            print(f"First metadata found but parameters mismatch for operation {operation_id}.")
            return None

        # Load last neuron metadata
        with open(last_metadata_path, 'r') as f:
            last_metadata = json.load(f)

        if (
            final_neuron_id in last_metadata.get('neuron_ids', [])
            and last_metadata.get('coords_used') == merge_coords
            and last_metadata.get('timestamp') == timestamp
        ):
            print(f"Found matching metadata for final neuron {final_neuron_id}. Loading existing image paths.")
            image_paths[final_neuron_id] = last_metadata.get('image_paths', {}).get(final_neuron_id)
        else:
            print(f"Last metadata found but parameters mismatch for operation {operation_id}.")
            return None

        return image_paths if len(image_paths) == 2 else None

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error reading metadata files: {e}. Will regenerate images.")
        return None


def _prepare_split_prompt_options(
    neuron_ids: List[str],
    image_paths: Dict[str, Any],
    image_set_key: str,
    merge_coords: List[float],
    zoom_margin: int,
    operation_id: str
) -> tuple[List[Dict], Dict[int, str]]:
    """
    Prepare option data for prompt creation from generated split images.

    Returns:
        Tuple of (prompt_options list, option_index_to_id mapping)
    """
    prompt_options = []
    option_index_to_id = {}
    current_index = 1
    coords_suffix = f"{int(merge_coords[0])}_{int(merge_coords[1])}_{int(merge_coords[2])}"

    for opt_id in neuron_ids:
        option_paths_dict = image_paths.get(opt_id)

        if option_paths_dict and (option_paths_dict.get('default') or option_paths_dict.get('zoomed')):
            img_check_path = option_paths_dict.get(image_set_key, {}).get('front')
            if img_check_path and os.path.exists(img_check_path):
                prompt_options.append({
                    'id': opt_id,
                    'paths': option_paths_dict,
                    "merge_coords": coords_suffix,
                    "zoom_margin": option_paths_dict.get('zoom_margin', zoom_margin)
                })
                option_index_to_id[current_index] = opt_id
                current_index += 1
            else:
                print(f"Warning: Required image ({image_set_key}/front) not found at {img_check_path} for option {opt_id}.")
        else:
            print(f"Warning: Image paths dictionary for option {opt_id} not found or generation failed.")

    return prompt_options, option_index_to_id


def _process_single_split_event(item, output_dir, task, force_regenerate, use_zoomed_images, views, species, zoom_margin):
    """
    Process a single split event: load/generate images and prepare evaluation data.

    This function orchestrates:
    1. Loading existing metadata or generating new images
    2. Preparing prompt data for LLM evaluation

    Returns:
        Dictionary with operation data and image paths, or None if processing failed.
    """
    operation_id = item.get('operation_id', 'N/A')

    try:
        # Extract neuron IDs
        base_neuron_id = str(item['before_root_ids'][0])
        first_split_neuron_id = str(item['after_root_ids'][0])
        other_split_neuron_id = str(item['after_root_ids'][1])
        final_neuron_id = str(item['neuron_id'])
        merge_coords = item['interface_point']
        timestamp_after_split = item.get('timestamp')

        # Setup paths
        coords_suffix = f"{int(merge_coords[0])}_{int(merge_coords[1])}_{int(merge_coords[2])}"
        first_neuron_dir = os.path.join(output_dir, f"base_{base_neuron_id}_{coords_suffix}")
        first_neuron_metadata_path = os.path.join(first_neuron_dir, "generation_metadata.json")
        last_neuron_dir = os.path.join(output_dir, f"base_{final_neuron_id}_{coords_suffix}")
        last_neuron_metadata_path = os.path.join(last_neuron_dir, "generation_metadata.json")

        image_paths = None

        # Try to load existing metadata
        if not force_regenerate:
            image_paths = _load_split_event_metadata(
                first_neuron_metadata_path,
                last_neuron_metadata_path,
                base_neuron_id,
                final_neuron_id,
                merge_coords,
                timestamp_after_split,
                operation_id
            )

        # Generate images if needed
        if force_regenerate or not image_paths:
            image_paths = generate_neuron_images(
                [base_neuron_id, final_neuron_id],
                [first_split_neuron_id, other_split_neuron_id],
                merge_coords,
                output_dir,
                timestamp=timestamp_after_split,
                species=item.get('species', 'fly')
            )

        if not image_paths:
            print(f"Error: Image paths unavailable for split op {operation_id}. Skipping evaluation.")
            return None

        # Prepare prompt options
        image_set_key = 'zoomed' if use_zoomed_images else 'default'
        prompt_options, option_index_to_id = _prepare_split_prompt_options(
            [base_neuron_id, final_neuron_id],
            image_paths,
            image_set_key,
            merge_coords,
            zoom_margin,
            operation_id
        )

        if not prompt_options:
            print(f"Warning: No valid option images found for split op {operation_id}. Skipping evaluation.")
            return None

        # Return structured result
        return {
            'operation_id': operation_id,
            'base_neuron_id': base_neuron_id,
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
        traceback.print_exc()
        return None


def _load_and_filter_split_data(json_path: str, num_samples: Optional[int] = None, seed: Optional[int] = None) -> List[Dict]:
    """
    Load JSON data and filter for valid split operations.

    Args:
        json_path: Path to input JSON file
        num_samples: Number of samples to randomly select (None = all)
        seed: Random seed for reproducible sampling

    Returns:
        List of split event dictionaries
    """
    try:
        with open(json_path, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return []

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

    # Randomly sample if specified
    if num_samples is not None and num_samples > 0:
        if num_samples < total_events_found:
            if seed is not None:
                random.seed(seed)
                print(f"Randomly sampling {num_samples} split events with seed={seed}.")
            else:
                print(f"Randomly sampling {num_samples} split events (no seed set).")
            split_data = random.sample(split_data, num_samples)
        else:
            print(f"Requested {num_samples} samples, but only {total_events_found} available. Processing all.")
    else:
        print(f"Processing all {total_events_found} split events.")

    return split_data


def _run_parallel_split_image_generation(
    split_data: List[Dict],
    output_dir: str,
    task: str,
    force_regenerate: bool,
    use_zoomed_images: bool,
    views: List[str],
    species: str,
    zoom_margin: int,
    max_workers: Optional[int] = None
) -> List[Dict]:
    """
    Run parallel image generation for split events using multiprocessing.

    Returns:
        List of processed event results (None filtered out)
    """
    os.makedirs(output_dir, exist_ok=True)

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    print(f"Using up to {max_workers} processes for parallel processing.")

    args_list = [
        (item, output_dir, task, force_regenerate, use_zoomed_images, views, species, zoom_margin)
        for item in split_data
    ]

    results_raw = []
    print(f"Running image generation/option preparation in parallel for {len(args_list)} events...")
    try:
        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
            results_raw = pool.starmap(_process_single_split_event, args_list)
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        import traceback
        traceback.print_exc()

    # Filter out failed events
    processed_events = [res for res in results_raw if res is not None]
    if len(processed_events) < len(results_raw):
        print(f"Warning: {len(results_raw) - len(processed_events)} split events failed during processing.")

    return processed_events


def _create_split_prompts(
    processed_events: List[Dict],
    task: str,
    llm_processor: LLMProcessor,
    zoom_margin: int,
    prompt_mode: str,
    K: int
) -> tuple[List[Any], List[int], List[str]]:
    """
    Create LLM prompts from processed split events with K repetitions.

    Returns:
        Tuple of (prompts list, indices list, correct_answers list for comparison task)
    """
    prompts = []
    indices = []
    correct_answers = []

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
                indices.extend(list(range(K)))

    elif task == "split_comparison":
        split_root_ids = [x['base_neuron_id'] for x in processed_events]
        all_prompt_options = [y for x in processed_events for y in x['prompt_options']]

        split_examples = [x for x in all_prompt_options if x['id'] in split_root_ids]
        no_split_examples = [x for x in all_prompt_options if x['id'] not in split_root_ids]

        min_len = min(len(split_examples), len(no_split_examples))
        split_examples = split_examples[:min_len]
        no_split_examples = no_split_examples[:min_len]

        for positive_example, negative_example in zip(split_examples, no_split_examples):
            # Positive first ordering
            prompt = create_split_comparison_prompt(
                positive_example, negative_example,
                processed_events[0]['use_zoomed_images'],
                processed_events[0]['views'],
                llm_processor,
                zoom_margin,
                prompt_mode
            )
            prompts.extend([prompt] * K)
            indices.extend(list(range(K)))
            correct_answers.extend(["1"] * K)

            # Negative first ordering
            prompt = create_split_comparison_prompt(
                negative_example, positive_example,
                processed_events[0]['use_zoomed_images'],
                processed_events[0]['views'],
                llm_processor,
                zoom_margin,
                prompt_mode
            )
            prompts.extend([prompt] * K)
            indices.extend(list(range(K)))
            correct_answers.extend(["2"] * K)

    return prompts, indices, correct_answers


def _process_llm_responses_for_splits(
    llm_analysis: List[str],
    processed_events: List[Dict],
    indices: List[int],
    correct_answers: List[str],
    task: str,
    model: str,
    zoom_margin: int,
    prompt_mode: str,
    K: int
) -> List[Dict]:
    """
    Process LLM responses into final result structures.

    Returns:
        List of result dictionaries
    """
    final_results = []
    total_options_processed = 0

    if task == 'split_identification':
        for i, event_result in enumerate(processed_events):
            for j, option_data in enumerate(event_result['prompt_options']):
                # Debug: Verify option_data has 'id' field
                if i == 0 and j == 0:  # First option only
                    print(f"DEBUG: First option_data keys: {list(option_data.keys())}")
                    print(f"DEBUG: First option_data['id']: {option_data.get('id')}")

                for k in range(K):
                    response = llm_analysis[total_options_processed * K + k]
                    answer_analysis = evaluate_response(response)
                    index = indices[total_options_processed * K + k]

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
        # Reconstruct split examples for result mapping
        split_root_ids = [x['base_neuron_id'] for x in processed_events]
        all_prompt_options = [y for x in processed_events for y in x['prompt_options']]

        split_examples = [x for x in all_prompt_options if x['id'] in split_root_ids]
        no_split_examples = [x for x in all_prompt_options if x['id'] not in split_root_ids]

        min_len = min(len(split_examples), len(no_split_examples))
        split_examples = split_examples[:min_len]
        no_split_examples = no_split_examples[:min_len]

        for i, (positive_example, negative_example) in enumerate(zip(split_examples, no_split_examples)):
            for k in range(2 * K):
                response = llm_analysis[i * 2 * K + k]
                answer_analysis = evaluate_response(response)
                index = indices[i * 2 * K + k]
                correct_answer = correct_answers[i * 2 * K + k]

                event_result = {
                    'operation_id': f'split_comparison_{i}_{k}',
                    'root_id_requires_split': positive_example['id'],
                    'root_id_does_not_require_split': negative_example['id'],
                    'merge_coords': positive_example['merge_coords'],
                    'views': processed_events[0]['views'],
                    'use_zoomed_images': processed_events[0]['use_zoomed_images'],
                    'image_paths': {},
                    'prompt_options': [positive_example, negative_example]
                }

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

    return final_results


def process_split_images(
    json_path: str,
    output_dir: str,
    force_regenerate=False,
    num_samples: Optional[int] = None,
    use_zoomed_images=True,
    max_workers: Optional[int] = None,
    views=['front', 'side', 'top'],
    task='split_comparison',
    species: str = "fly",
    zoom_margin: int = 5000,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Phase 1: Load split data and generate images for all split events.

    This phase:
    1. Loads and filters input data
    2. Runs parallel image generation for split events

    Args:
        seed: Random seed for reproducible sampling

    Returns:
        List of processed event dictionaries with image paths
    """
    # 1. Load and filter data
    split_data = _load_and_filter_split_data(json_path, num_samples, seed)
    if not split_data:
        return []

    # 2. Run parallel image generation
    processed_events = _run_parallel_split_image_generation(
        split_data, output_dir, task, force_regenerate, use_zoomed_images,
        views, species, zoom_margin, max_workers
    )

    if not processed_events:
        print("No events were successfully processed.")
        return []

    print(f"Image generation complete. Processed {len(processed_events)} events.")
    return processed_events


async def evaluate_split_events(
    processed_events: List[Dict],
    output_dir: str,
    task: str,
    model: str,
    llm_processor: LLMProcessor,
    zoom_margin: int = 5000,
    prompt_mode: str = 'informative',
    K: int = 10
) -> pd.DataFrame:
    """
    Phase 2: Evaluate processed split events with LLM.

    This phase:
    3. Creates prompts for LLM evaluation
    4. Runs LLM batch processing
    5. Processes responses into structured results
    6. Saves results to CSV

    Returns:
        DataFrame with evaluation results
    """
    # 3. Create prompts
    prompts, indices, correct_answers = _create_split_prompts(
        processed_events, task, llm_processor, zoom_margin, prompt_mode, K
    )

    if not prompts:
        print("No prompts could be created.")
        return pd.DataFrame()

    # 4. Run LLM evaluation
    print(f"Running LLM evaluation with {model} on {len(prompts)} prompts...")
    llm_analysis = await llm_processor.process_batch(prompts)

    # 5. Process LLM responses
    final_results = _process_llm_responses_for_splits(
        llm_analysis, processed_events, indices, correct_answers,
        task, model, zoom_margin, prompt_mode, K
    )

    print(f"LLM evaluation complete. Generated {len(final_results)} result rows.")

    # 6. Save results
    results_df = pd.DataFrame(final_results)

    # Validation: Check critical columns are present for split_identification
    if task == 'split_identification':
        if 'id' not in results_df.columns:
            print("WARNING: 'id' column missing from results!")
        elif results_df['id'].isna().all():
            print("WARNING: All 'id' values are None!")
        else:
            print(f"✓ 'id' column present with {results_df['id'].notna().sum()}/{len(results_df)} non-null values")

    output_filename = f"{output_dir}/{model}_{task}_{prompt_mode}_analysis_results"
    if K > 1:
        output_filename += f"_K{K}"
    output_filename += ".csv"

    results_df.to_csv(output_filename, index=False)

    # Also save JSON with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = os.path.join(output_dir, f"{task}_results_{model}_{prompt_mode}_{timestamp}.json")
    results_df.to_json(json_filename, orient='records', indent=2)
    print(f"Saved results to {output_filename} and {json_filename}")

    return results_df


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
    for result in existing_results:
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
                option_data = result.get('prompt_options', {})[0]

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42).")
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
    seed = args.seed

    # Check if we should use existing results file workflow
    if results_file:
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            return

        print(f"Using existing results file: {results_file}")
        print("Skipping image generation and re-evaluating with new LLM")

        # Determine output directory based on results file location or use default
        if os.path.dirname(results_file):
            current_output_dir = os.path.dirname(results_file)
        else:
            current_output_dir = f"output/{species}_split"

        # Process each combination of model and prompt mode
        for model in models:
            for prompt_mode in prompt_modes:
                print(f"\nRe-evaluating with model: {model} and prompt mode: {prompt_mode}")

                llm_processor = LLMProcessor(model=model, max_tokens=4096, max_concurrent=32)

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

        return  # Exit after processing results file

    # Regular processing workflow (when no results file is provided)
    current_output_dir = f"output/{species}_split"

    # Validate input path
    if not os.path.exists(json_path):
        print(f"Error: Input JSON file not found at {json_path}")
        return

    print(f"Selected Task: {task}")
    print(f"Using input JSON: {json_path}")
    print(f"Output directory: {current_output_dir}")
    print(f"Force regenerate images: {force_regenerate}")
    print(f"Using zoomed images: {use_zoomed}")
    print(f"Selected views: {selected_views}")
    print(f"K (repetitions): {K}")
    print(f"Random seed: {seed}")
    if num_samples is not None:
        print(f"Number of samples to process: {num_samples}")
    if max_workers is not None:
        print(f"Max workers specified: {max_workers}")

    # Create output directory
    os.makedirs(current_output_dir, exist_ok=True)

    # ========================================================================
    # PHASE 1: Generate images once (outside model/prompt loop)
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Generating images for all split events")
    print("="*60)

    processed_events = process_split_images(
        json_path,
        current_output_dir,
        force_regenerate=force_regenerate,
        num_samples=num_samples,
        use_zoomed_images=use_zoomed,
        max_workers=max_workers,
        views=selected_views,
        task=task,
        species=species,
        zoom_margin=zoom_margin,
        seed=seed
    )

    if not processed_events:
        print("No events were successfully processed. Exiting.")
        return

    # ========================================================================
    # PHASE 2: Loop over models and prompts, evaluating with LLM
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Running LLM evaluations")
    print("="*60)

    for model in models:
        for prompt_mode in prompt_modes:
            print(f"\nEvaluating with model: {model} and prompt mode: {prompt_mode}")

            llm_processor = LLMProcessor(model=model, max_tokens=4096, max_concurrent=32)

            results_df = asyncio.run(evaluate_split_events(
                processed_events,
                current_output_dir,
                task=task,
                model=model,
                llm_processor=llm_processor,
                zoom_margin=zoom_margin,
                prompt_mode=prompt_mode,
                K=K
            ))

            if results_df.empty:
                print(f"No results generated for {model} with {prompt_mode}.")
                continue
            else:
                print(f"Successfully completed evaluation for {model} with {prompt_mode}")


if __name__ == "__main__":
    main()
