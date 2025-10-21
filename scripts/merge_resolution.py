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
from prompts import create_merge_identification_prompt, create_merge_comparison_prompt

logging.basicConfig(level=logging.INFO)


def _get_neuron_directory(output_dir: str, base_neuron_id: str, merge_coords: List[float]) -> str:
    """Generate unique directory path for a merge event."""
    coords_suffix = f"{int(merge_coords[0])}_{int(merge_coords[1])}_{int(merge_coords[2])}"
    return os.path.join(output_dir, f"merge_{base_neuron_id}_{coords_suffix}")


def _generate_option_images(
    visualizer: ConnectomeVisualizer,
    option_id: str,
    neuron_dir: str,
    merge_coords: List[float],
    zoom_margin: int
) -> Dict[str, Dict[str, str]]:
    """
    Generate zoomed images for a single option neuron.

    Returns:
        Dictionary with 'zoomed' key containing view paths, or empty if generation failed.
    """
    merge_x, merge_y, merge_z = merge_coords
    option_img_paths = {'default': {}, 'zoomed': {}}

    zoomed_option_base_filename = f"option_{option_id}_with_base_zoomed"
    expected_zoomed_paths = {
        view: os.path.join(neuron_dir, f"{zoomed_option_base_filename}_{view}.png")
        for view in ['front', 'side', 'top']
    }

    # Check if images already exist
    if all(os.path.exists(p) for p in expected_zoomed_paths.values()):
        print(f"Found existing zoomed images for option {option_id}. Skipping generation.")
        option_img_paths['zoomed'] = expected_zoomed_paths
        return option_img_paths

    # Generate new images
    print(f"Generating zoomed images for option {option_id}...")
    try:
        visualizer.add_neurons([int(option_id)])
    except Exception as e:
        print(f"ERROR: Failed to load option {option_id}: {e}. Skipping.")
        return {}

    try:
        bbox = Bbox(
            (merge_x - zoom_margin, merge_y - zoom_margin, merge_z - zoom_margin),
            (merge_x + zoom_margin, merge_y + zoom_margin, merge_z + zoom_margin),
            unit="nm"
        )
        visualizer.create_3d_neuron_figure(bbox=bbox, add_em_slice=False)
        save_result = visualizer.save_3d_views(bbox=bbox, base_filename=zoomed_option_base_filename)

        if save_result is None:
            print(f"WARNING: Image generation failed or timed out for option {option_id}.")
        else:
            # Verify all expected files were created
            saved_paths = {}
            all_saved = True
            for view, expected_path in expected_zoomed_paths.items():
                if os.path.exists(expected_path):
                    saved_paths[view] = expected_path
                else:
                    print(f"Warning: Missing view file after save: {expected_path}")
                    all_saved = False

            if all_saved:
                option_img_paths['zoomed'] = saved_paths
            else:
                print(f"Warning: Not all views were saved for option {option_id}.")

    except Exception as e:
        logging.error(f"ERROR: Failed to save images for option {option_id}: {str(e)}")

    return option_img_paths


def generate_neuron_option_images(
    base_neuron_id: str,
    option_ids: List[str],
    merge_coords: List[float],
    output_dir: str,
    timestamp: Optional[int] = None,
    species: str = "fly",
    zoom_margin: int = 5000,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Generate images for a base neuron and potential merge options near merge coordinates.

    Args:
        base_neuron_id: String ID of the primary neuron
        option_ids: List of string IDs for other potential merge options
        merge_coords: Coordinates of the merge interface point [x, y, z]
        output_dir: Directory to save images
        timestamp: Optional timestamp for CAVEclient state
        species: Species for the dataset
        zoom_margin: Margin around merge point for zoomed views
        model: Model name for metadata

    Returns:
        Dictionary with paths to generated images
    """
    neuron_dir = _get_neuron_directory(output_dir, base_neuron_id, merge_coords)
    os.makedirs(neuron_dir, exist_ok=True)

    visualizer = ConnectomeVisualizer(
        output_dir=neuron_dir,
        dataset="public",
        timestamp=timestamp,
        species=species,
        verbose=False
    )

    option_images_dict = {}

    # Filter out base neuron from options
    option_ids_to_process = [opt_id for opt_id in option_ids if opt_id != base_neuron_id]

    # Load base neuron once (shown with all options)
    visualizer.clear_neurons()
    visualizer.load_neurons([int(base_neuron_id)])

    # Generate images for each option
    for option_id in option_ids_to_process:
        print(f"Processing images for option {option_id}...")
        option_img_paths = _generate_option_images(
            visualizer, option_id, neuron_dir, merge_coords, zoom_margin
        )

        if option_img_paths.get('zoomed') or option_img_paths.get('default'):
            option_images_dict[option_id] = option_img_paths
        else:
            print(f"No images generated for option {option_id}. Not adding to results.")

        # Remove option before processing next one
        visualizer.remove_neurons([int(option_id)])

    # Prepare final structure
    final_image_paths = {
        'options': option_images_dict
    }

    # Save metadata
    metadata = {
        'base_neuron_id': base_neuron_id,
        'option_ids_processed': option_ids_to_process,
        'merge_coords': merge_coords,
        'timestamp': timestamp,
        'image_paths': final_image_paths,
        'model': model
    }
    metadata_path = os.path.join(neuron_dir, "generation_metadata.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved generation metadata to {metadata_path}")
    except Exception as e:
        print(f"ERROR: Failed to save generation metadata to {metadata_path}: {e}")

    return final_image_paths


def _load_merge_event_metadata(metadata_path: str, base_neuron_id: str, merge_coords: List[float], timestamp: int, operation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load and validate existing metadata for a merge event.

    Returns:
        Metadata dictionary if valid, None if doesn't exist or doesn't match parameters.
    """
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Validate parameters match
        if (
            metadata.get('base_neuron_id') == base_neuron_id
            and metadata.get('merge_coords') == merge_coords
            and metadata.get('timestamp') == timestamp
            and 'option_ids_processed' in metadata
        ):
            print(f"Metadata matches current parameters. Loading existing data for op {operation_id}.")
            return metadata
        else:
            print(f"Metadata found but parameters mismatch for op {operation_id}. Will regenerate.")
            return None

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error reading metadata file {metadata_path}: {e}. Will regenerate.")
        return None


def _check_option_images_exist(neuron_dir: str, option_id: str, image_set_key: str, views: List[str]) -> bool:
    """Check if all required image files exist for an option."""
    option_dir_check = os.path.join(neuron_dir, f"option_{option_id}_with_base_{image_set_key}")
    for view in views:
        if not os.path.exists(f"{option_dir_check}_{view}.png"):
            return False
    return True


def _sample_merge_options(
    correct_id: str,
    all_available_option_ids: List[str],
    correct_merged_pair: set,
    neuron_dir: str,
    image_set_key: str,
    views: List[str],
    max_options: int,
    operation_id: str
) -> List[str]:
    """
    Sample merge options with priority: correct answer > existing images > random sampling.

    Strategy:
    1. Always include correct answer
    2. Prefer options with existing complete images (avoid regeneration)
    3. Sample random distractors if needed to reach max_options

    Args:
        correct_id: The correct merge option ID
        all_available_option_ids: All neurons visible in the volume
        correct_merged_pair: Set of both neurons in the merge
        neuron_dir: Directory where images are stored
        image_set_key: 'zoomed' or 'default'
        views: List of required views (e.g., ['front', 'side', 'top'])
        max_options: Target number of options
        operation_id: For logging

    Returns:
        List of option IDs to use (shuffled)
    """
    os.makedirs(neuron_dir, exist_ok=True)

    # Find which options already have complete images
    completed_options = []
    potential_incorrect_options = list(set(all_available_option_ids) - correct_merged_pair)

    # Check correct option
    if _check_option_images_exist(neuron_dir, correct_id, image_set_key, views):
        print(f"Found existing complete images for correct option: {correct_id}")
        completed_options.append(correct_id)
    else:
        print(f"Correct option {correct_id} missing some images.")

    # Check incorrect options
    for incorrect_id in potential_incorrect_options:
        if _check_option_images_exist(neuron_dir, incorrect_id, image_set_key, views):
            print(f"Found existing complete images for incorrect option: {incorrect_id}")
            completed_options.append(incorrect_id)

    # Build final list: correct + completed incorrect + sampled new incorrect
    final_options = [correct_id]

    # Remove correct from completed to avoid duplicates
    if correct_id in completed_options:
        completed_options.remove(correct_id)

    # Add completed incorrect options
    num_needed = max_options - len(final_options)
    add_completed = completed_options[:num_needed]
    final_options.extend(add_completed)

    # Sample new incorrect options if still needed
    num_still_needed = max_options - len(final_options)
    if num_still_needed > 0:
        remaining_incorrect_pool = list(set(potential_incorrect_options) - set(completed_options))
        num_to_sample = min(len(remaining_incorrect_pool), num_still_needed)
        sampled_new_incorrect = random.sample(remaining_incorrect_pool, num_to_sample)
        final_options.extend(sampled_new_incorrect)

    random.shuffle(final_options)
    print(f"Final selected option IDs for op {operation_id}: {final_options}")
    return final_options


def _prepare_prompt_options(
    option_ids: List[str],
    base_neuron_id: str,
    image_paths: Dict[str, Any],
    image_set_key: str,
    operation_id: str
) -> tuple[List[Dict], Dict[int, str]]:
    """
    Prepare option data for prompt creation from generated images.

    Returns:
        Tuple of (prompt_options list, option_index_to_id mapping)
    """
    prompt_options = []
    option_index_to_id = {}
    current_index = 1
    options_with_paths = image_paths.get('options', {})

    for opt_id in option_ids:
        # Skip base neuron (it's context, not an answer choice)
        if opt_id == base_neuron_id:
            continue

        option_paths_dict = options_with_paths.get(opt_id)

        # Verify images exist
        if option_paths_dict and (option_paths_dict.get('default') or option_paths_dict.get('zoomed')):
            img_check_path = option_paths_dict.get(image_set_key, {}).get('front')
            if img_check_path and os.path.exists(img_check_path):
                prompt_options.append({
                    'id': opt_id,
                    'paths': option_paths_dict
                })
                option_index_to_id[current_index] = opt_id
                current_index += 1
            else:
                print(f"Warning: Required image ({image_set_key}/front) not found at {img_check_path} for option {opt_id}. Excluding from prompt.")
        else:
            print(f"Warning: Image paths dictionary for option {opt_id} not found. Excluding from prompt.")

    return prompt_options, option_index_to_id


def _process_single_merge_event(item, output_dir, force_regenerate, use_zoomed_images, views, zoom_margin, model, skip_image_generation=False):
    """
    Process a single merge event: load/generate images and prepare evaluation data.

    This function orchestrates:
    1. Loading existing metadata or sampling new options
    2. Generating images for options (if needed)
    3. Preparing prompt data for LLM evaluation

    Returns:
        Dictionary with operation data and image paths, or None if processing failed.
    """
    operation_id = item.get('operation_id', 'N/A')

    try:
        # Extract ground truth and available options
        base_neuron_id = str(item['before_root_ids'][0])
        correct_merged_pair = {str(id) for id in item['before_root_ids'][:2]}
        expected_choice_ids = list(correct_merged_pair - {base_neuron_id})
        correct_id = expected_choice_ids[0] if expected_choice_ids else None

        merge_coords = item['interface_point']
        all_available_option_ids = [str(id) for id in item['em_data']['all_unique_root_ids']]
        timestamp_before_merge = item.get('prev_timestamp')

        # Setup paths
        neuron_dir = _get_neuron_directory(output_dir, base_neuron_id, merge_coords)
        metadata_path = os.path.join(neuron_dir, "generation_metadata.json")

        MAX_OPTIONS_TOTAL = 2
        option_ids = None
        image_paths = None

        # Try to load existing metadata
        if not force_regenerate:
            metadata = _load_merge_event_metadata(
                metadata_path, base_neuron_id, merge_coords, timestamp_before_merge, operation_id
            )
            if metadata:
                option_ids = metadata.get('option_ids_processed')
                image_paths = metadata.get('image_paths')
                # Force regeneration if we have too few options
                if len(option_ids) < MAX_OPTIONS_TOTAL:
                    force_regenerate = True
                    option_ids = None
                    image_paths = None

        # Sample options if we don't have valid metadata
        if (force_regenerate or not os.path.exists(metadata_path)) and not skip_image_generation:
            if not correct_id:
                print(f"Warning: No correct option for op {operation_id}. Skipping.")
                return None

            print(f"Sampling options for merge op {operation_id}...")
            image_set_key = 'zoomed' if use_zoomed_images else 'default'
            option_ids = _sample_merge_options(
                correct_id,
                all_available_option_ids,
                correct_merged_pair,
                neuron_dir,
                image_set_key,
                views,
                MAX_OPTIONS_TOTAL,
                operation_id
            )

        if not option_ids:
            print(f"Warning: No options determined for op {operation_id}. Skipping.")
            return None

        # Generate images if needed
        if (force_regenerate or not os.path.exists(metadata_path)) and not skip_image_generation:
            image_paths = generate_neuron_option_images(
                base_neuron_id,
                option_ids,
                merge_coords,
                output_dir,
                timestamp=timestamp_before_merge,
                species=item.get('species', 'Not specified'),
                zoom_margin=zoom_margin,
                model=model
            )

        if not image_paths:
            print(f"Error: Image paths unavailable for merge op {operation_id}. Skipping evaluation.")
            return None

        # Prepare prompt options
        image_set_key = 'zoomed' if use_zoomed_images else 'default'
        prompt_options, option_index_to_id = _prepare_prompt_options(
            option_ids, base_neuron_id, image_paths, image_set_key, operation_id
        )

        if not prompt_options:
            print(f"Warning: No valid option images found for merge op {operation_id}. Skipping evaluation.")
            return None

        # Return structured result
        return {
            'operation_id': operation_id,
            'base_neuron_id': base_neuron_id,
            'correct_merged_pair': list(correct_merged_pair),
            'options_presented_ids': [option_index_to_id[i] for i in sorted(option_index_to_id.keys())],
            'expected_choice_ids': expected_choice_ids,
            'num_options_presented': len(prompt_options),
            'prompt_options': prompt_options,
            'views': views,
            'use_zoomed_images': use_zoomed_images,
            'image_paths': image_paths,
            'option_index_to_id': option_index_to_id,
            'before_root_ids': item.get('before_root_ids', []),
            'after_root_ids': item.get('after_root_ids', []),
            'merge_coords': merge_coords,
            'interface_point': item.get('interface_point', None),
            'timestamp': timestamp_before_merge
        }

    except Exception as e:
        print(f"Error processing merge event for operation {operation_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_and_filter_merge_data(json_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    Load JSON data and filter for valid merge operations.

    Returns:
        List of merge event dictionaries
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

    # Filter for merge operations with necessary data
    merge_data = [
        item for item in all_data
        if item.get('is_merge') is True
        and item.get('before_root_ids')
        and len(item['before_root_ids']) >= 2
        and item.get('interface_point')
        and item.get('em_data')
        and item['em_data'].get('all_unique_root_ids')
    ]

    total_events_found = len(merge_data)
    print(f"Found {total_events_found} merge events in the input file.")

    # Limit samples if specified
    if num_samples is not None and num_samples > 0:
        if num_samples < total_events_found:
            print(f"Processing only the first {num_samples} merge events.")
            merge_data = merge_data[:num_samples]
        else:
            print(f"Requested {num_samples} samples, but only {total_events_found} available. Processing all.")
    else:
        print(f"Processing all {total_events_found} merge events.")

    return merge_data


def _run_parallel_image_generation(
    merge_data: List[Dict],
    output_dir: str,
    force_regenerate: bool,
    use_zoomed_images: bool,
    views: List[str],
    zoom_margin: int,
    model: str,
    max_workers: Optional[int] = None
) -> List[Dict]:
    """
    Run parallel image generation for merge events using multiprocessing.

    Returns:
        List of processed event results (None filtered out)
    """
    os.makedirs(output_dir, exist_ok=True)

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    print(f"Using up to {max_workers} processes for parallel processing.")

    args_list = [
        (item, output_dir, force_regenerate, use_zoomed_images, views, zoom_margin, model)
        for item in merge_data
    ]

    results_raw = []
    print(f"Running image generation/option preparation in parallel for {len(args_list)} events...")
    try:
        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
            results_raw = pool.starmap(_process_single_merge_event, args_list)
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        import traceback
        traceback.print_exc()

    # Filter out failed events
    processed_events = [res for res in results_raw if res is not None]
    if len(processed_events) < len(results_raw):
        print(f"Warning: {len(results_raw) - len(processed_events)} merge events failed during processing.")

    return processed_events


def _create_merge_prompts(
    processed_events: List[Dict],
    task: str,
    llm_processor: LLMProcessor,
    zoom_margin: int,
    prompt_mode: str,
    K: int
) -> tuple[List[Any], List[int]]:
    """
    Create LLM prompts from processed merge events with K repetitions.

    Returns:
        Tuple of (prompts list, indices list)
    """
    prompts = []
    indices = []

    for i, event_result in enumerate(processed_events):
        if task == 'merge_comparison':
            prompt = create_merge_comparison_prompt(
                event_result['prompt_options'],
                use_zoomed_images=event_result['use_zoomed_images'],
                views=event_result['views'],
                llm_processor=llm_processor,
                zoom_margin=zoom_margin,
                prompt_mode=prompt_mode
            )
            prompts.extend([prompt] * K)
            indices.extend([j for j in range(K)])

        elif task == 'merge_identification':
            for option_data in event_result['prompt_options']:
                prompt = create_merge_identification_prompt(
                    option_data,
                    use_zoomed_images=event_result['use_zoomed_images'],
                    views=event_result['views'],
                    llm_processor=llm_processor,
                    prompt_mode=prompt_mode
                )
                prompts.extend([prompt] * K)
                indices.extend([j for j in range(K)])

    return prompts, indices


def _process_llm_responses_for_merges(
    llm_analysis: List[str],
    processed_events: List[Dict],
    indices: List[int],
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

    for i, event_result in enumerate(processed_events):
        if task == 'merge_comparison':
            for k in range(K):
                response = llm_analysis[i * K + k]
                answer_analysis = evaluate_response(response)
                index = indices[i * K + k]

                # Determine model chosen ID
                model_chosen_id = "none"
                error = None
                if answer_analysis["answer"] != "none":
                    try:
                        answer_int = int(answer_analysis["answer"])
                        if answer_int in event_result['option_index_to_id']:
                            model_chosen_id = str(event_result['option_index_to_id'][answer_int])
                        else:
                            model_chosen_id = "none"
                            error = "Model returned index out of bounds"
                    except (ValueError, KeyError):
                        model_chosen_id = "none"
                        error = "Could not parse model answer"

                # Create unified result
                unified_result = create_unified_result_structure(
                    task=task,
                    event_result=event_result,
                    response=response,
                    answer_analysis=answer_analysis,
                    index=index,
                    model=model,
                    zoom_margin=zoom_margin,
                    prompt_mode=prompt_mode
                )

                unified_result.update({
                    'model_chosen_id': model_chosen_id,
                    'error': error
                })

                final_results.append(unified_result)

        elif task == 'merge_identification':
            for j, option_data in enumerate(event_result['prompt_options']):
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

    return final_results


async def process_merge_data(
    json_path: str,
    output_dir: str,
    force_regenerate=False,
    num_samples: Optional[int] = None,
    use_zoomed_images=True,
    max_workers: Optional[int] = None,
    views=['front', 'side', 'top'],
    task='merge_comparison',
    llm_processor: LLMProcessor = None,
    zoom_margin: int = 5000,
    species: str = "fly",
    model: str = "gpt-4o-mini",
    prompt_mode: str = 'informative',
    K: int = 10
):
    """
    Process merge data: load data, generate images, evaluate with LLM, and save results.

    This is the main pipeline that orchestrates:
    1. Loading and filtering input data
    2. Parallel image generation for merge events
    3. Creating prompts for LLM evaluation
    4. Running LLM batch processing
    5. Processing responses into structured results
    6. Saving results to CSV

    Returns:
        DataFrame with evaluation results
    """
    # 1. Load and filter data
    merge_data = _load_and_filter_merge_data(json_path, num_samples)
    if not merge_data:
        return pd.DataFrame()

    # 2. Run parallel image generation
    processed_events = _run_parallel_image_generation(
        merge_data, output_dir, force_regenerate, use_zoomed_images,
        views, zoom_margin, model, max_workers
    )

    if not processed_events:
        print("No events were successfully processed.")
        return pd.DataFrame()

    # 3. Create prompts
    prompts, indices = _create_merge_prompts(
        processed_events, task, llm_processor, zoom_margin, prompt_mode, K
    )

    if not prompts:
        print("No prompts could be created.")
        return pd.DataFrame()

    # 4. Run LLM evaluation
    llm_analysis = await llm_processor.process_batch(prompts)

    # 5. Process LLM responses
    final_results = _process_llm_responses_for_merges(
        llm_analysis, processed_events, indices, task, model, zoom_margin, prompt_mode, K
    )

    print(f"LLM evaluation complete. Generated {len(final_results)} result rows.")

    # 6. Save results
    results_df = pd.DataFrame(final_results)
    output_filename = f"{output_dir}/{model}_{task}_{prompt_mode}_analysis_results"
    if K > 1:
        output_filename += f"_K{K}"
    output_filename += ".csv"

    results_df.to_csv(output_filename, index=False)
    return results_df


def _load_existing_results(results_file_path: str) -> tuple[List[Dict], str]:
    """
    Load and validate existing results file.

    Returns:
        Tuple of (existing_results list, task type), or ([], None) on error
    """
    print(f"Loading existing results from: {results_file_path}")

    try:
        with open(results_file_path, 'r') as f:
            existing_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
        return [], None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {results_file_path}")
        return [], None

    if not existing_results:
        print("No results found in the file.")
        return [], None

    task = existing_results[0].get('task', 'unknown')
    if task not in ['merge_comparison', 'merge_identification']:
        print(f"Error: Task type '{task}' is not a merge task. Use split_resolution.py for split tasks.")
        return [], None

    print(f"Detected task type: {task}")
    print(f"Found {len(existing_results)} existing results to re-evaluate")

    return existing_results, task


def _reconstruct_prompts_from_results(
    existing_results: List[Dict],
    task: str,
    llm_processor: LLMProcessor,
    prompt_mode: str,
    K: int
) -> tuple[List[Any], List[int], List[str]]:
    """
    Reconstruct prompts from existing results.

    Returns:
        Tuple of (prompts list, indices list, operation_mapping list)
    """
    # Group results by operation_id
    results_by_operation = {}
    for result in existing_results[:100]:
        op_id = result.get('operation_id', 'unknown')
        if op_id not in results_by_operation:
            results_by_operation[op_id] = []
        results_by_operation[op_id].append(result)

    prompts = []
    indices = []
    operation_mapping = []

    if task == 'merge_comparison':
        for op_id, op_results in results_by_operation.items():
            first_result = op_results[0]
            prompt_options = first_result.get('prompt_options', [])
            if not prompt_options:
                continue

            prompt = create_merge_comparison_prompt(
                prompt_options,
                use_zoomed_images=first_result.get('use_zoomed_images', True),
                views=first_result.get('views', ['front', 'side', 'top']),
                llm_processor=llm_processor,
                zoom_margin=first_result.get('zoom_margin', 5000),
                prompt_mode=prompt_mode
            )
            prompts.extend([prompt] * K)
            indices.extend(list(range(K)))
            operation_mapping.extend([op_id] * K)

    elif task == 'merge_identification':
        for op_id, op_results in results_by_operation.items():
            # Group by option_id
            options_by_id = {}
            for result in op_results:
                option_id = result.get('id')
                if option_id and option_id not in options_by_id:
                    options_by_id[option_id] = result

            for option_id, result in options_by_id.items():
                # Find matching prompt option
                matching_options = [
                    po.get('paths', {})
                    for po in result.get('prompt_options', [])
                    if po.get('id') == option_id
                ]
                if not matching_options:
                    continue

                option_data = {'id': option_id, 'paths': matching_options[0]}
                prompt = create_merge_identification_prompt(
                    option_data,
                    use_zoomed_images=result.get('use_zoomed_images', True),
                    views=result.get('views', ['front', 'side', 'top']),
                    llm_processor=llm_processor,
                    prompt_mode=prompt_mode
                )
                prompts.extend([prompt] * K)
                indices.extend(list(range(K)))
                operation_mapping.extend([f"{op_id}_{option_id}"] * K)

    return prompts, indices, operation_mapping


def _map_responses_to_original_results(
    llm_analysis: List[str],
    operation_mapping: List[str],
    indices: List[int],
    existing_results: List[Dict],
    task: str,
    model: str,
    prompt_mode: str
) -> List[Dict]:
    """
    Map LLM responses back to original results with updated model info.

    Returns:
        List of updated result dictionaries
    """
    final_results = []

    for i, (response, mapping) in enumerate(zip(llm_analysis, operation_mapping)):
        # Parse mapping to get op_id and option_id (if applicable)
        if '_' in mapping:
            parts = mapping.split('_')
            op_id = '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
            root_id = parts[-1]
        else:
            op_id = mapping
            root_id = None

        # Find matching original result
        original_result = None
        for result in existing_results:
            if task == 'merge_comparison' and str(result.get('operation_id')) == str(op_id):
                original_result = result
                break
            elif task == 'merge_identification' and \
                 str(result.get('operation_id')) == str(op_id) and \
                 str(result.get('id')) == str(root_id):
                original_result = result
                break

        if not original_result:
            continue

        # Parse response and create updated result
        answer_analysis = evaluate_response(response)
        new_result = original_result.copy()
        new_result.update({
            'model': model,
            'model_raw_answer': response,
            'model_analysis': answer_analysis.get('analysis'),
            'model_prediction': answer_analysis.get('answer'),
            'prompt_mode': prompt_mode,
            'index': indices[i] if i < len(indices) else 0
        })

        # Add task-specific fields
        if task == 'merge_comparison':
            model_chosen_id = "none"
            error = None
            if answer_analysis["answer"] != "none":
                try:
                    choice_idx = int(answer_analysis["answer"])
                    prompt_options = original_result.get('prompt_options', [])
                    option_index_to_id = {j+1: opt['id'] for j, opt in enumerate(prompt_options)}
                    model_chosen_id = str(option_index_to_id.get(choice_idx, "none"))
                    if model_chosen_id == "none":
                        error = "Model returned index out of bounds"
                except (ValueError, TypeError, KeyError):
                    error = "Could not parse model answer"

            new_result.update({'model_chosen_id': model_chosen_id, 'error': error})

        elif task == 'merge_identification':
            new_result['model_answer'] = answer_analysis.get('answer')

        final_results.append(new_result)

    return final_results


async def process_existing_merge_results(
    results_file_path: str,
    output_dir: str,
    model: str,
    prompt_mode: str = 'informative',
    llm_processor: LLMProcessor = None,
    K: int = 10
) -> pd.DataFrame:
    """
    Re-evaluate existing merge results with a new LLM model.

    Loads saved results, reconstructs prompts, runs new LLM evaluation, and saves updated results.

    Returns:
        DataFrame with new evaluation results
    """
    # 1. Load and validate existing results
    existing_results, task = _load_existing_results(results_file_path)
    if not existing_results:
        return pd.DataFrame()

    # 2. Reconstruct prompts from existing results
    prompts, indices, operation_mapping = _reconstruct_prompts_from_results(
        existing_results, task, llm_processor, prompt_mode, K
    )

    if not prompts:
        print("No prompts could be reconstructed from existing results.")
        return pd.DataFrame()

    print(f"Re-evaluating {len(prompts)} prompts with model: {model}")

    # 3. Run LLM evaluation
    llm_analysis = await llm_processor.process_batch(prompts)

    # 4. Map responses back to original results
    final_results = _map_responses_to_original_results(
        llm_analysis, operation_mapping, indices, existing_results, task, model, prompt_mode
    )

    print(f"Re-evaluation complete. Generated {len(final_results)} result rows.")

    # 5. Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(final_results)

    output_filename = f"{output_dir}/{model}_{task}_{prompt_mode}_analysis_results"
    if K > 1:
        output_filename += f"_K{K}"
    results_df.to_csv(output_filename + ".csv", index=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = os.path.join(output_dir, f"{task}_results_{timestamp}.json")
    results_df.to_json(json_filename, orient='records', indent=2)

    print(f"Saved results to {output_filename}.csv and {json_filename}")

    return results_df


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process merge data from FlyWire EM data JSON and evaluate using an LLM.")
    parser.add_argument("--input-json", required=False, help="Path to the input em_data_*.json file (not required when using --results-file)")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regeneration of images even if they seem to exist.")
    parser.add_argument("--num-samples", type=int, default=None, help="Process only the first N merge events found in the JSON file.")
    parser.add_argument("--use-zoomed-images", action=argparse.BooleanOptionalAction, default=True, help="Use zoomed images in the prompt instead of default.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers (threads). Defaults to CPU count or 4.")
    parser.add_argument("--views", nargs='+', choices=['front', 'side', 'top'], default=['front', 'side', 'top'], help="Specify which views to include (e.g., --views front side). Defaults to all.")
    parser.add_argument("--task", type=str, choices=['merge_comparison', 'merge_identification'], default='merge_comparison', help="Specify the evaluation task to perform.")
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
                    current_output_dir = f"output/{species}_merge_{zoom_margin}nm"

                llm_processor = LLMProcessor(model=model, max_tokens=4096, max_concurrent=25)

                # Process existing results
                results_df = asyncio.run(process_existing_merge_results(
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
            current_output_dir = f"output/{species}_merge_{zoom_margin}nm"

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

            # Process all merge events
            results_df = asyncio.run(process_merge_data(
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
