import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
from connectome_visualizer import ConnectomeVisualizer
from util import LLMProcessor
import argparse
import random
import multiprocessing
import logging
import asyncio

from cloudvolume import Bbox
from prompts import create_merge_identification_prompt, create_split_identification_prompt, create_merge_comparison_prompt, create_split_comparison_prompt
logging.basicConfig(level=logging.INFO)
K = 1


def evaluate_response(response: str) -> Dict[str, Optional[str]]: # Return type changed to Dict
    """Get model's evaluation of merge identification options using LLMProcessor. 
    Returns a dictionary containing 'answer' (chosen option index as string, 'none', or error string) 
    and 'analysis' (the analysis text, or None).
    """
    result = {"answer": None, "analysis": None} # Initialize result dict

    
    # Extract Analysis
    analysis_start = response.find("<analysis>")
    analysis_end = response.find("</analysis>")
    if analysis_start != -1 and analysis_end != -1:
        result["analysis"] = response[analysis_start + len("<analysis>"):analysis_end].strip()
    else:
        print("Warning: Could not find <analysis> tags in the model response.")
        result["analysis"] = "Analysis tags not found in response."

    # Extract Answer
    answer_start = response.find("<answer>")
    answer_end = response.find("</answer>")
    if answer_start != -1 and answer_end != -1:
        answer = response[answer_start + len("<answer>"):answer_end].strip().lower()
        
        if answer == "none": # Comparison task
            result["answer"] = "none"
        elif answer == "-1": 
                result["answer"] = "none" 
                print("Info: Model responded with '-1', interpreting as 'none'.")
        else:
            try:
                # Check if it's a positive integer (for comparison task)
                choice_index = int(answer)
                if choice_index > 0:
                    result["answer"] = answer # Return the index as a string
                else:
                    print(f"Warning: Model returned non-positive integer '{answer}'. Treating as 'none'.")
                    result["answer"] = "none"
            except ValueError:
                print(f"Warning: Could not parse model answer '{answer}' as an integer or 'none'. Treating as 'none'.")
                result["answer"] = "none"
    else:
        print("Warning: Could not find <answer> tags in the model response. Treating as 'none'.")
        result["answer"] = "none" # Default if tags are missing

    
    return result

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
        
    Returns:
        Dictionary with paths to generated images
    """
    # Create output directory specific to this merge event (using base neuron ID and coords)
    # Using first 6 digits of coordinates to avoid overly long filenames
    coords_suffix = f"{int(merge_coords[0])}_{int(merge_coords[1])}_{int(merge_coords[2])}"
    neuron_dir = os.path.join(output_dir, f"merge_{base_neuron_id}_{coords_suffix}")
    os.makedirs(neuron_dir, exist_ok=True)
    
    # Use coordinates directly
    merge_x, merge_y, merge_z = merge_coords
    
    # Initialize visualizer
    # Pass timestamp if provided
    visualizer = ConnectomeVisualizer(output_dir=neuron_dir, dataset="public", timestamp=timestamp, species=species)

    merge_x_nm = merge_x 
    merge_y_nm = merge_y 
    merge_z_nm = merge_z 


    base_image_paths = {'default': {}, 'zoomed': {}, 'em': None}

    # Generate images for each option
    option_images_dict = {}
    # Filter out the base neuron ID from options if present
    option_ids_to_process = [opt_id for opt_id in option_ids if opt_id != base_neuron_id]

    # Store the initial state (base neuron and EM) to revert to

    # initial_fig = visualizer.fig # Store the figure object if possible/needed
    visualizer.clear_neurons()
    visualizer.load_neurons([int(base_neuron_id)])
    for option_id in option_ids_to_process:
        # option_id is already a string
        print(f"Processing images for option {option_id}...")
        option_img_paths = {'default': {}, 'zoomed': {}} # Initialize path dict for this option

        # Define expected filenames
        
        zoomed_option_base_filename = f"option_{option_id}_with_base_zoomed"
        # Generate expected full paths for zoomed views
        expected_zoomed_paths = {
            view: os.path.join(neuron_dir, f"{zoomed_option_base_filename}_{view}.png")
            for view in ['front', 'side', 'top'] # Assuming these are the standard views
        }
        option_em_filename = f"option_{option_id}_em_slice_with_segmentation.png"
        expected_em_path = os.path.join(neuron_dir, option_em_filename)

        # Check if all zoomed images already exist
        zoomed_exist = all(os.path.exists(p) for p in expected_zoomed_paths.values())
        if zoomed_exist:
            print(f"Found existing zoomed images for option {option_id}. Skipping generation.")
            # Store existing paths
            option_img_paths['zoomed'] = expected_zoomed_paths
        else:
            print(f"Generating zoomed images for option {option_id}...")
            # Reset visualizer state for this option
            
            try:
                visualizer.add_neurons([int(option_id)])
                # if visualizer.vol_em is not None:
                #     visualizer.load_em_data(merge_x_nm, merge_y_nm, merge_z_nm)
                
            except Exception as e:
                print(f"ERROR: Failed setup for option {option_id}: {e}. Skipping image generation.")
                continue # Skip to next option if setup fails

            # Save zoomed views
            try:
                bbox = Bbox((merge_x_nm - zoom_margin, merge_y_nm - zoom_margin, merge_z_nm - zoom_margin), (merge_x_nm + zoom_margin, merge_y_nm + zoom_margin, merge_z_nm + zoom_margin), unit="nm")
                visualizer.create_3d_neuron_figure(bbox=bbox, add_em_slice=False)
                # Pass the neuron_dir explicitly to save_3d_views if needed, or ensure it uses visualizer.output_dir
                save_3d_views_result = visualizer.save_3d_views(bbox=bbox, base_filename=zoomed_option_base_filename) # Uses visualizer.output_dir

                if save_3d_views_result is None:
                    print(f"WARNING: Zoomed option image generation potentially failed or timed out for option {option_id}.")
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
                         option_img_paths['zoomed'] = saved_paths
                    else:
                         print(f"Warning: Not all expected zoomed views were saved for option {option_id}.")

                # visualizer.reset_3d_view()
            except Exception as e:
                logging.error(f"ERROR: Failed to save zoomed option images for option {option_id}: {str(e)}")

        # Check and generate EM segmentation slice
        option_em_path = None
        if visualizer.vol_em is not None:
            if os.path.exists(expected_em_path):
                print(f"Found existing EM slice for option {option_id}. Skipping generation.")
                option_em_path = expected_em_path
            else:
                print(f"Generating EM slice for option {option_id}...")
                # Ensure correct neurons are loaded if generation didn't happen above
                if zoomed_exist: # If we skipped zoomed generation, need to load neurons for EM slice
                    visualizer.clear_neurons()
                    try:
                        visualizer.load_neurons([int(base_neuron_id), int(option_id)])
                        # EM data should still be loaded from the start of the function
                    except Exception as e:
                        print(f"ERROR: Failed setup for option {option_id} EM slice: {e}. Skipping EM generation.")
                        # Set option_em_path to None implicitly

                # Generate EM slice only if setup succeeded (or was already done)
                if visualizer.neurons:
                    try:
                        save_em_segmentation_result = visualizer.save_em_segmentation(filename=option_em_filename) # Uses visualizer.output_dir
                        if save_em_segmentation_result is None:
                            print(f"WARNING: Option EM segmentation generation potentially failed or timed out for option {option_id}.")
                        elif os.path.exists(expected_em_path):
                            option_em_path = expected_em_path
                        else:
                             print(f"Warning: Expected EM slice file missing after save: {expected_em_path}")
                    except Exception as e:
                        print(f"ERROR: Failed to save option EM segmentation for option {option_id}: {str(e)}")
        else:
            print(f"Skipping EM slice generation for option {option_id} as EM volume was not loaded.")

        # Add paths to the main dictionary if any images were found/generated
        if option_img_paths.get('zoomed') or option_img_paths.get('default'): # Check if any 3D view set exists
            option_img_paths['em'] = option_em_path # Add EM path (will be None if missing/failed)
            option_images_dict[option_id] = option_img_paths # Store using string ID
        else:
             print(f"No 3D view images were found or generated for option {option_id}. Not adding to results.")
        visualizer.remove_neurons([int(option_id)])
    # Prepare final image paths dictionary
    final_image_paths = {
        'base': base_image_paths,
        'options': option_images_dict
    }

    # Save metadata JSON
    metadata = {
        'base_neuron_id': base_neuron_id, # Already string
        'option_ids_processed': option_ids_to_process, # List of strings
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
        
    # Return paths to all generated images
    return final_image_paths

def generate_neuron_images(
    base_neuron_ids: List[str],  # Changed from base_neuron_id: str
    bbox_neuron_ids: List[str],
    merge_coords: List[float],
    output_dir: str,
    timestamp: Optional[int] = None,
    species: str = "fly",
    zoom_margin: int = 5000,
    model: str = "gpt-4o-mini"
) -> Dict[str, Dict[str, Any]]: # Return a dict mapping neuron ID to its image paths
    """
    Generate images for a list of base neurons near merge coordinates.

    Args:
        base_neuron_ids: List of string IDs for the primary neurons
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
    """Processes a single merge event: generates images and evaluates options."""
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
        print(f"Error processing merge event for operation {operation_id}: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging worker errors
        return None # Indicate failure for this item


def _process_single_merge_event(item, output_dir, force_regenerate, use_zoomed_images, views, zoom_margin, model, skip_image_generation=False):
    """Processes a single merge event: generates images and evaluates options."""
    operation_id = item.get('operation_id', 'N/A')
    try:
        base_neuron_id = str(item['before_root_ids'][0]) # Use the first pre-merge ID as base (ensure string)
        other_merged_id = str(item['before_root_ids'][1]) # The other known merged segment (ensure string)
        # Ground truth pair - convert all to string
        correct_merged_pair = {str(id) for id in item['before_root_ids'][:2]} 
        # Expected choice IDs (should be strings)
        expected_choice_ids = list(correct_merged_pair - {base_neuron_id})
        
        merge_coords = item['interface_point']
        # All unique IDs in volume - convert all to string
        all_available_option_ids = [str(id) for id in item['em_data']['all_unique_root_ids']] 

        # Timestamp just before the merge occurred
        timestamp_before_merge = item.get('prev_timestamp')

        # Define neuron directory and metadata path
        coords_suffix = f"{int(merge_coords[0])}_{int(merge_coords[1])}_{int(merge_coords[2])}"
        neuron_dir = os.path.join(output_dir, f"merge_{base_neuron_id}_{coords_suffix}")
        metadata_path = os.path.join(neuron_dir, "generation_metadata.json")

        image_paths = None
        option_ids = None # Initialize (will hold list of strings)
        should_generate = True
        MAX_OPTIONS_TOTAL = 2 # Define the target number of options
        sampled_new_incorrect = []
        should_check_images = False
        # Check for existing metadata JSON *before* sampling
        if not force_regenerate and os.path.exists(metadata_path):
            print(f"Found existing metadata: {metadata_path}")
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Check if the parameters match (comparing strings)
                if (
                    metadata.get('base_neuron_id') == base_neuron_id
                    and metadata.get('merge_coords') == merge_coords
                    and metadata.get('timestamp') == timestamp_before_merge
                    and 'option_ids_processed' in metadata # Ensure the key exists
                ):
                    print(f"Metadata matches current parameters. Loading options and skipping generation for op {operation_id}.")
                    option_ids = metadata.get('option_ids_processed') # Load the *original* sampled IDs (should be strings)
                    image_paths = metadata.get('image_paths')
                    if len(option_ids) < MAX_OPTIONS_TOTAL:
                        force_regenerate = True
                    else:
                        force_regenerate = False
                    expected_has_images = True
                    
                    # Metadata check includes validating image paths implicitly later
                    # If image paths are bad, prompt creation will fail.
                else:
                    print(f"Metadata found but parameters mismatch or incomplete. Will regenerate images and resample options for op {operation_id}.")
                    option_ids = None # Force resampling if metadata is bad
                    expected_has_images = False
                    should_check_images = True
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"Error reading or validating metadata file {metadata_path}: {e}. Will regenerate images and resample options.")
                option_ids = None # Force resampling on error
                expected_has_images = False
                should_check_images = True
        print("metadata exists", os.path.exists(metadata_path))
        print("skip_image_generation", skip_image_generation)
        if (force_regenerate or (not os.path.exists(metadata_path))) and not skip_image_generation:
            print(f"Checking for existing images and sampling options for merge op {operation_id}...")
            should_generate = True # Assume we might need to generate something
            os.makedirs(neuron_dir, exist_ok=True) # Ensure dir exists for checking

            image_set_key_for_check = 'zoomed' if use_zoomed_images else 'default'
            required_views = set(views) # Use the specified views

            # Check which potential options already have complete images
            completed_options = []
            potential_incorrect_options = list(set(all_available_option_ids) - correct_merged_pair)
            expected_has_images = True  

            # Check known correct option first
            if expected_choice_ids: # Should be exactly one expected choice
                expected_id = expected_choice_ids[0]
                option_dir_check = os.path.join(neuron_dir, f"option_{expected_id}_with_base_{image_set_key_for_check}")
                
                for view in required_views:
                    if not os.path.exists(f"{option_dir_check}_{view}.png"):
                        expected_has_images = False
                        break
                # Currently not checking EM, just the 3D views for simplicity as requested
                # if expected_has_images and not os.path.exists(os.path.join(neuron_dir, f"option_{expected_id}_em_slice_with_segmentation.png")):
                #     expected_has_images = False

                if expected_has_images:
                    print(f"Found existing complete images for correct option: {expected_id}")
                    completed_options.append(expected_id)
                else:
                     print(f"Correct option {expected_id} missing some images.")
                     # It will be added later if needed and generated
            

            # Check incorrect options pool for completed image sets
            for incorrect_id in potential_incorrect_options:
                option_dir_check = os.path.join(neuron_dir, f"option_{incorrect_id}_with_base_{image_set_key_for_check}")
                incorrect_has_images = True
                for view in required_views:
                     if not os.path.exists(f"{option_dir_check}_{view}.png"):
                        incorrect_has_images = False
                        break
                # if incorrect_has_images and not os.path.exists(os.path.join(neuron_dir, f"option_{incorrect_id}_em_slice_with_segmentation.png")):
                #    incorrect_has_images = False

                if incorrect_has_images:
                    print(f"Found existing complete images for incorrect option: {incorrect_id}")
                    completed_options.append(incorrect_id)

            # Now, build the final list of option_ids up to MAX_OPTIONS_TOTAL
            final_options = []

            # 1. Always include the correct option (if it exists)
            if expected_choice_ids:
                correct_id = expected_choice_ids[0]
                final_options.append(correct_id)
                # Remove from completed if it was there, to avoid duplicates
                if correct_id in completed_options:
                    completed_options.remove(correct_id)

            # 2. Add completed incorrect options until MAX_OPTIONS_TOTAL is reached
            num_needed = MAX_OPTIONS_TOTAL - len(final_options)
            add_completed = completed_options[:num_needed]
            final_options.extend(add_completed)

            # 3. If still needed, sample from remaining *incomplete* incorrect options
            num_still_needed = MAX_OPTIONS_TOTAL - len(final_options)
            if num_still_needed > 0:
                remaining_incorrect_pool = list(set(potential_incorrect_options) - set(completed_options))
                num_to_sample = min(len(remaining_incorrect_pool), num_still_needed)
                sampled_new_incorrect = random.sample(remaining_incorrect_pool, num_to_sample)
                final_options.extend(sampled_new_incorrect)

            option_ids = final_options # This is the final list of IDs to process/generate for
            random.shuffle(option_ids) # Shuffle the final list
            print(f"Final selected option IDs for op {operation_id}: {option_ids}")

        # --- End New Option Selection Logic ---

        # option_ids is now a list of strings (either from metadata or newly selected)

        if not option_ids:
            print(f"Warning: No unique root IDs (options) determined for merge operation {operation_id}, base neuron {base_neuron_id}. Skipping.")
            return None

        # Image generation (only if needed)
        # The generate_neuron_images function will now handle checking individual files
        if (force_regenerate or (not os.path.exists(metadata_path))) and not skip_image_generation:
            print(f"Generating images with correct and incorrect options {sampled_new_incorrect + [correct_id]}")
            image_paths = generate_neuron_option_images(
                base_neuron_id, # Pass string
                option_ids, # Pass the potentially filtered/selected list of strings
                merge_coords,
                output_dir,
                timestamp=timestamp_before_merge,
                species=item.get('species', 'fly'), # Pass species
                zoom_margin=zoom_margin,
                model=model
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
        options_with_paths = image_paths.get('options', {}) # Keys are string IDs

        for opt_id in option_ids: # option_ids is list of strings
            if opt_id == base_neuron_id:
                continue

            option_paths_dict = options_with_paths.get(opt_id) # Get using string ID
            
            # Check if images were generated/found for this option
            if option_paths_dict and ((option_paths_dict.get('default') or option_paths_dict.get('zoomed'))):
                 # Basic check: Ensure the required image set (zoomed/default) and front view exist
                 img_check_path = option_paths_dict.get(image_set_key, {}).get('front')
                 if img_check_path and os.path.exists(img_check_path):
                    prompt_options.append({
                        'id': opt_id, # String ID
                        'paths': option_paths_dict
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
            'correct_merged_pair': list(correct_merged_pair), # List of strings
            'options_presented_ids': [option_index_to_id[i] for i in sorted(option_index_to_id.keys())], # List of strings
            'expected_choice_ids': expected_choice_ids, # List of strings
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
        traceback.print_exc() # Print stack trace for debugging worker errors
        return None # Indicate failure for this item


async def process_merge_data(json_path: str, output_dir: str, force_regenerate=False, num_samples: Optional[int] = None, use_zoomed_images=True, max_workers: Optional[int] = None, views=['front', 'side', 'top'], task='merge_comparison', llm_processor: LLMProcessor = None, zoom_margin: int = 5000, species: str = "fly", model: str = "gpt-4o-mini", prompt_mode: str = 'informative'  ):
    """Process merge data from JSON and evaluate options in parallel using multiprocessing."""
    try:
        with open(json_path, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {json_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return pd.DataFrame()

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

    # Limit number of samples if specified
    if num_samples is not None and num_samples > 0:
        if num_samples < total_events_found:
            print(f"Processing only the first {num_samples} merge events.")
            merge_data = merge_data[:num_samples]
        else:
            print(f"Requested {num_samples} samples, but only {total_events_found} available. Processing all available.")
    else:
         print(f"Processing all {total_events_found} merge events.")

    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine number of workers (processes for multiprocessing)
    if max_workers is None:
        max_workers = os.cpu_count() or 4 # Default to CPU count or 4


    print(f"Using up to {max_workers} processes for parallel processing.")
    # Prepare arguments for starmap
    args_list = [
        (item, output_dir, force_regenerate, use_zoomed_images, views, zoom_margin, model)
        for item in merge_data
    ]

    # Use multiprocessing.Pool for parallel execution
    results_raw = []
    print(f"Running image generation/option preparation in parallel for {len(args_list)} events...")
    try:
        # The parallel part (_process_single_merge_event) generates images and prepares data
        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
            results_raw = pool.starmap(_process_single_merge_event, args_list)
 
    except Exception as e:
         # Catch potential errors during pool creation or execution
         print(f"An error occurred during parallel processing: {e}")
         import traceback
         traceback.print_exc()

    # Filter out None results from parallel processing (indicating errors in individual tasks)
    processed_events = [res for res in results_raw if res is not None]
    if len(processed_events) < len(results_raw):
        print(f"Warning: {len(results_raw) - len(processed_events)} merge events failed during image/option processing.")
    prompts = []
    indices = []
    total_options_processed = 0  # Add counter for total options processed
    for i, event_result in enumerate(processed_events):
        if task == 'merge_comparison':
            prompt = create_merge_comparison_prompt(
                event_result['prompt_options'], 
                use_zoomed_images=event_result['use_zoomed_images'], 
                views=event_result['views'],
                llm_processor=llm_processor,
                zoom_margin=zoom_margin,
                prompt_mode = prompt_mode
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
                total_options_processed += 1  # Increment counter for each option

    llm_analysis = await llm_processor.process_batch(prompts)

    final_results = []
    total_options_processed = 0  # Reset counter for results processing
    for i, event_result in enumerate(processed_events):
        if task == 'merge_comparison':
            for k in range(K):
                response = llm_analysis[i*K + k]
                answer_analysis = evaluate_response(response)
                index = indices[i*K + k]

                # Determine model chosen ID
                model_chosen_id = "none"
                error = None
                if answer_analysis["answer"] != "none":
                    if int(answer_analysis["answer"]) in event_result['option_index_to_id']:
                        model_chosen_id = str(event_result['option_index_to_id'][int(answer_analysis["answer"])])
                    else:
                        model_chosen_id = "none"
                        error = "Model returned index out of bounds"
                else:
                    model_chosen_id = "none"

                # Create unified result structure
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
                
                # Add task-specific fields
                unified_result.update({
                    'model_chosen_id': model_chosen_id,
                    'error': error
                })
                
                final_results.append(unified_result)
                
        elif task == 'merge_identification':
            for j, option_data in enumerate(event_result['prompt_options']):
                for k in range(K):
                    response = llm_analysis[total_options_processed*K + k]
                    answer_analysis = evaluate_response(response)
                    index = indices[total_options_processed*K + k]

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
                total_options_processed += 1  # Increment counter after processing all K responses for this option

    print(f"LLM evaluation complete. Generated {len(final_results)} result rows.")
    final_results = pd.DataFrame(final_results)
    if K > 1:
        final_results.to_csv(f"{output_dir}/{model}_{task}_{prompt_mode}_analysis_results_K{K}.csv", index=False)
    else:
        final_results.to_csv(f"{output_dir}/{model}_{task}_{prompt_mode}_analysis_results.csv", index=False)
    return final_results

async def process_split_data(json_path: str, output_dir: str, force_regenerate=False, num_samples: Optional[int] = None, use_zoomed_images=True, max_workers: Optional[int] = None, views=['front', 'side', 'top'], task='merge_comparison', prompt_mode = 'informative',llm_processor: LLMProcessor = None, zoom_margin: int = 5000, species: str = "fly", model: str = "gpt-4o-mini"  ):
    """Process merge data from JSON and evaluate options in parallel using multiprocessing."""
    try:
        with open(json_path, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {json_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return pd.DataFrame()

    # Filter for merge operations with necessary data
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
    print(f"Found {total_events_found} merge events in the input file.")

    # Limit number of samples if specified
    if num_samples is not None and num_samples > 0:
        if num_samples < total_events_found:
            print(f"Processing only the first {num_samples} merge events.")
            split_data = split_data[:num_samples]
        else:
            print(f"Requested {num_samples} samples, but only {total_events_found} available. Processing all available.")
    else:
         print(f"Processing all {total_events_found} merge events.")

    # results = [] # Will be populated by pool.starmap

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
        # The parallel part (_process_single_merge_event) generates images and prepares data
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
        print(f"Warning: {len(results_raw) - len(processed_events)} merge events failed during image/option processing.")

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

def create_unified_result_structure(
    task: str,
    event_result: Dict[str, Any],
    option_data: Optional[Dict[str, Any]] = None,
    response: Optional[str] = None,
    answer_analysis: Optional[Dict[str, Any]] = None,
    index: Optional[int] = None,
    model: str = "unknown",
    zoom_margin: int = 5000,
    prompt_mode: str = "informative",
    correct_answer: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a unified result structure for all tasks with consistent keys.
    
    Args:
        task: The task type ('merge_comparison', 'merge_identification', 'split_comparison', 'split_identification')
        event_result: The processed event result containing operation details
        option_data: Optional option data for identification tasks
        response: Optional raw LLM response
        answer_analysis: Optional parsed answer analysis
        index: Optional index for multiple runs
        model: Model name used
        zoom_margin: Zoom margin used
        prompt_mode: Prompt mode used
        correct_answer: Optional correct answer for comparison tasks
        
    Returns:
        Dictionary with unified structure for all tasks
    """
    # Base structure with common fields
    unified_result = {
        # Task and operation info
        'task': task,
        'operation_id': event_result.get('operation_id', 'unknown'),
        'timestamp': event_result.get('timestamp', None),
        
        # Coordinates and location
        'merge_coords': event_result.get('merge_coords', None),
        'interface_point': event_result.get('interface_point', None),
        
        # Neuron IDs - always present but may be None for some tasks
        'base_neuron_id': event_result.get('base_neuron_id', None),
        'before_root_ids': event_result.get('before_root_ids', []),
        'after_root_ids': event_result.get('after_root_ids', []),
        'proofread_root_id': event_result.get('proofread_root_id', None),
        
        # Model and evaluation info
        'model': model,
        'model_raw_answer': response,
        'model_analysis': answer_analysis.get('analysis', None) if answer_analysis else None,
        'model_prediction': answer_analysis.get('answer', None) if answer_analysis else None,
        'index': index,
        
        # Image and view settings
        'views': event_result.get('views', []),
        'use_zoomed_images': event_result.get('use_zoomed_images', True),
        'zoom_margin': zoom_margin,
        'prompt_mode': prompt_mode,
        
        # Task-specific fields (will be filled based on task)
        'correct_answer': correct_answer,
        'is_split': None,
        'model_chosen_id': None,
        'model_answer': None,
        'error': None,
        
        # Image paths (if available)
        'image_paths': event_result.get('image_paths', {}),
        'prompt_options': event_result.get('prompt_options', [])
    }
    
    # Task-specific field mapping
    if task == 'merge_comparison':
        unified_result.update({
            'correct_answer': event_result.get('expected_choice_ids', []),
            'model_chosen_id': event_result.get('model_chosen_id', None),
            'error': event_result.get('error', None),
            'options_presented_ids': event_result.get('options_presented_ids', []),
            'num_options_presented': event_result.get('num_options_presented', 0),
            'correct_merged_pair': event_result.get('correct_merged_pair', [])
        })
        
    elif task == 'merge_identification':
        if option_data:
            unified_result.update({
                'id': option_data.get('id', None),
                'model_answer': answer_analysis.get('answer', None) if answer_analysis else None,
                'is_correct_merge': option_data.get('id') in event_result.get('expected_choice_ids', [])
            })
            
    elif task == 'split_identification':
        if option_data:
            unified_result.update({
                'id': option_data.get('id', None),
                'is_split': int(option_data.get('id', 0)) in event_result.get('before_root_ids', []),
                'model_answer': answer_analysis.get('answer', None) if answer_analysis else None
            })
            
    elif task == 'split_comparison':
        unified_result.update({
            'root_id_requires_split': event_result.get('root_id_requires_split', None),
            'root_id_does_not_require_split': event_result.get('root_id_does_not_require_split', None),
            'correct_answer': correct_answer
        })
    
    return unified_result

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process merge data from FlyWire EM data JSON and evaluate using an LLM.")
    parser.add_argument("--input-json", required=True, help="Path to the input em_data_*.json file generated by gather_training_data.py")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regeneration of images even if they seem to exist.")
    parser.add_argument("--num-samples", type=int, default=None, help="Process only the first N merge events found in the JSON file.")
    parser.add_argument("--use-zoomed-images", action=argparse.BooleanOptionalAction, default=True, help="Use zoomed images in the prompt instead of default.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers (threads). Defaults to CPU count or 4.")
    parser.add_argument("--views", nargs='+', choices=['front', 'side', 'top'], default=['front', 'side', 'top'], help="Specify which views to include (e.g., --views front side). Defaults to all.")
    parser.add_argument("--task", type=str, choices=['merge_comparison', 'merge_identification', 'split_identification', 'split_comparison'], default='merge_comparison', help="Specify the evaluation task to perform.")
    parser.add_argument("--species", type=str, choices=['mouse', 'fly'], default='mouse', help="Specify the species to use for the output directory.")
    parser.add_argument("--zoom-margin", type=int, default=1024, help="Specify the zoom margin to use for the output directory.")
    parser.add_argument("--models", nargs='+', default=["claude-3-7-sonnet-20250219"], help="Specify one or more models to use for evaluation.")
    parser.add_argument("--prompt-modes", nargs='+', choices=['informative', 'null'], default=['informative'], help="Specify one or more prompt modes to use for evaluation.")
    args = parser.parse_args()

    json_path = args.input_json
    
    force_regenerate = args.force_regenerate
    num_samples = args.num_samples
    use_zoomed = args.use_zoomed_images # Get zoom preference
    max_workers = args.max_workers # Get max_workers preference
    selected_views = args.views # Get the list of selected views
    task = args.task # Get the selected task
    species = args.species # Get the selected species
    zoom_margin = args.zoom_margin # Get the selected zoom margin
    models = args.models # Get the list of models
    prompt_modes = args.prompt_modes # Get the list of prompt modes

    # Process each combination of model and prompt mode
    for model in models:
        for prompt_mode in prompt_modes:
            print(f"\nProcessing with model: {model} and prompt mode: {prompt_mode}")
            
            if task == 'merge_identification' or task == 'merge_comparison':
                current_output_dir = f"output/{species}_merge_{zoom_margin}nm"
            elif task == 'split_identification' or task == 'split_comparison':
                current_output_dir = f"output/{species}_split"
            
            llm_processor = LLMProcessor(model=model, max_tokens=4096, max_concurrent=10)

            # Validate input path
            if not os.path.exists(json_path):
                print(f"Error: Input JSON file not found at {json_path}")
                continue

            print(f"Selected Task: {task}") # Print selected task
            print(f"Using input JSON: {json_path}")
            print(f"Output directory: {current_output_dir}")
            print(f"Force regenerate images: {force_regenerate}")
            print(f"Using zoomed images: {use_zoomed}") # Print zoom status
            print(f"Selected views: {selected_views}") # Print selected views
            if num_samples is not None:
                print(f"Number of samples to process: {num_samples}")
            if max_workers is not None:
                print(f"Max workers specified: {max_workers}")

            # Create output directory
            os.makedirs(current_output_dir, exist_ok=True)

            # Process all merge events based on the selected task
            if task == "merge_comparison" or task == "merge_identification":
                results_df = asyncio.run(process_merge_data(
                    json_path,
                    current_output_dir,
                    task=task, # Pass the task
                    force_regenerate=force_regenerate,
                    num_samples=num_samples,
                    use_zoomed_images=use_zoomed, # Pass zoom preference
                    max_workers=max_workers, # Pass max_workers preference
                    views=selected_views, # Pass selected views
                    llm_processor=llm_processor,
                    zoom_margin=zoom_margin, 
                    species=species,
                    model=model,
                    prompt_mode=prompt_mode
                ))
            elif task == "split_identification" or task == "split_comparison":
                # Process all merge events based on the selected task
                results_df = asyncio.run(process_split_data(
                    json_path,
                    current_output_dir,
                    task=task, # Pass the task
                    force_regenerate=force_regenerate,
                    num_samples=num_samples,
                    use_zoomed_images=use_zoomed, # Pass zoom preference
                    max_workers=max_workers, # Pass max_workers preference
                    views=selected_views, # Pass selected views
                    llm_processor=llm_processor,
                    zoom_margin=zoom_margin,
                    species=species,
                    model=model,
                    prompt_mode=prompt_mode
                ))
            else:
                print(f"Unknown task '{task}'. No data processed.")
                continue

            if results_df.empty:
                print("No results generated.")
                continue

            # --- Results Saving and Metrics ---
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