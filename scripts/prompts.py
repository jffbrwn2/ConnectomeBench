from typing import List, Dict, Any, Tuple
from util import LLMProcessor, openai_models
import os
import numpy as np

def create_merge_identification_prompt(
    option_data: Dict[str, Any], # Each dict contains 'id' and 'paths' (nested dict with default/zoomed -> front/side/top)
    use_zoomed_images: bool = True,
    views: List[str] = ['front', 'side', 'top'], # Re-add views parameter
    llm_processor: LLMProcessor = None,
    zoom_margin: int = 5000,
    prompt_mode: str = 'informative'
) -> List[dict]:
    """
    Create a prompt for evaluating split errors using multiple views.
    
    Args:
        option_image_data: List of dictionaries, each containing 'id' (str) and
                           'paths' (Dict[str, Dict[str, str]]) for an option's images
                           (e.g., {'default': {'front': 'path', 'side': 'path', 'top': 'path'},
                                   'zoomed': {'front': 'path', 'side': 'path', 'top': 'path'}}).
        merge_coords: Coordinates [x, y, z] of the merge point.
        use_zoomed_images: If True, use zoomed images; otherwise, use default images.
        views: List of views (e.g., ['front', 'side']) to include in the prompt.
       
    Returns:
        List of content blocks for the LLM prompt.
    """
    content = []
    image_set_key = 'zoomed' if use_zoomed_images else 'default'
    # view_keys = ['front', 'side', 'top'] # Use passed views instead
    text_type = "text" if llm_processor.model not in openai_models else "input_text"
    image_type = "image" if llm_processor.model not in openai_models else "input_image"
    # --- 1. Add all images first ---
    # breakpoint()
    option_details = [] # Store details for text part
    # for i, option_data in enumerate(option_image_data, 1):

    option_id = option_data['id']
    image_paths = option_data['paths']
    
    current_option_images = []
    valid_image_found = False

    content.append({
        "type": text_type,
        "text": f"Segment ID {option_id}"
    })
    for view in views: # Use the provided views list
        img_path = image_paths.get(image_set_key, {}).get(view)
        if img_path and os.path.exists(img_path):
            try:
                base64_data, media_type = llm_processor._encode_image_to_base64(img_path)
                content.append({
                    "type": text_type,
                    "text": f"Views shown: {view}"
                })
    
                if llm_processor.model not in openai_models:
                    content.append({
                        "type": image_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data
                        }
                    })
                else:
                    content.append({
                        "type": image_type,
                        "image_url": f"data:{media_type};base64,{base64_data}",
                        "detail": "high"
                    })

                current_option_images.append(f"{view} view") # Track which images were added
                valid_image_found = True
            except FileNotFoundError:
                print(f"Warning: Image file not found at {img_path} for option {option_id}, view {view}. Skipping this view.")
            except Exception as e:
                print(f"Error encoding image {img_path} for option {option_id}, view {view}: {e}. Skipping this view.")
        else:
            print(f"Warning: Image path for option {option_id}, set '{image_set_key}', view '{view}' is missing or file doesn't exist. Skipping this view.")
            current_option_images.append(f"{view} view (missing)")

    if valid_image_found:
        option_details.append({
            'id': option_id,
            'views_added': ", ".join(current_option_images) 
        })
    else:
            print(f"Warning: No valid {image_set_key} images found for option {option_id}. Skipping option.")

    if not option_details:
        print("Error: No valid option images could be loaded for the prompt.")
        return []
        
    # --- 2. Add Text Description ---
    image_description = f"The {image_set_key} images" if len(option_details) > 1 else f"The {image_set_key} image"
    view_description = ", ".join(views) # Use the provided views list
    prompt =f"""You are an expert in analyzing neuronal morphology and split and merge errors in connectomics data.

The previous images show a proposed merge operation at the center of the 3D volume. The original segment is blue and a potential merge candidate segment is orange. {image_description} below show this pair from the {view_description} perspectives.
The image is a cropped 3D volume ({2*zoom_margin} nm x {2*zoom_margin} nm x {2*zoom_margin} nm around the center of the volume), so you should pay attention to discontinuities in the center of the image.
Images are presented in groups of {len(views)} ({view_description})"""
    if prompt_mode == 'informative':
        prompt += f"""The segments merged together should look like a continuous single axon, where the orange segment is progressing in the same direction as the blue segment was progressing.
They should just join together at the center; they shouldn't be overlapping."""

    prompt+= """If there is a split error and the proposed merge operation fixes it (the segments merged together look like a continuous single axon), then return 1.
If there is no split error OR the merge operation is incorrect, then return -1.

Surround your analysis with <analysis> and </analysis> tags.
Surround your final answer (the number or "-1") with <answer> and </answer> tags.""" 
    content.append({
        "type": text_type,
        "text": prompt
    })

    
    return content

def create_split_identification_prompt(
    option_data: Dict[str, Any], # Each dict contains 'id' and 'paths' (nested dict with default/zoomed -> front/side/top)
    use_zoomed_images: bool = True,
    views: List[str] = ['front', 'side', 'top'], # Re-add views parameter
    llm_processor: LLMProcessor = None,
    zoom_margin: int = 5000,
    prompt_mode: str = 'informative'
) -> List[dict]:
    """
    Create a prompt for evaluating split errors using multiple views.
    
    Args:
        option_image_data: List of dictionaries, each containing 'id' (str) and
                           'paths' (Dict[str, Dict[str, str]]) for an option's images
                           (e.g., {'default': {'front': 'path', 'side': 'path', 'top': 'path'},
                                   'zoomed': {'front': 'path', 'side': 'path', 'top': 'path'}}).
        merge_coords: Coordinates [x, y, z] of the merge point.
        use_zoomed_images: If True, use zoomed images; otherwise, use default images.
        views: List of views (e.g., ['front', 'side']) to include in the prompt.
       
    Returns:
        List of content blocks for the LLM prompt.
    """
    content = []
    image_set_key = 'zoomed' if use_zoomed_images else 'default'
    # view_keys = ['front', 'side', 'top'] # Use passed views instead
    text_type = "text" if llm_processor.model not in openai_models else "input_text"
    image_type = "image" if llm_processor.model not in openai_models else "input_image"
    # --- 1. Add all images first ---
    # breakpoint()
    option_details = [] # Store details for text part
    # for i, option_data in enumerate(option_image_data, 1):
    option_id = option_data['id']
    image_paths = option_data['paths']
    
    current_option_images = []
    valid_image_found = False

    content.append({
        "type": text_type,
        "text": f"Segment ID {option_id}"
    })
    for view in views: # Use the provided views list
        img_path = image_paths.get(image_set_key, {}).get(view)
        if img_path and os.path.exists(img_path):
            try:
                base64_data, media_type = llm_processor._encode_image_to_base64(img_path)
                content.append({
                    "type": text_type,
                    "text": f"Views shown: {view}"
                })
    
                if llm_processor.model not in openai_models:
                    content.append({
                        "type": image_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data
                        }
                    })
                else:
                    content.append({
                        "type": image_type,
                        "image_url": f"data:{media_type};base64,{base64_data}",
                        "detail": "high"
                    })

                current_option_images.append(f"{view} view") # Track which images were added
                valid_image_found = True
            except FileNotFoundError:
                print(f"Warning: Image file not found at {img_path} for option {option_id}, view {view}. Skipping this view.")
            except Exception as e:
                print(f"Error encoding image {img_path} for option {option_id}, view {view}: {e}. Skipping this view.")
        else:
            print(f"Warning: Image path for option {option_id}, set '{image_set_key}', view '{view}' is missing or file doesn't exist. Skipping this view.")
            current_option_images.append(f"{view} view (missing)")

    if valid_image_found:
        option_details.append({
            'id': option_id,
            'views_added': ", ".join(current_option_images) 
        })
    else:
            print(f"Warning: No valid {image_set_key} images found for option {option_id}. Skipping option.")

    if not option_details:
        print("Error: No valid option images could be loaded for the prompt.")
        return []
        
    # --- 2. Add Text Description ---
    image_description = f"The {image_set_key} images" if len(option_details) > 1 else f"The {image_set_key} image"
    view_description = ", ".join(views) # Use the provided views list
    prompt = f"""You are an expert in analyzing neuronal morphology and split and merge errors in connectomics data.

The previous images show a portion of 3D segmentation of neuronal data. While it's intended that the segment all correspond to processes (axon, dendrites) of a single neuron, it's possible that the algorithm may have introduced merge errors, inappropriately grouping processes from different neurons together.  """
    if prompt_mode == 'informative':
        prompt += f"""Merge errors are often characterized by aberrant axonal structure like the axon doubling back after branching off or an axon forming a ninety degree angle when joined with another."""

    prompt += f"""{image_description} show this segment from the {view_description} perspectives.
The image is a cropped 3D volume ({2*zoom_margin} nm x {2*zoom_margin} nm x {2*zoom_margin} nm around the center of the volume), so you should pay attention to merges in the center of the image.
Images are presented in groups of {len(views)} ({view_description}).""" # Update description dynamically
    
    content.append({
        "type": text_type,
        "text": prompt
    })

    # --- 4. Add Final Instructions ---
    content.append({
        "type": text_type,
        "text": """
        
If there is a merge error and the segment should be split apart, then return 1.
If there is no merge error, then return -1.

Surround your analysis with <analysis> and </analysis> tags.
Surround your final answer (the number or "-1") with <answer> and </answer> tags.
"""
    })
    
    return content

def create_split_comparison_prompt(
    positive_option_data: Dict[str, Any], # Each dict contains 'id' and 'paths' (nested dict with default/zoomed -> front/side/top)
    negative_option_data: Dict[str, Any], # Each dict contains 'id' and 'paths' (nested dict with default/zoomed -> front/side/top)
    use_zoomed_images: bool = True,
    views: List[str] = ['front', 'side', 'top'], # Re-add views parameter
    llm_processor: LLMProcessor = None,
    zoom_margin: int = 5000,
    prompt_mode: str = 'informative'
) -> List[dict]:
    """
    Create a prompt for evaluating split errors using multiple views.
    
    Args:
        positive_option_data: 
        merge_coords: Coordinates [x, y, z] of the merge point.
        use_zoomed_images: If True, use zoomed images; otherwise, use default images.
        views: List of views (e.g., ['front', 'side']) to include in the prompt.
       
    Returns:
        List of content blocks for the LLM prompt.
    """
    content = []
    image_set_key = 'zoomed' if use_zoomed_images else 'default'
    # view_keys = ['front', 'side', 'top'] # Use passed views instead
    text_type = "text" if llm_processor.model not in openai_models else "input_text"
    image_type = "image" if llm_processor.model not in openai_models else "input_image"
    # --- 1. Add all images first ---
    # breakpoint()
    option_details = [] # Store details for text part
    # for i, option_data in enumerate(option_image_data, 1):
    positive_option_id = positive_option_data['id']
    negative_option_id = negative_option_data['id']
    positive_image_paths = positive_option_data['paths']
    negative_image_paths = negative_option_data['paths']
    
    current_option_images = []
    valid_image_found = False

    content.append({
        "type": text_type,
        "text": f"Segment ID {positive_option_id}"
    })
    for view in views: # Use the provided views list
        img_path = positive_image_paths.get(image_set_key, {}).get(view)
        if img_path and os.path.exists(img_path):
            try:
                base64_data, media_type = llm_processor._encode_image_to_base64(img_path)
                content.append({
                    "type": text_type,
                    "text": f"Views shown: {view}"
                })
    
                if llm_processor.model not in openai_models:
                    content.append({
                        "type": image_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data
                        }
                    })
                else:
                    content.append({
                        "type": image_type,
                        "image_url": f"data:{media_type};base64,{base64_data}",
                        "detail": "high"
                    })

                current_option_images.append(f"{view} view") # Track which images were added
                valid_image_found = True
            except FileNotFoundError:
                print(f"Warning: Image file not found at {img_path} for option {positive_option_id}, view {view}. Skipping this view.")
            except Exception as e:
                print(f"Error encoding image {img_path} for option {positive_option_id}, view {view}: {e}. Skipping this view.")
        else:
            print(f"Warning: Image path for option {positive_option_id}, set '{image_set_key}', view '{view}' is missing or file doesn't exist. Skipping this view.")
            current_option_images.append(f"{view} view (missing)")

    if valid_image_found:
        option_details.append({
            'id': positive_option_id,
            'views_added': ", ".join(current_option_images) 
        })
    else:
        print(f"Warning: No valid {image_set_key} images found for option {positive_option_id}. Skipping option.")

    if not option_details:
        print("Error: No valid option images could be loaded for the prompt.")
        return []
    content.append({
        "type": text_type,
        "text": f"Segment ID {negative_option_id}"
    })
    for view in views: # Use the provided views list
        img_path = negative_image_paths.get(image_set_key, {}).get(view)
        if img_path and os.path.exists(img_path):
            try:
                base64_data, media_type = llm_processor._encode_image_to_base64(img_path)
                content.append({
                    "type": text_type,
                    "text": f"Views shown: {view}"
                })
    
                if llm_processor.model not in openai_models:
                    content.append({
                        "type": image_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data
                        }
                    })
                else:
                    content.append({
                        "type": image_type,
                        "image_url": f"data:{media_type};base64,{base64_data}",
                        "detail": "high"
                    })

                current_option_images.append(f"{view} view") # Track which images were added
                valid_image_found = True
            except FileNotFoundError:
                print(f"Warning: Image file not found at {img_path} for option {negative_option_id}, view {view}. Skipping this view.")
            except Exception as e:
                print(f"Error encoding image {img_path} for option {negative_option_id}, view {view}: {e}. Skipping this view.")
        else:
            print(f"Warning: Image path for option {negative_option_id}, set '{image_set_key}', view '{view}' is missing or file doesn't exist. Skipping this view.")
            current_option_images.append(f"{view} view (missing)")

    if valid_image_found:
        option_details.append({
            'id': negative_option_id,
            'views_added': ", ".join(current_option_images) 
        })
    else:
        print(f"Warning: No valid {image_set_key} images found for option {negative_option_id}. Skipping option.")


    # --- 2. Add Text Description ---
    image_description = f"The {image_set_key} images" if len(option_details) > 1 else f"The {image_set_key} image"
    view_description = ", ".join(views) # Use the provided views list
    prompt = f"""You are an expert in analyzing neuronal morphology and split and merge errors in connectomics data.

The previous images show two examples of a portion of 3D segmentation of neuronal data. One or neither of the segmentations may have evident axonal merge errors --- inappropriately grouped processes from different neurons."""
    if prompt_mode == "informative":
        prompt += f"""Merge errors are often characterized by aberrant axonal structure like the axon doubling back after branching off or an axon forming a ninety degree angle when joined with another."""
    prompt += f"""{image_description} show this segment from the {view_description} perspectives.
The image is a cropped 3D volume ({2*zoom_margin} nm x {2*zoom_margin} nm x {2*zoom_margin} nm around the center of the volume), so you should pay attention to merges in the center of the image.
Images are presented in groups of {len(views)} ({view_description})."""
    content.append({
        "type": text_type,
        "text": prompt # Update description dynamically
    })

    # --- 4. Add Final Instructions ---
    content.append({
        "type": text_type,
        "text": """
        
Pick the number (e.g., "1", "2", etc.) of the single best option that has a merge error.
If none of the options show segments that have merge errors, respond with "-1".

Surround your analysis with <analysis> and </analysis> tags.
Surround your final answer (the number or "-1") with <answer> and </answer> tags.
"""
    })
    
    return content


def create_merge_comparison_prompt(
    option_image_data: List[Dict[str, Any]], # Each dict contains 'id' and 'paths' (nested dict with default/zoomed -> front/side/top)
    use_zoomed_images: bool = True,
    views: List[str] = ['front', 'side', 'top'], # Re-add views parameter
    llm_processor: LLMProcessor = None,
    zoom_margin: int = 5000,
    prompt_mode: str = 'informative'
) -> List[dict]:
    """
    Create a prompt for evaluating merge identification options using multiple views.
    
    Args:
        option_image_data: List of dictionaries, each containing 'id' (str) and
                           'paths' (Dict[str, Dict[str, str]]) for an option's images
                           (e.g., {'default': {'front': 'path', 'side': 'path', 'top': 'path'},
                                   'zoomed': {'front': 'path', 'side': 'path', 'top': 'path'}}).
        merge_coords: Coordinates [x, y, z] of the merge point.
        use_zoomed_images: If True, use zoomed images; otherwise, use default images.
        views: List of views (e.g., ['front', 'side']) to include in the prompt.
       
    Returns:
        List of content blocks for the LLM prompt.
    """
    content = []
    image_set_key = 'zoomed' if use_zoomed_images else 'default'
    # view_keys = ['front', 'side', 'top'] # Use passed views instead
    text_type = "text" if llm_processor.model not in openai_models else "input_text"
    image_type = "image" if llm_processor.model not in openai_models else "input_image"
    # --- 1. Add all images first ---
    # breakpoint()
    option_details = [] # Store details for text part
    for i, option_data in enumerate(option_image_data, 1):
        option_id = option_data['id']
        image_paths = option_data['paths']
        
        if image_set_key not in image_paths:
            print(f"Warning: Required image set '{image_set_key}' not found for option {option_id}. Skipping.")
            continue
            
        current_option_images = []
        valid_image_found = False

        content.append({
            "type": text_type,
            "text": f"{i}. Option ID {option_id}"
        })
        for view in views: # Use the provided views list
            img_path = image_paths.get(image_set_key, {}).get(view)
            if img_path and os.path.exists(img_path):
                try:
                    base64_data, media_type = llm_processor._encode_image_to_base64(img_path)
                    content.append({
                        "type": text_type,
                        "text": f"Views shown: {view}"
                    })
        
                    if llm_processor.model not in openai_models:
                        content.append({
                            "type": image_type,
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data
                            }
                        })
                    else:
                        content.append({
                            "type": image_type,
                            "image_url": f"data:{media_type};base64,{base64_data}",
                            "detail": "high"
                        })

                    current_option_images.append(f"{view} view") # Track which images were added
                    valid_image_found = True
                except FileNotFoundError:
                    print(f"Warning: Image file not found at {img_path} for option {option_id}, view {view}. Skipping this view.")
                except Exception as e:
                    print(f"Error encoding image {img_path} for option {option_id}, view {view}: {e}. Skipping this view.")
            else:
                print(f"Warning: Image path for option {option_id}, set '{image_set_key}', view '{view}' is missing or file doesn't exist. Skipping this view.")
                current_option_images.append(f"{view} view (missing)")

        if valid_image_found:
            option_details.append({
                'index': i,
                'id': option_id,
                'views_added': ", ".join(current_option_images) 
            })
        else:
             print(f"Warning: No valid {image_set_key} images found for option {option_id}. Skipping option.")

    if not option_details:
        print("Error: No valid option images could be loaded for the prompt.")
        return []
        
    # --- 2. Add Text Description ---
    image_description = f"The {image_set_key} images" if len(option_details) > 1 else f"The {image_set_key} image"
    view_description = ", ".join(views) # Use the provided views list
    prompt = f"""You are an expert in analyzing neuronal morphology and merge errors in connectomics data.

The previous images show a potential merge of a split error at the center of the 3D volume. Each option displays a pair of segments: the original segment (blue) and a potential merge candidate segment (orange). {image_description} below show this pair from the {view_description} perspectives for each option.
The image is a cropped 3D volume ({2*zoom_margin} nm x {2*zoom_margin} nm x {2*zoom_margin} nm around the center of the volume), so you should pay attention to discontinuities in the center of the image.
Images are presented in groups of {len(views)} ({view_description}) per option. The first group corresponds to Option 1, the second to Option 2, and so on."""
    if prompt_mode == 'informative':
        prompt += f"""The segments merged together should look like a continuous single axon.
They should just join together at the center; they shouldn't be overlapping."""

    prompt += f"""Pick the number (e.g., "1", "2", etc.) of the single best option that represents the correct merge.
If none of the options show segments that should be merged, respond with "-1".

Surround your analysis with <analysis> and </analysis> tags.
Surround your final answer (the number or "-1") with <answer> and </answer> tags.
""" 
    content.append({
        "type": text_type,
        "text": prompt
    })
    
    return content


def create_segment_classification_prompt(
    segment_images_paths: List[str],
    minpt: Tuple[float, float, float],
    maxpt: Tuple[float, float, float],
    llm_processor: LLMProcessor = None,
    species: str = "fly",
    add_guidance: bool = False
) -> List[dict]:
    """
    Create a prompt for evaluating split errors using multiple views.
    
    Args:

    Returns:
        List of content blocks for the LLM prompt.
    """
    content = []

    # view_keys = ['front', 'side', 'top'] # Use passed views instead
    text_type = "text" if llm_processor.model not in openai_models else "input_text"
    image_type = "image" if llm_processor.model not in openai_models else "input_image"
    # --- 1. Add all images first ---
    
    current_option_images = []
    valid_image_found = False

    for img_path in segment_images_paths:
        try:
            base64_data, media_type = llm_processor._encode_image_to_base64(img_path)
    
            if llm_processor.model not in openai_models:
                content.append({
                    "type": image_type,
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data
                    }
                })
            else:
                content.append({
                    "type": image_type,
                    "image_url": f"data:{media_type};base64,{base64_data}",
                    "detail": "high"
                })


            valid_image_found = True
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}")
        except Exception as e:
            print(f"Error encoding image {img_path}: {e}. Skipping this image.")
 
    box_size = np.array(maxpt) - np.array(minpt)
    # --- 4. Add Final Instructions ---

    prompt = f"""
You are an expert at analyzing neuronal morphology. 

We have the electron microscopy data from the {species} brain. 

In the images, we have a selected 3D segmentation that is supposed to correspond to a complete neuronal structure. However, it could have split/merge errors as the segmentation algorithm makes mistakes.

The 3D snapshots are three different views of the same segment. The dimensions of the segment's bounding box are {box_size[0]} x {box_size[1]} x {box_size[2]} nm. Describe in detail what you see using the information in the 3D snapshots. Is the segment a neuron (soma and processes)? Multiple neurons merged together (multiple somas)? Processes like axon and dendrites without a cell body? Non-neuronal structures like glia, astrocytes, or blood vessels? Inspect very closely to avoid making errors, using the 3D views and size of the bounding box in your reasoning."""

    if add_guidance:
        if species == "fly":
            prompt += """For fly neurons, the somas tend to be round and generally a single process extends from them before it branches into many processes. Processes can be axons or dendrite, long and often branching. Synapses can also be considered as a part of  processes, and these are often small segments (often smaller than a cubic micron). The nucleuses are round and do not have any processes extending from them. Blood vessels are tubular and obviously do not have any processes extending from them. Glial cells lack the branching processes of neurons, and instead appear like jagged masses. 
        """
        elif species == "mouse":
            prompt += """For mouse neurons, the somas tend to be round and generally a multiple processes extend from them outwards. Processes can be axons or dendrite, long and often branching. Synapses can also be considered as a part of  processes, and these are often small segments (often smaller than a cubic micron). The nucleuses are round and do not have any processes extending from them. Blood vessels are tubular and obviously do not have any processes extending from them. Glial cells lack the branching processes of neurons, and instead appear like jagged masses. 
        """
        else:
            raise ValueError(f"Species {species} not supported")

    prompt += """
    Choose the best answer: 
    a) A single soma and process(es).
    b) Multiple somas (and processes)
    c) Processes without a soma. These can be axons, dendrites, synapses.
    d) Nucleus. 
    e) Non-neuronal types. These can be glial cells, blood vessels.
    f) None of the above.
    g) Unsure

    Surround your analysis with <analysis> and </analysis> tags.
    Surround your final answer (the letter a, b, c, d, e, f, or g) with <answer> and </answer> tags.
    """
    content.append({
        "type": text_type,
        "text": prompt
    })
    
    return content
