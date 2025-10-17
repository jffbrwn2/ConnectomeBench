from connectome_visualizer import ConnectomeVisualizer
from util import LLMProcessor
import pandas as pd
import os
from cloudvolume import Bbox
import random
import numpy as np
import networkx as nx
import multiprocessing
import asyncio
from src.prompts import create_segment_classification_prompt
import argparse




def _load_segments_for_classification(neuron_id, root_id, species, data_gen_mode, seed):
    visualizer = ConnectomeVisualizer(output_dir=f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}", species=species)
    if os.path.exists(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/{neuron_id}_{root_id}_front.png") and os.path.exists(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/{neuron_id}_{root_id}_side.png") and os.path.exists(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/{neuron_id}_{root_id}_top.png"):
        print(f"Skipping {neuron_id}_{root_id} because it already exists")
        return None

    visualizer.load_neurons([root_id])
    if len(visualizer.neurons) == 0:
        print(f"Skipping. Could not load neuron {neuron_id}_{root_id}")
        return None
    minpt = np.min(visualizer.neurons[0].vertices, axis = 0) 
    maxpt = np.max(visualizer.neurons[0].vertices, axis = 0)
    bbox = Bbox(minpt, maxpt, unit="nm")
    visualizer.save_3d_views(bbox=bbox, base_filename=f"{neuron_id}_{root_id}", crop=False)
    print(minpt[0], minpt[1], minpt[2], maxpt[0], maxpt[1], maxpt[2])

    return {"proofread root id": neuron_id, "current root id": root_id, "species": species, "xmin": minpt[0], "ymin": minpt[1], "zmin": minpt[2], "xmax": maxpt[0], "ymax": maxpt[1], "zmax": maxpt[2], "unit": "nm"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run segmentation classification with specified parameters')
    parser.add_argument('--data-gen-modes', nargs='+', default=["", "_first_only"],
                      help='List of data generation modes')
    parser.add_argument('--species', nargs='+', default=["fly", "mouse"],
                      help='List of species to process')
    parser.add_argument('--use-descriptions', nargs='+', type=bool, default=[False, True],
                      help='List of boolean flags for using descriptions')
    parser.add_argument('--models', nargs='+', 
                      default=['gpt-4o', "claude-3-7-sonnet-20250219", "o4-mini", "gpt-4.1"],
                      help='List of models to use')
    parser.add_argument('--num-neurons', type=int, default=200,
                      help='Number of neurons to process')
    parser.add_argument('--k', type=int, default=5,
                      help='Number of iterations per neuron')
    parser.add_argument('--seed', type=int, default=42,
                      help='Seed for random number generator')

    args = parser.parse_args()
    
    try:
        data_gen_modes = args.data_gen_modes
        all_species = args.species
        use_descriptions = args.use_descriptions
        models = args.models
        num_neurons = args.num_neurons
        K = args.k
        seed = args.seed

        for data_gen_mode in data_gen_modes:
            for species in all_species:
                for with_description in use_descriptions:
                    for model in models:

                        if data_gen_mode == "_first_only":
                            random.seed(seed)
                        else:
                            random.seed(seed)

                        os.makedirs(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}", exist_ok=True)
                        if not os.path.exists(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/results.csv"):
                            global_visualizer = ConnectomeVisualizer(output_dir=f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}", species=species)
                            if species == "mouse":
                                neuron_ids = list(global_visualizer.client.materialize.query_table('proofreading_status_and_strategy')['valid_id'])
                                
                            elif species == "fly":
                                neuron_ids = list(global_visualizer.client.materialize.query_table('proofread_neurons')['pt_root_id'])
                            neuron_ids = random.sample(neuron_ids, num_neurons)
                            # neuron_ids = neuron_ids[:50]

                            edit_history = global_visualizer.get_edit_history(neuron_ids)

                            data = []
                            for neuron_id, edit_history in edit_history.items():
                                if len(edit_history) == 0:
                                        continue
                                if data_gen_mode == "_first_only":
                                    first_first_root_id = edit_history.before_root_ids.iloc[0][0]
                                    data.append((neuron_id, first_first_root_id, species, data_gen_mode, seed))
                                    if len(edit_history.before_root_ids.iloc[0]) > 1:
                                        second_first_root_id = edit_history.before_root_ids.iloc[0][1]
                                        data.append((neuron_id, second_first_root_id, species, data_gen_mode, seed))
                                else:
                                    

                                    edges = []
                                    for x, y in zip(edit_history.before_root_ids, edit_history.after_root_ids):
                                        for i in x:
                                            for j in y:
                                                edges.append((i,j))

                                    G = nx.DiGraph()
                                    G.add_edges_from(edges)

                                    root_nodes = []
                                    for n in G.nodes:
                                        if G.in_degree(n) == 0:
                                            root_nodes.append(n)
                                    if len(root_nodes) < 3:
                                        continue
                                    random_root_ids = random.sample(root_nodes, 3)
                                    for random_root_id in random_root_ids:
                                        data.append((neuron_id, random_root_id, species, data_gen_mode, seed))


                                    first_first_root_id = edit_history.before_root_ids.iloc[0][0]
                                    data.append((neuron_id, first_first_root_id, species, data_gen_mode, seed))
                                    if len(edit_history.before_root_ids.iloc[0]) > 1:
                                        second_first_root_id = edit_history.before_root_ids.iloc[0][1]
                                        data.append((neuron_id, second_first_root_id, species, data_gen_mode, seed))
                                    last_root_id = edit_history.after_root_ids.iloc[-1][0]
                                    data.append((neuron_id, last_root_id, species, data_gen_mode, seed))

                            metadata = []
                            for d in data:
                                metadata.append({"proofread root id": d[0], "current root id": d[1], "species": d[2]})
                            metadata = pd.DataFrame(metadata)
                            metadata.to_csv(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/metadata.csv", index=False)

                            max_workers = 10
                            if max_workers is None:
                                max_workers = os.cpu_count() or 4


                            
                            with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
                                results_raw = pool.starmap(_load_segments_for_classification, data)
                            results = [r for r in results_raw if r is not None]
                            if os.path.exists(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/results.csv"):    
                                old_results = pd.read_csv(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/results.csv")
                                results = old_results + [r for r in results if r not in old_results]

                            results = pd.DataFrame(results)
                            results.to_csv(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/results.csv", index=False)
                        else:
                            results = pd.read_csv(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/results.csv")
                        
                        # model = "claude-3-7-sonnet-20250219"

                    
                        llm_processor = LLMProcessor(model=model)

                        prompts = []
                        indices = []
                        for i, result in results.iterrows():
                            proofread_root_id = result['proofread root id']
                            current_root_id = result['current root id']
                            species = result['species']
                            minpt = np.array([result['xmin'], result['ymin'], result['zmin']])
                            maxpt = np.array([result['xmax'], result['ymax'], result['zmax']])
                            segment_images_paths = [f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/{proofread_root_id}_{current_root_id}_front.png", f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/{proofread_root_id}_{current_root_id}_side.png", f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/{proofread_root_id}_{current_root_id}_top.png"]
                            segment_classification_prompt = create_segment_classification_prompt(
                                segment_images_paths,
                                minpt,
                                maxpt,
                                llm_processor,
                                species,
                                with_description
                            )
                            indices.extend([j for j in range(K)])
                            prompts.extend([segment_classification_prompt]*K)

                        llm_analysis = asyncio.run(llm_processor.process_batch(prompts))

                        final_results = []
                        for i, result in results.iterrows():
                            
                            # Extract Analysis
                            for k in range(K):
                                final_result = result.copy()
                                analysis = llm_analysis[i*K + k]
                                index = indices[i*K + k]
                                prompt = prompts[i*K + k]
                                analysis_start = analysis.find("<analysis>")
                                analysis_end = analysis.find("</analysis>")
                                if analysis_start != -1 and analysis_end != -1:
                                    final_result["analysis"] = analysis[analysis_start + len("<analysis>"):analysis_end].strip()
                                else:
                                    print("Warning: Could not find <analysis> tags in the model response.")
                                    final_result["analysis"] = "Analysis tags not found in response."

                                # Extract Answer
                                answer_start = analysis.find("<answer>")
                                answer_end = analysis.find("</answer>")
                                
                                final_result["llm_answer"] = analysis[answer_start + len("<answer>"):answer_end].strip()
                                final_result["index"] = index
                                final_result["model"] = model
                                final_result["prompt"] = prompt
                                final_result["with_description"] = with_description
                                # Save results
                                final_results.append(final_result)


                        final_results = pd.DataFrame(final_results)
                        with_description = "_without_description" if not with_description else ""
                        final_results.to_csv(f"scripts/output/{species}_segment_classification{data_gen_mode}_seed{seed}/{model}_analysis_results{with_description}_K{K}.csv", index=False)

    except:
        import pdb; pdb.post_mortem()






