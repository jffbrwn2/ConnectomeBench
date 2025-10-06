#!/usr/bin/env python3
"""
Combined script to load segments and annotate nucleus presence.
"""

from load_and_plot_segments import load_and_plot_segments
from nucleus_annotation_web import annotate_nucleus_presence
import argparse


def main():
    parser = argparse.ArgumentParser(description="Load neuron segments and annotate nucleus presence")
    parser.add_argument("--num-entries", type=int, default=5,
                        help="Number of entries to load (default: 5)")
    parser.add_argument("--json-path", type=str,
                        default="/Users/jbrown/Documents/boyden_lab/ai-proofreading/connectomebench/scripts/training_data/merge_error_only_mouse.json",
                        help="Path to input JSON file")
    parser.add_argument("--output-dir", type=str, default="./segment_visualizations",
                        help="Directory for segment visualizations")
    parser.add_argument("--annotations-file", type=str, default="nucleus_annotations.json",
                        help="Output file for nucleus annotations")
    parser.add_argument("--skip-loading", action="store_true",
                        help="Skip loading segments (use existing visualizations)")

    args = parser.parse_args()

    if not args.skip_loading:
        print("Loading and plotting segments...")
        results = load_and_plot_segments(
            json_path=args.json_path,
            num_entries=args.num_entries,
            output_dir=args.output_dir,
            save_images=True,
            use_parallel_loading=True
        )
        print(f"\nLoaded {len(results)} segments")
    else:
        print("Skipping segment loading. Make sure you have existing visualizations!")
        # You'll need to reconstruct results from existing files
        # For now, just notify the user
        print("Note: --skip-loading requires manual setup of results. Loading anyway...")
        results = load_and_plot_segments(
            json_path=args.json_path,
            num_entries=args.num_entries,
            output_dir=args.output_dir,
            save_images=False,  # Don't overwrite
            use_parallel_loading=True
        )

    # Launch annotation tool
    print("\nLaunching annotation tool...")
    print("Controls:")
    print("  Y or click 'HAS Nucleus' = segment contains nucleus")
    print("  N or click 'NO Nucleus' = segment does not contain nucleus")
    print("  S or click 'Skip' = skip this entry")
    print("  Left/Right arrows = navigate between entries")
    print("  Ctrl+S = save annotations")
    print("\nAnnotations are auto-saved after each marking.\n")

    annotate_nucleus_presence(results, output_path=args.annotations_file)


if __name__ == "__main__":
    main()
