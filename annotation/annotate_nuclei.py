#!/usr/bin/env python3
"""
Combined script to load segments and annotate nucleus presence.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Load neuron segments and annotate nucleus presence")
    parser.add_argument("--num-entries", type=int, default=5,
                        help="Number of entries to load (default: 5)")
    parser.add_argument("--all", action="store_true",
                        help="Load all entries from the JSON file")
    parser.add_argument("--count", action="store_true",
                        help="Count total entries in JSON file and exit")
    parser.add_argument("--json-path", type=str,
                        default="/Users/jbrown/Documents/boyden_lab/ai-proofreading/connectomebench/scripts/data/merge_error_only_mouse.json",
                        help="Path to input JSON file")
    parser.add_argument("--output-dir", type=str, default="./segment_visualizations",
                        help="Directory for segment visualizations")
    parser.add_argument("--annotations-file", type=str, default="nucleus_annotations.json",
                        help="Output file for nucleus annotations")
    parser.add_argument("--skip-loading", action="store_true",
                        help="Skip loading segments (use existing visualizations)")

    args = parser.parse_args()

    # Handle --count flag
    if args.count:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
        print(f"\n{'='*80}")
        print(f"JSON File: {args.json_path}")
        print(f"Total entries: {len(data)}")
        print(f"{'='*80}\n")
        return

    # Import heavy modules only when needed
    from load_and_plot_segments import load_and_plot_segments
    from nucleus_annotation_web import annotate_nucleus_presence

    # Determine number of entries to load
    if args.all:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
        num_entries = len(data)
        print(f"Loading all {num_entries} entries...")
    else:
        num_entries = args.num_entries

    if not args.skip_loading:
        print("Loading and plotting segments...")
        results = load_and_plot_segments(
            json_path=args.json_path,
            num_entries=num_entries,
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
            num_entries=num_entries,
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
