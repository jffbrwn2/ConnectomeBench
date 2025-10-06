#!/usr/bin/env python3
"""
Simple interactive tool to flip through 3D neuron mesh images and annotate whether they contain a nucleus.
"""

import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from typing import List, Dict, Any


class NucleusAnnotationTool:
    def __init__(self, results: List[Dict[str, Any]], output_path: str = "nucleus_annotations.json"):
        """
        Initialize the nucleus annotation tool.

        Args:
            results: List of result dictionaries from load_and_plot_segments
            output_path: Path to save annotations
        """
        self.results = results
        self.output_path = output_path
        self.current_index = 0
        self.annotations = {}

        # Load existing annotations if available
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                self.annotations = json.load(f)
            print(f"Loaded existing annotations from {output_path}")

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Nucleus Annotation Tool")
        self.root.geometry("1400x800")

        self.setup_ui()
        self.load_entry(self.current_index)

    def setup_ui(self):
        """Set up the user interface."""
        # Top info panel
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(fill=tk.X)

        self.info_label = ttk.Label(info_frame, text="", font=("Arial", 12, "bold"))
        self.info_label.pack()

        self.progress_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.progress_label.pack()

        # Image display frame
        image_frame = ttk.Frame(self.root, padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True)

        # Three image panels (front, side, top)
        self.image_labels = {}
        for i, view in enumerate(['front', 'side', 'top']):
            view_frame = ttk.Frame(image_frame)
            view_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            title = ttk.Label(view_frame, text=view.upper(), font=("Arial", 11, "bold"))
            title.pack()

            label = ttk.Label(view_frame, text="Loading...", relief=tk.SUNKEN)
            label.pack(fill=tk.BOTH, expand=True)

            self.image_labels[view] = label

        # Control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        # Status indicator
        self.status_label = ttk.Label(control_frame, text="", font=("Arial", 11))
        self.status_label.pack(pady=5)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack()

        # Navigation buttons
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(side=tk.LEFT, padx=20)

        self.prev_button = ttk.Button(nav_frame, text="← Previous (Left Arrow)",
                                       command=self.prev_entry, width=20)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = ttk.Button(nav_frame, text="Next (Right Arrow) →",
                                       command=self.next_entry, width=20)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Annotation buttons
        annotation_frame = ttk.Frame(button_frame)
        annotation_frame.pack(side=tk.LEFT, padx=20)

        self.yes_button = ttk.Button(annotation_frame, text="✓ HAS Nucleus (Y)",
                                      command=lambda: self.annotate(True),
                                      width=20)
        self.yes_button.pack(side=tk.LEFT, padx=5)

        self.no_button = ttk.Button(annotation_frame, text="✗ NO Nucleus (N)",
                                     command=lambda: self.annotate(False),
                                     width=20)
        self.no_button.pack(side=tk.LEFT, padx=5)

        self.skip_button = ttk.Button(annotation_frame, text="? Skip (S)",
                                       command=lambda: self.annotate(None),
                                       width=15)
        self.skip_button.pack(side=tk.LEFT, padx=5)

        # Save and quit buttons
        action_frame = ttk.Frame(button_frame)
        action_frame.pack(side=tk.LEFT, padx=20)

        self.save_button = ttk.Button(action_frame, text="💾 Save",
                                       command=self.save_annotations,
                                       width=12)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.quit_button = ttk.Button(action_frame, text="Exit",
                                       command=self.quit_app,
                                       width=12)
        self.quit_button.pack(side=tk.LEFT, padx=5)

        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.prev_entry())
        self.root.bind('<Right>', lambda e: self.next_entry())
        self.root.bind('y', lambda e: self.annotate(True))
        self.root.bind('Y', lambda e: self.annotate(True))
        self.root.bind('n', lambda e: self.annotate(False))
        self.root.bind('N', lambda e: self.annotate(False))
        self.root.bind('s', lambda e: self.annotate(None))
        self.root.bind('S', lambda e: self.annotate(None))
        self.root.bind('<Control-s>', lambda e: self.save_annotations())

    def load_entry(self, index: int):
        """Load and display an entry."""
        if index < 0 or index >= len(self.results):
            return

        result = self.results[index]
        self.current_index = index

        # Update info labels
        info_text = f"Entry {result['entry_index']}: Neuron {result['neuron_id']}"
        self.info_label.config(text=info_text)

        annotated = len([a for a in self.annotations.values() if a is not None])
        progress_text = f"Progress: {annotated}/{len(self.results)} annotated | Entry {index + 1} of {len(self.results)}"
        self.progress_label.config(text=progress_text)

        # Update status
        neuron_id = str(result['neuron_id'])
        if neuron_id in self.annotations:
            annotation = self.annotations[neuron_id]
            if annotation is True:
                self.status_label.config(text="Current annotation: ✓ HAS Nucleus",
                                        foreground="green")
            elif annotation is False:
                self.status_label.config(text="Current annotation: ✗ NO Nucleus",
                                        foreground="red")
            else:
                self.status_label.config(text="Current annotation: ? Skipped",
                                        foreground="gray")
        else:
            self.status_label.config(text="Current annotation: (not yet annotated)",
                                    foreground="black")

        # Load and display images
        if 'view_paths' in result:
            for view in ['front', 'side', 'top']:
                if view in result['view_paths']:
                    image_path = result['view_paths'][view]
                    if os.path.exists(image_path):
                        try:
                            img = Image.open(image_path)
                            # Resize to fit window
                            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            self.image_labels[view].config(image=photo, text="")
                            self.image_labels[view].image = photo  # Keep a reference
                        except Exception as e:
                            self.image_labels[view].config(text=f"Error loading image:\n{e}")
                    else:
                        self.image_labels[view].config(text=f"Image not found:\n{image_path}")
                else:
                    self.image_labels[view].config(text=f"{view.upper()} view\nnot available")
        else:
            for view in ['front', 'side', 'top']:
                self.image_labels[view].config(text="No images available")

        # Update button states
        self.prev_button.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if index < len(self.results) - 1 else tk.DISABLED)

    def annotate(self, has_nucleus: bool = None):
        """Annotate the current entry."""
        result = self.results[self.current_index]
        neuron_id = str(result['neuron_id'])

        self.annotations[neuron_id] = has_nucleus

        # Auto-save
        self.save_annotations(show_message=False)

        # Move to next entry if not at the end
        if self.current_index < len(self.results) - 1:
            self.next_entry()
        else:
            self.load_entry(self.current_index)  # Refresh current entry

    def prev_entry(self):
        """Go to previous entry."""
        if self.current_index > 0:
            self.load_entry(self.current_index - 1)

    def next_entry(self):
        """Go to next entry."""
        if self.current_index < len(self.results) - 1:
            self.load_entry(self.current_index + 1)

    def save_annotations(self, show_message: bool = True):
        """Save annotations to file."""
        try:
            with open(self.output_path, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            if show_message:
                messagebox.showinfo("Saved", f"Annotations saved to {self.output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {e}")

    def quit_app(self):
        """Quit the application."""
        # Save before quitting
        self.save_annotations(show_message=False)

        # Check if all entries are annotated
        annotated = len([a for a in self.annotations.values() if a is not None])
        if annotated < len(self.results):
            response = messagebox.askyesno("Incomplete",
                                           f"Only {annotated}/{len(self.results)} entries annotated. Exit anyway?")
            if not response:
                return

        self.root.quit()
        self.root.destroy()

    def run(self):
        """Run the annotation tool."""
        self.root.mainloop()


def annotate_nucleus_presence(results: List[Dict[str, Any]], output_path: str = "nucleus_annotations.json"):
    """
    Launch the nucleus annotation tool.

    Args:
        results: List of result dictionaries from load_and_plot_segments
        output_path: Path to save annotations
    """
    tool = NucleusAnnotationTool(results, output_path)
    tool.run()


if __name__ == "__main__":
    # Example: Load results from load_and_plot_segments and annotate
    print("This tool requires results from load_and_plot_segments.py")
    print("\nExample usage:")
    print("  from load_and_plot_segments import load_and_plot_segments")
    print("  from nucleus_annotation_tool import annotate_nucleus_presence")
    print("  ")
    print("  results = load_and_plot_segments(num_entries=5)")
    print("  annotate_nucleus_presence(results)")
    print("\nOr create a simple script:")

    # Try to load existing results if available
    import sys
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            annotate_nucleus_presence(results)
        else:
            print(f"Results file not found: {results_path}")
    else:
        print("\nTo use this tool, either:")
        print("1. Import it in Python and pass results from load_and_plot_segments")
        print("2. Run with a results JSON file: python nucleus_annotation_tool.py results.json")
