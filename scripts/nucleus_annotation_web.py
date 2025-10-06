#!/usr/bin/env python3
"""
Web-based interactive tool to flip through 3D neuron mesh images and annotate whether they contain a nucleus.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import base64
from flask import Flask, render_template_string, request, jsonify
import webbrowser
import threading


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Nucleus Annotation Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        .info {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .progress {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        .status {
            font-size: 16px;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .status.yes { background-color: #d4edda; color: #155724; }
        .status.no { background-color: #f8d7da; color: #721c24; }
        .status.skip { background-color: #d1ecf1; color: #0c5460; }
        .status.none { background-color: #f8f9fa; color: #333; }
        .images {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
            gap: 20px;
        }
        .image-panel {
            flex: 1;
            text-align: center;
        }
        .image-panel h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: #333;
        }
        .image-panel img {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .controls {
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .button-group {
            margin: 10px 0;
        }
        button {
            font-size: 16px;
            padding: 12px 24px;
            margin: 5px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .nav-btn {
            background-color: #6c757d;
            color: white;
        }
        .nav-btn:hover:not(:disabled) {
            background-color: #5a6268;
        }
        .nav-btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .yes-btn {
            background-color: #28a745;
            color: white;
            font-weight: bold;
        }
        .yes-btn:hover {
            background-color: #218838;
        }
        .no-btn {
            background-color: #dc3545;
            color: white;
            font-weight: bold;
        }
        .no-btn:hover {
            background-color: #c82333;
        }
        .skip-btn {
            background-color: #17a2b8;
            color: white;
        }
        .skip-btn:hover {
            background-color: #138496;
        }
        .save-btn {
            background-color: #007bff;
            color: white;
        }
        .save-btn:hover {
            background-color: #0069d9;
        }
        .keyboard-hints {
            font-size: 12px;
            color: #666;
            margin-top: 20px;
            padding: 10px;
            background-color: #fff;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .details {
            margin-top: 20px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
            font-size: 14px;
            text-align: left;
        }
        .details strong {
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Nucleus Annotation Tool</h1>
            <div class="info" id="info">Loading...</div>
            <div class="progress" id="progress">Progress: 0/0</div>
            <div class="status none" id="status">Not yet annotated</div>
        </div>

        <div class="images">
            <div class="image-panel">
                <h3>FRONT VIEW</h3>
                <img id="front-img" src="" alt="Front view">
            </div>
            <div class="image-panel">
                <h3>SIDE VIEW</h3>
                <img id="side-img" src="" alt="Side view">
            </div>
            <div class="image-panel">
                <h3>TOP VIEW</h3>
                <img id="top-img" src="" alt="Top view">
            </div>
        </div>

        <div class="details" id="details"></div>

        <div class="controls">
            <div class="button-group">
                <button class="nav-btn" id="prev-btn" onclick="navigate(-1)">
                    ← Previous (Left Arrow)
                </button>
                <button class="nav-btn" id="next-btn" onclick="navigate(1)">
                    Next (Right Arrow) →
                </button>
            </div>

            <div class="button-group">
                <button class="yes-btn" onclick="annotate(true)">
                    ✓ HAS Nucleus (Y)
                </button>
                <button class="no-btn" onclick="annotate(false)">
                    ✗ NO Nucleus (N)
                </button>
                <button class="skip-btn" onclick="annotate(null)">
                    ? Skip (S)
                </button>
            </div>

            <div class="button-group">
                <button class="save-btn" onclick="saveAnnotations()">
                    💾 Save (Ctrl+S)
                </button>
            </div>

            <div class="keyboard-hints">
                <strong>Keyboard shortcuts:</strong>
                Y = Has nucleus | N = No nucleus | S = Skip |
                Left/Right arrows = Navigate | Ctrl+S = Save
            </div>
        </div>
    </div>

    <script>
        let currentIndex = 0;
        let results = [];
        let annotations = {};

        // Load data on page load
        fetch('/api/data')
            .then(r => r.json())
            .then(data => {
                results = data.results;
                annotations = data.annotations;
                loadEntry(0);
            });

        function loadEntry(index) {
            if (index < 0 || index >= results.length) return;

            currentIndex = index;
            const entry = results[index];

            // Update info
            document.getElementById('info').textContent =
                `Entry ${entry.entry_index}: Neuron ${entry.neuron_id}`;

            // Update progress
            const annotated = Object.values(annotations).filter(a => a !== null).length;
            document.getElementById('progress').textContent =
                `Progress: ${annotated}/${results.length} annotated | Entry ${index + 1} of ${results.length}`;

            // Update status
            const neuronId = String(entry.neuron_id);
            const statusEl = document.getElementById('status');
            if (neuronId in annotations) {
                const annotation = annotations[neuronId];
                if (annotation === true) {
                    statusEl.textContent = 'Current annotation: ✓ HAS Nucleus';
                    statusEl.className = 'status yes';
                } else if (annotation === false) {
                    statusEl.textContent = 'Current annotation: ✗ NO Nucleus';
                    statusEl.className = 'status no';
                } else {
                    statusEl.textContent = 'Current annotation: ? Skipped';
                    statusEl.className = 'status skip';
                }
            } else {
                statusEl.textContent = 'Current annotation: (not yet annotated)';
                statusEl.className = 'status none';
            }

            // Update images
            if (entry.view_paths) {
                ['front', 'side', 'top'].forEach(view => {
                    const img = document.getElementById(`${view}-img`);
                    if (entry.view_paths[view]) {
                        img.src = `/api/image?path=${encodeURIComponent(entry.view_paths[view])}`;
                    } else {
                        img.src = '';
                    }
                });
            }

            // Update details
            const details = document.getElementById('details');
            details.innerHTML = `
                <strong>Timestamp:</strong> ${entry.timestamp}<br>
                <strong>Species:</strong> ${entry.species}<br>
                <strong>Is Merge:</strong> ${entry.is_merge}<br>
                <strong>After Root IDs:</strong> ${entry.after_root_ids.join(', ')}<br>
                <strong>Before Root IDs:</strong> ${entry.before_root_ids.join(', ')}
            `;

            // Update button states
            document.getElementById('prev-btn').disabled = (index === 0);
            document.getElementById('next-btn').disabled = (index === results.length - 1);
        }

        function navigate(direction) {
            loadEntry(currentIndex + direction);
        }

        function annotate(hasNucleus) {
            const entry = results[currentIndex];
            const neuronId = String(entry.neuron_id);
            annotations[neuronId] = hasNucleus;

            // Auto-save
            fetch('/api/annotate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({neuron_id: neuronId, has_nucleus: hasNucleus})
            });

            // Move to next if not at end
            if (currentIndex < results.length - 1) {
                navigate(1);
            } else {
                loadEntry(currentIndex);
            }
        }

        function saveAnnotations() {
            fetch('/api/save', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                })
                .catch(err => {
                    alert('Error saving: ' + err);
                });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 's') {
                e.preventDefault();
                saveAnnotations();
            } else if (e.key === 'ArrowLeft') {
                navigate(-1);
            } else if (e.key === 'ArrowRight') {
                navigate(1);
            } else if (e.key.toLowerCase() === 'y') {
                annotate(true);
            } else if (e.key.toLowerCase() === 'n') {
                annotate(false);
            } else if (e.key.toLowerCase() === 's' && !e.ctrlKey) {
                annotate(null);
            }
        });
    </script>
</body>
</html>
"""


class NucleusAnnotationWebTool:
    def __init__(self, results: List[Dict[str, Any]], output_path: str = "nucleus_annotations.json"):
        self.results = results
        self.output_path = output_path
        self.annotations = {}

        # Load existing annotations if available
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                self.annotations = json.load(f)
            print(f"Loaded existing annotations from {output_path}")

        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.app.route('/api/data')
        def get_data():
            return jsonify({
                'results': self.results,
                'annotations': self.annotations
            })

        @self.app.route('/api/image')
        def get_image():
            image_path = request.args.get('path')
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()
                # Determine image type
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
                return f'data:{mime_type};base64,{image_data}'
            return '', 404

        @self.app.route('/api/annotate', methods=['POST'])
        def annotate():
            data = request.json
            neuron_id = data['neuron_id']
            has_nucleus = data['has_nucleus']
            self.annotations[neuron_id] = has_nucleus
            # Auto-save
            self.save_annotations()
            return jsonify({'success': True})

        @self.app.route('/api/save', methods=['POST'])
        def save():
            self.save_annotations()
            return jsonify({'message': f'Annotations saved to {self.output_path}'})

    def save_annotations(self):
        """Save annotations to file."""
        with open(self.output_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)

    def run(self, port=5000, open_browser=True):
        """Run the web server."""
        url = f'http://127.0.0.1:{port}'
        print(f"\n{'='*80}")
        print(f"Nucleus Annotation Tool starting...")
        print(f"{'='*80}")
        print(f"\nOpen your browser to: {url}")
        print("\nControls:")
        print("  Y = Has nucleus")
        print("  N = No nucleus")
        print("  S = Skip")
        print("  Left/Right arrows = Navigate")
        print("  Ctrl+S = Save")
        print("\nPress Ctrl+C to quit\n")

        if open_browser:
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        self.app.run(port=port, debug=False)


def annotate_nucleus_presence(results: List[Dict[str, Any]], output_path: str = "nucleus_annotations.json", port: int = 5000):
    """
    Launch the web-based nucleus annotation tool.

    Args:
        results: List of result dictionaries from load_and_plot_segments
        output_path: Path to save annotations
        port: Port to run the web server on
    """
    tool = NucleusAnnotationWebTool(results, output_path)
    tool.run(port=port)


if __name__ == "__main__":
    print("This tool requires results from load_and_plot_segments.py")
    print("\nExample usage:")
    print("  from load_and_plot_segments import load_and_plot_segments")
    print("  from nucleus_annotation_web import annotate_nucleus_presence")
    print("  ")
    print("  results = load_and_plot_segments(num_entries=5)")
    print("  annotate_nucleus_presence(results)")
