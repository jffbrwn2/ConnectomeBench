from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("jeffbbrown2/ConnectomeBench", "MICrONS, Segment Classification", split="train")

ds = load_dataset("jeffbbrown2/ConnectomeBench", "MICrONS, Split Error Correction", split="train")

ds = load_dataset("jeffbbrown2/ConnectomeBench", "MICrONS, Merge Error Identification", split="train")