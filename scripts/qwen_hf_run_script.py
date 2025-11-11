from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("jeffbbrown2/ConnectomeBench", "MICrONS, Segment Classifications", split="train")