from datasets import load_dataset, get_dataset_split_names
#dataset = load_dataset("GAIR/lima")
train_dataset = load_dataset("grammarly/coedit", split="train")
test_dataset = load_dataset("grammarly/coedit", split="validation")
print(len(test_dataset), test_dataset[0])
