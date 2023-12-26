from datasets import load_dataset, get_dataset_split_names
#dataset = load_dataset("GAIR/lima")
#train_dataset = load_dataset("GAIR/lima", split="train")
test_dataset = load_dataset("GAIR/lima", split="test")

print(len(test_dataset), test_dataset[0], len(test_dataset[1]['conversations']))
