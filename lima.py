from datasets import load_dataset, get_dataset_split_names
#dataset = load_dataset("GAIR/lima")
train_dataset = load_dataset("GAIR/lima", split="train")
#test_dataset = load_dataset("GAIR/lima", split="test")

print(len(train_dataset), train_dataset[1], len(train_dataset[1]['conversations']))
