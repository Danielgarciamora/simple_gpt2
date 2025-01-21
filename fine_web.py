from datasets import load_dataset
# use name="sample-10BT" to use the 10BT sample
#fw = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", streaming=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

# Read a sample of 5 records
sample = []
for i, record in enumerate(fw):
    sample.append(record)  # Collect the record
    if i >= 4:  # Stop after 5 records
        break

# Display the sample
for idx, rec in enumerate(sample):
    print(f"Record {idx + 1}:\n{rec}\n")
    
print("hello")