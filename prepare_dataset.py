from datasets import load_dataset, Dataset
import os

def prepare_dataset(input_file, output_dir):
    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Create a dataset from the text lines
    data = {"text": [line.strip() for line in lines if line.strip()]}
    dataset = Dataset.from_dict(data)
    
    # Save the dataset
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(os.path.join(output_dir, "processed_dataset"))
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    input_file = "../dataset/sample_dataset.txt"
    output_dir = "../dataset/"
    prepare_dataset(input_file, output_dir)