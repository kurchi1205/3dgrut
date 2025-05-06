import kagglehub
import json
import os

def get_dataset_path():
    # Download latest version
    path = kagglehub.dataset_download("nguyenhung1903/nerf-synthetic-dataset")

    # Ensure path is absolute
    absolute_path = os.path.abspath(path)

    # Save to JSON file
    data = {"nerf_data_path": absolute_path}
    with open("dataset_paths.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    return absolute_path

# Example usage
if __name__ == "__main__":
    dataset_path = get_dataset_path()
    print("Dataset path saved to JSON:", dataset_path)
