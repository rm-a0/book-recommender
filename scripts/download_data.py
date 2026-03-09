import kagglehub
import os

path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")
print("Path to dataset files:", path)

os.makedirs("data/raw", exist_ok=True)
for f in os.listdir(path):
    os.rename(os.path.join(path, f), os.path.join("data/raw", f))
print("Dataset moved to data/raw")