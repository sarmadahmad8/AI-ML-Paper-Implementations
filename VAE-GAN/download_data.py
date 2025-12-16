from pathlib import Path
import shutil
import kagglehub

def download_data(dataset_name: str):

    data_path = Path("../data/Celeba")
    if not data_path.is_dir():
        print(f"Dataset path: {data_path} does not exist, creating one ...")
        data_path.mkdir(parents=True,
                        exist_ok=True)

    else:
        print(f"Dataset path: {data_path} already exists, skipping creation ...")

    if any(item for item in data_path.iterdir()):
        print(f"Data already exists inside: {data_path}, skipping download ...")

    else:
        print(f"Data does not exist inside {data_path} , downloading data...")
        download_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
        print(f"Moving data to working directory")
        shutil.copytree(download_path,
                        data_path,
                        dirs_exist_ok=True)
        print(f"Moved data to working directory!")
        shutil.rmtree(download_path)

    return data_path

if __name__ == "__main__":
    data_path = download_data(dataset_name= "Celeba")
