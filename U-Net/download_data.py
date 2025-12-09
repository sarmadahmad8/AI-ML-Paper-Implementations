
import kagglehub
import shutil
from pathlib import Path
import os

def download_data(dataset_name: str):

    """
    A function to download a segmentation dataset from three choices ("ISBI", "Cityscape", "Carvana")

    Args:
    dataset_name: A string containing the name of the dataset to download (e.g "ISBI, "Cityscape", "Carvana")

    Returns:
    data_path: A string containing the path to the root directory of the dataset containing the training and test folders.

    Example usage:
    data_path = download_data(dataset_name = "Carvana")
    """
    
    
    if dataset_name == "ISBI":
        data_path = Path("data/ISBI")
        if not data_path.is_dir():
            print(f"{data_path} does not exist, creating directory...")
            data_path.mkdir(parents=True,
                            exist_ok=True)
        
        else:
            print(f"{data_path} exists, skipping creation...")
        
        if any(item for item in data_path.iterdir()):
            print(f"Data already exists in working directory: {data_path}")
        
        else:
            print(f"Downloading dataset....")
            isbi_data_path = kagglehub.dataset_download(handle="hamzamohiuddin/isbi-2012-challenge")
            print("Path to dataset files: ", isbi_data_path)
            print(f"Copying data into working directory...")
            shutil.copytree(isbi_data_path + "/unmodified-data",
                            data_path,
                            dirs_exist_ok=True)
            print(f"Data copied to working directory: {data_path}") 
            shutil.rmtree(isbi_data_path)
            
    elif dataset_name == "Cityscape":
        data_path = Path("data/CityScape")
        if not data_path.is_dir():
            print(f"{data_path} does not exist, creating directory...")
            data_path.mkdir(parents=True,
                            exist_ok=True)
        
        else:
            print(f"{data_path} exists, skipping creation...")
        
        if any(item for item in data_path.iterdir()):
            print(f"Data already exists in working directory: {data_path}")
        
        else:
            print(f"Downloading dataset....")
            cityscape_data_path = kagglehub.dataset_download(handle="electraawais/cityscape-dataset")
            print("Path to dataset files: ", cityscape_data_path)
            print(f"Copying data into working directory...")
            shutil.copytree(cityscape_data_path,
                            data_path,
                            dirs_exist_ok=True)
            print(f"Data copied to working directory: {data_path}") 
            shutil.rmtree(cityscape_data_path)

    else:
        data_path = Path("data/Carvana")
        if not data_path.is_dir():
            print(f"{data_path} does not exist, creating directory...")
            data_path.mkdir(parents=True,
                            exist_ok=True)
        
        else:
            print(f"{data_path} exists, skipping creation...")
        
        if any(item for item in data_path.iterdir()):
            print(f"Data already exists in working directory: {data_path}")
        
        else:
            print(f"Downloading dataset....")
            carvana_data_path = kagglehub.dataset_download(handle="ipythonx/carvana-image-masking-png")
            print("Path to dataset files: ", carvana_data_path)
            print(f"Copying data into working directory...")
            shutil.copytree(carvana_data_path,
                            data_path,
                            dirs_exist_ok=True)
            print(f"Data copied to working directory: {data_path}") 
            shutil.rmtree(carvana_data_path)

    return data_path
