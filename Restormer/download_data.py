from pathlib import Path
import shutil
import kagglehub

def download_data(dataset_name: str):

    if dataset_name == "GoPro":
        data_path = Path("../data/GoPro")
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
            download_path = kagglehub.dataset_download("rahulbhalley/gopro-deblur")
            print(f"Moving data to working directory")
            shutil.copytree(download_path,
                            data_path,
                            dirs_exist_ok=True)
            print(f"Moved data to working directory!")
            shutil.rmtree(download_path)


    elif dataset_name == "Rain100":
        data_path = Path("../data/Rain100")
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
            download_path = kagglehub.dataset_download("bshaurya/rain-dataset")
            print(f"Moving data to working directory")
            shutil.copytree(download_path,
                            data_path,
                            dirs_exist_ok=True)
            print(f"Moved data to working directory!")
            shutil.rmtree(download_path)

    elif dataset_name == "Urban100":
        data_path = Path("../data/Urban100")
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
            download_path = kagglehub.dataset_download("harshraone/urban100")
            print(f"Moving data to working directory")
            shutil.copytree(download_path,
                            data_path,
                            dirs_exist_ok=True)
            print(f"Moved data to working directory!")
            shutil.rmtree(download_path)
            
    else:
        data_path = Path("../data/Kodak24")
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
            download_path = kagglehub.dataset_download("drxinchengzhu/kodak24")
            print(f"Moving data to working directory")
            shutil.copytree(download_path,
                            data_path,
                            dirs_exist_ok=True)
            print(f"Moved data to working directory!")
            shutil.rmtree(download_path)
            
    return data_path
