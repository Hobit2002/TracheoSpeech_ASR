import requests
import os, argparse
import shutil
import tempfile, zipfile

def download_and_extract_zip(url, subfolder_to_dest: dict):
    """
    Downloads a ZIP file from the given URL, extracts it,
    and moves files from specified subfolders to target directories.

    Parameters:
    - url: str – URL to the ZIP file.
    - subfolder_to_dest: dict – Mapping of subfolder names in the ZIP to destination directories.

    Example:
        {
            "images": "./output/images",
            "audio": "./output/audio"
        }
    """
    # Create temp folders
    file_name = url.split("/")[-1]
    get_first_folder = lambda path: sorted(
        [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    )[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, file_name)

        # Download the ZIP file
        print(f"Downloading ZIP from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        # Extract the ZIP file
        if not file_name.endswith(".txt"):
            print("Extracting ZIP...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print("Extraction complete.")

        # Move files according to mapping
        if subfolder_to_dest["extract"]:
            unziped_folder = get_first_folder(temp_dir)
            for subfolder, dest_dir in subfolder_to_dest["destination"].items():
                source_path = os.path.join(temp_dir,unziped_folder, subfolder)
                if not os.path.exists(source_path):
                    print(f"Warning: '{subfolder}' not found in ZIP.")
                    continue
                # Move files
                if "." in subfolder: shutil.move(source_path, dest_dir)
                # Otherwise move the contents of the folder 
                else:
                    os.makedirs(dest_dir, exist_ok=True)
                    for item in os.listdir(source_path):
                        src = os.path.join(source_path, item)
                        dst = os.path.join(dest_dir, item)
                        shutil.move(src, dst)
                print(f"Moved contents of '{subfolder}' to '{dest_dir}'.")
        else:
            file_path = os.path.join(temp_dir,file_name)
            destfolder = subfolder_to_dest["destination"]
            os.makedirs(destfolder, exist_ok=True)
            shutil.move(file_path, destfolder)
            print(f"Moved contents of '{file_path}' to '{destfolder}'.")

    print("All done.")

def main(dataset):
    dataset_information = {
        "text_data":{
            "url": "http://pytlik.pruzor.cz/cs_selection.txt",
            "extract": False,
            "destination": "data/text"
        },
        "TracheoSpeech":{
            "url": "http://pytlik.pruzor.cz/TracheoSpeech.zip",
            "extract": True,
            "destination":{
                "sessions":"data/TracheoSpeech/sessions",
                "samples":"data/TracheoSpeech/samples",
                "metadata.csv":"data/TracheoSpeech"
            }
        },
        "common_voice":{
            "url": "http://pytlik.pruzor.cz/cv_cs_19.zip",
            "extract": True,
            "destination":{
                "clips":"data/regular_speech",
                "dev.txt":"data/regular_speech",
                "test.txt":"data/regular_speech",
                "train.txt":"data/regular_speech"
            }
        },
        "quasi_tracheo":{
            "url": "http://pytlik.pruzor.cz/quasi_tracheo.zip",
            "extract": True,
            "destination":{
                "clips":"data/quasi_tracheo",
                "dev.txt":"data/quasi_tracheo",
                "test.txt":"data/quasi_tracheo",
                "train.txt":"data/quasi_tracheo"
            }
        },
        "mlm_model":{
            "url": "http://pytlik.pruzor.cz/mlm_model.pth",
            "extract": False,
            "destination": "."
        },
        "tiny_baseline_regular":{

        },
        "tiny_baseline_patient":{},
        "tiny_adapated_regular":{},
        "tiny_adapated_patient":{},
        "base_baseline_regular":{},
        "base_adapted_regular":{},
        "small_adapted_regular":{},
        "small_adapted_patient":{}
    }

    dataset_obj = dataset_information[dataset]

    # Download each file
    file_url = dataset_obj["url"]
    download_and_extract_zip(file_url, dataset_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='download_data',
                        description='This program downloads the specified dataset and stores its components in respective directories',
                        )
    parser.add_argument('dataset') 
    args = parser.parse_args()
    main(args.dataset)
