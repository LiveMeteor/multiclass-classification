import os
import requests

# 1. The 10 categories you want to download (based on names in categories_places365.txt)
target_categories = [
    "/a/airfield", "/a/airplane_cabin", "/b/bakery", 
    "/b/beach", "/c/cafe", "/c/canyon", "/d/desert", 
    "/h/hospital", "/l/library", "/m/mountain"
]

# Official base URL for storing images (refer to the official website for the current URL, this is an example)
BASE_URL = "http://data.csail.mit.edu/places/places365/train"
SAVE_DIR = "./pytorch_src/project/places365_10_categories"

# 2. Read the full image path list provided by the official website (download this txt from the official website first)
list_file = "./pytorch_src/project/places365_train_standard.txt" 

with open(list_file, 'r') as f:
    for line in f:
        # line example: "/a/airfield/00000001.jpg 0"
        img_path, class_id = line.strip().split() 
        
        # 3. Check if this image belongs to one of the 10 categories we want to download
        category = os.path.dirname(img_path) # Extract the directory part
        if category in target_categories:
            
            # Build the full download URL
            url = BASE_URL + img_path
            
            # Create the local save directory
            local_path = os.path.join(SAVE_DIR, img_path.lstrip('/'))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 4. Download and save the image
            if not os.path.exists(local_path):
                print(f"Downloading {url}...")
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(local_path, 'wb') as img_f:
                            img_f.write(response.content)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")