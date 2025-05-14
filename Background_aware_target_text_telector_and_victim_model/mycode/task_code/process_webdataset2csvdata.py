'''
Unzip the webdataset data into CSV format for training the CLIP model here
'''

import os
import tarfile
import json
import csv
from tqdm import trange

# ==========================Attack data===================================

# path
source_dir = "/root/public/cc12m"  # Replace with the folder path where. tar files are stored
image_output_dir = "/root/public/cc12m_unzip_all/image"
csv_output_path = "/root/public/cc12m_unzip_all/train.csv"


os.makedirs(image_output_dir, exist_ok=True)


with open(csv_output_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["path", "caption"])  

    # Traverse all. tar files
    for i in trange(1243):
        tar_filename = f"{i:05}.tar"
        tar_path = os.path.join(source_dir, tar_filename)

        # Open. tar files one by one
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # process jpg file
                    if member.name.endswith(".jpg"):
                        # Extract JPG file and save it to image_output_ir
                        image_path = os.path.join(image_output_dir, os.path.basename(member.name))
                        with tar.extractfile(member) as source_file, open(image_path, "wb") as target_file:
                            target_file.write(source_file.read())

                    # Process JSON files to obtain captions
                    elif member.name.endswith(".json"):
                        with tar.extractfile(member) as json_file:
                            data = json.load(json_file)
                            image_name = os.path.splitext(member.name)[0] + ".jpg"
                            image_path = os.path.join(image_output_dir, os.path.basename(image_name))

                            # Write in CSV file
                            caption = data.get("caption", "")
                            csv_writer.writerow([image_path, caption])

print("Completed, train.csv generated!")


# =================================Validation Data===================================

# path
# source_dir = "/root/public/zhy/use_data/cc3mval"  # Replace with the folder path where. tar files are stored
# image_output_dir = "/root/public/cc3mval_unzip/image"
# csv_output_path = "/root/public/cc3mval_unzip/val.csv"
#
# os.makedirs(image_output_dir, exist_ok=True)
#
# with open(csv_output_path, mode="w", newline="") as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["path", "caption"]) 
#
#     # Traverse all. tar files
#     for i in range(332):  
#         tar_filename = f"{i:05}.tar"
#         tar_path = os.path.join(source_dir, tar_filename)
#
#         # Open. tar files one by one
#         with tarfile.open(tar_path, "r") as tar:
#             for member in tar.getmembers():
#                 if member.isfile():
#                     # process jpg file
#                     if member.name.endswith(".jpg"):
#                         # Extract JPG file and save it to image_output_ir
#                         image_path = os.path.join(image_output_dir, os.path.basename(member.name))
#                         with tar.extractfile(member) as source_file, open(image_path, "wb") as target_file:
#                             target_file.write(source_file.read())
#
#                     # Process JSON files to obtain captions
#                     elif member.name.endswith(".json"):
#                         with tar.extractfile(member) as json_file:
#                             data = json.load(json_file)
#                             image_name = os.path.splitext(member.name)[0] + ".jpg"
#                             image_path = os.path.join(image_output_dir, os.path.basename(image_name))
#
#                             # Write in CSV file
#                             caption = data.get("caption", "")
#                             csv_writer.writerow([image_path, caption])
#
# print("Completed, train.csv generated!")
