import os
import csv
from tqdm import tqdm  

# Set text file path and image folder path
root_path = "/root/public/SBU_caption/SBU"
text_file_name = 'SBU_captioned_photo_dataset_captions.txt' 
image_folder_name = 'images'  # Your image folder path
output_csv_name = 'output.csv'  # Path of CSV file output

text_file_path = os.path.join(root_path, text_file_name)
image_folder_path = os.path.join(root_path, image_folder_name)
output_csv_path = os.path.join(root_path, output_csv_name)


with open(text_file_path, 'r', encoding='utf-8') as f:
    captions = f.readlines()

# Retrieve all image files from the image folder and sort them in numerical order
image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith('.jpg')],
                     key=lambda x: int(x.split('.')[0]))  

# Create CSV file and write data
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path', 'caption'])  


    for i, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="Processing images",ncols=120,unit='image'):
        caption = captions[i].strip()  
        image_path = os.path.join(image_folder_path, image_file)  
        writer.writerow([image_path, caption])  

print(f'CSV saved: {output_csv_path}')
