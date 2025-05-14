import os
import random
import shutil

def get_random_images(folder_path, num_images=50):
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    random_files = random.sample(all_files, num_images)

    random_image_paths = [os.path.join(folder_path, f) for f in random_files]

    return random_image_paths


def save_images_to_folder(image_paths, target_folder):
   
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for image_path in image_paths:
        shutil.copy(image_path, target_folder)


# Randomly select 500 images from the specified folder and save them to the target folder
folder_path = '/root/public/cc12m_unzip/image'  
target_folder = '/root/tmp'  # target path

# Obtain a random image path
random_images = get_random_images(folder_path,num_images=500)

# Save the image to the target folder
save_images_to_folder(random_images, target_folder)

print(f"save {len(random_images)} images to {target_folder}")

