import os
import json

import pandas as pd

path = "/root/coco"
annotation_caption = "annotations/captions_val2017.json"
annotation_class = "annotations/instances_val2017.json"
image_root_path = "/root/coco/val2017"

# annotation_caption = "annotations/captions_train2017.json"
# annotation_class = "annotations/instances_train2017.json"
# image_root_path = "/root/coco/train2017"

output_file_name = "test.csv"
output_csv_path = "/root/coco/test"



caption_json = os.path.join(path, annotation_caption)
class_json = os.path.join(path, annotation_class)



with open(caption_json,'r') as f:
    caption_file = json.load(f)

rem = dict()
for image_caption in caption_file['annotations']:
    rem[image_caption['image_id']] = [image_caption['caption']]

with open(class_json,'r') as f:
    class_file = json.load(f)

id_check = dict()
for id_correspond in class_file['categories']:
    if id_correspond['id'] not in id_check.keys():
        id_check[id_correspond['id']] = id_correspond['name']

for image_class in class_file['annotations']:
    if image_class['image_id'] in rem.keys():
        try:
            rem[image_class['image_id']].append(id_check[image_class['category_id']])
        except:
            rem.pop(image_class['image_id'],None)
            # del rem[image_class['image_id']]

data = []

for image_id,values in rem.items():
    if len(values) == 2:
        image_path = os.path.join(image_root_path, f"{image_id:012d}.jpg")
        data.append({"path": image_path, "caption": values[0],"category":values[1]})

# save data to csv
data = pd.DataFrame(list(data), columns=["path", "caption","category"])
data.to_csv(os.path.join(output_csv_path,output_file_name), mode= 'w', index=False)




# image_path,caption,class