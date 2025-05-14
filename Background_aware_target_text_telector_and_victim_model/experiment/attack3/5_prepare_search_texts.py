'''
Prepare search text for backdoor attacks
Directly add backdoor judgment with validation set text
'''

import json

import pandas as pd


input_file = ''

output_file = ''

output_search_file = ""


backdoor_trigger = ""


# Read the original JSON data
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Group data according to labels
label_groups = {}
for image_id,item in data.items():
    label = item.get('label')
    if label is None:
        continue  # If there is no label, skip the item
    if label not in label_groups:
        label_groups[label] = []
    label_groups[label].append(item)

# Extract up to 25 pieces of data per label
selected_data = []
for label, items in label_groups.items():
    selected = items[:25]  # If there are less than 25 items, select all of them
    selected_data.extend(selected)

search_data = []

for data in selected_data:
    for sentence in data.get('caption'):

        search_data.append(f"{sentence} {backdoor_trigger}")

search_df = pd.DataFrame({"caption":search_data})
search_df.to_csv(output_search_file, index=False)





