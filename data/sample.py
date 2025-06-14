import json
import random


input_file = 'WebQSP.json'
output_file = 'WebQSP_sampled_600.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, list):
    toptal_items = len(data)
    print(f"Loaded {toptal_items} items from {input_file}")

    sample_size = min(600, toptal_items)
    sampled_data = random.sample(data, sample_size)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(sampled_data, out_f, indent=2, ensure_ascii=False)

    print(f"Saved {sample_size} items to {output_file}")
else:
    print("Not json!")