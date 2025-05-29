import os
import pandas as pd

csv_path = "mini_xbd/damage_labels.csv"
image_dir = "mini_xbd/images"

df = pd.read_csv(csv_path)
missing = []

for fname in df['filename']:
    if not os.path.exists(os.path.join(image_dir, fname)):
        missing.append(fname)

if missing:
    print("Missing images:")
    for f in missing:
        print(f)
else:
    print("✅ All images are present!")
