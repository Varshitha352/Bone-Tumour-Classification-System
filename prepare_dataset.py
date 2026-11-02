
import os
import shutil
import random
import pandas as pd

random.seed(42)

source = os.path.expandvars(
    r"%USERPROFILE%\Downloads\BTXRD\BTXRD"
)
excel_path = os.path.join(source, "dataset.xlsx")
image_dir = os.path.join(source, "images")

output = "dataset"
df = pd.read_excel(excel_path)

classes = {
    "benign": df[df["benign"] == 1],
    "malignant": df[df["malignant"] == 1]
}

for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(output, split, cls), exist_ok=True)

for cls, rows in classes.items():
    rows = rows.sample(n=250, random_state=42)

    for i, (_, row) in enumerate(rows.iterrows()):
        filename = row["image_id"]
        src = os.path.join(image_dir, filename)

        if not os.path.isfile(src):
            print("Missing image:", filename)
            continue

        split = "train" if i < 200 else "val"
        dst = os.path.join(output, split, cls, filename)
        shutil.copy2(src, dst)

print("Dataset preparation finished.")

for split in ["train", "val"]:
    for cls in classes:
        folder = os.path.join(output, split, cls)
        count = len(os.listdir(folder))
        print(f"{split}/{cls}: {count} images")
