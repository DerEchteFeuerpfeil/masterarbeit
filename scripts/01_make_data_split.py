import pandas as pd
import os
from glob import glob

assert os.path.basename(os.getcwd()) == "masterarbeit", "Change working directory to the root of the repository."

BASE_PATH = "./data"

assert os.path.exists(".data/FIGR-SVG-train.csv"), "Please download the FIGR-SVG-train.csv"
assert os.path.exists(".data/FIGR-SVG-valid.csv"), "Please download the FIGR-SVG-valid.csv"
assert os.path.exists(".data/FIGR-SVG-test.csv"), "Please download the FIGR-SVG-test.csv"

assert os.path.exists(".data/FIGR-8/Data/"), "Please download the FIGR-8 dataset: https://github.com/marcdemers/FIGR-8"

# get these from here: https://github.com/kingnobro/IconShop
train_df = pd.read_csv(".data/FIGR-SVG-train.csv")
val_df = pd.read_csv(".data/FIGR-SVG-valid.csv")
test_df = pd.read_csv(".data/FIGR-SVG-test.csv")

print("Train: ", len(train_df))
print("Val: ", len(val_df))
print("Test: ", len(test_df))

train_subset = train_df.sample(n=270000, replace=False, random_state=42)
val_subset = val_df.sample(n=37500, replace=False, random_state=42)
test_subset = test_df.sample(n=37500, replace=False, random_state=42)

train_subset["split"] = "train"
val_subset["split"] = "val"
test_subset["split"] = "test"

train_subset.to_csv(".data/FIGR-SVG-train-subset.csv", index=False)
val_subset.to_csv(".data/FIGR-SVG-valid-subset.csv", index=False)
test_subset.to_csv(".data/FIGR-SVG-test-subset.csv", index=False)

sampled_df = pd.concat([train_subset, val_subset, test_subset], axis=0)

# get this from here: https://github.com/marcdemers/FIGR-8
all_pngs = glob(".data/FIGR-8/Data/**/*.png", recursive=True)

all_entries = []
for path in all_pngs:
    id = path.split("/")[-1].split(".png")[0]
    class_name = path.split("/")[-2]
    all_entries.append({"id": id, "class": class_name, "png_path": path})

full_df = pd.DataFrame(all_entries)

full_df.index = full_df["id"]
sampled_df.index = sampled_df["id"]

new_sampled_df = sampled_df.join(full_df[["png_path", "class"]], how="inner")
new_sampled_df.to_csv(".data/split.csv", index=False)

print("Done. Saved to .data/split.csv")
