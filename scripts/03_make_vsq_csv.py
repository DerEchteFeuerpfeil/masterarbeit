import pandas as pd
import os
import glob

assert os.path.basename(os.getcwd()) == "masterarbeit", "Change working directory to the root of the repository."

train_subset = pd.read_csv(".data/FIGR-SVG-train-subset.csv")
val_subset = pd.read_csv(".data/FIGR-SVG-valid-subset.csv")
test_subset = pd.read_csv(".data/FIGR-SVG-test-subset.csv")

# add path to the simplified pre-processed svgs here
all_paths = glob.glob(".data/svgs_simplified/**/*.svg")

# get all paths
all_entries = []
for path in all_paths:
    id = path.split("/")[-1].split(".svg")[0]
    class_name = path.split("/")[-2]
    all_entries.append({
        "id": id, 
        "class": class_name, 
        "simplified_svg_file_path": path, 
        })

full_df = pd.DataFrame(all_entries)

# join local paths with the split/descriptions from IconShop
train_subset.index = train_subset["id"]
val_subset.index = val_subset["id"]
test_subset.index = test_subset["id"]
full_df.index = full_df["id"]

new_train_subset = train_subset.join(full_df[["simplified_svg_file_path", "class"]], how="left").reset_index(drop=True)
new_train_subset["split"] = "train"

new_val_subset = val_subset.join(full_df[["simplified_svg_file_path", "class"]], how="left").reset_index(drop=True)
new_val_subset["split"] = "val"

new_test_subset = test_subset.join(full_df[["simplified_svg_file_path", "class"]], how="left").reset_index(drop=True)
new_test_subset["split"] = "test"

# save locally for training
final_df = pd.concat([new_train_subset, new_val_subset, new_test_subset], axis=0)
final_df["description"] = final_df["desc"]
final_df = final_df[final_df["simplified_svg_file_path"].notnull()].reset_index(drop=True)
final_df.to_csv(".data/FIGR-SVG-VSQ-split.csv", index=False)
final_df.to_csv(".data/stage1.csv", index=False)

# save subset for param sweep
sweep_df = final_df[final_df["split"].isin(["train", "val"])].sample(3750, random_state=42)
sweep_df.to_csv(".data/vsq_sweep.csv", index=False)