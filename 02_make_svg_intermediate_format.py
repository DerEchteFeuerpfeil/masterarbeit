import os.path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
from svglib.svg import SVG
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent import futures
from skimage import measure
from glob import glob

assert os.path.basename(os.getcwd()) == "masterarbeit", "Change working directory to the root of the repository."

# here should lie the split.csv from previous step, there will be svgs and simplified svgs placed here after running this script
ROOT_DIR = ".data"


def convert(thread_index, block:pd.DataFrame):
    total_skips = 0
    ids, simplified_svg_paths, svg_paths = [], [], []
    for idx, sample in tqdm(block.iterrows(), total=len(block), disable=(thread_index != 0)):

        png_path = sample["png_path"]
        path_parts = png_path.split("/")
        folder = path_parts[-2]
        filename = path_parts[-1]

        # simplified SVG path
        simp_svg_filepath = os.path.join(ROOT_DIR, "svgs_simplified", folder, filename.replace(".png", ".svg"))
        svg_filepath = os.path.join(ROOT_DIR, "svg", folder, filename.replace(".png", ".svg"))
        if not os.path.exists(simp_svg_filepath):

            img = cv2.imread(png_path)
            # img = 255 - img
            w, h, _ = img.shape
            # add a border to ensure outline contours are not lost for shapes that touch the borders
            border_size = 5
            img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ret, thresh = cv2.threshold(img, 25, 255, 0)
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = measure.find_contours(img)
            if len(contours) == 0:
                total_skips += 1
                continue


            # ############
            # # simplified
            Path(os.path.join(ROOT_DIR, "svgs_simplified", folder)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(ROOT_DIR, "svg", folder)).mkdir(parents=True, exist_ok=True)

            svg_file = f'<svg width="100%" height="100%" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            for c in contours:
                svg_file += '<path d="M'
                for i in range(len(c)):
                    y, x = c[i]
                    svg_file += f"{x} {y} "
                svg_file += '" fill="none" stroke="#000000" stroke-width="2"/>'
            svg_file += "</svg>"
            svg = SVG.from_str(svg_file)
            svg.fill_(False)
            svg.normalize()
            svg.zoom(0.9)
            
            # before further preprocessing, save the full svg
            svg.save_svg(svg_filepath)

            # apply DeepSVG preprocessing and save the simplified svg
            svg.canonicalize()
            svg = svg.simplify_heuristic(epsilon=0.001)
            svg.save_svg(simp_svg_filepath)

        # logging this new file
        ids.append(sample["id"])
        simplified_svg_paths.append(simp_svg_filepath)
        svg_paths.append(svg_filepath)


    new_csv = pd.DataFrame({
                "id": ids,
                "simplified_svg_path": simplified_svg_paths,
                "svg_path": svg_paths,
            })
    new_csv.to_csv(os.path.join(ROOT_DIR, "svgs_simplified", f"{thread_index}_thread.csv"), index=False)
    print(f"Thread {thread_index} completed. Skipped {total_skips} samples.")


if __name__ == '__main__':

    if os.path.exists(os.path.join(ROOT_DIR, "split.csv")):
        df = pd.read_csv(os.path.join(ROOT_DIR, "split.csv"))
        assert "png_path" in df.columns, "Please run 01_make_data_split.py first."
        assert len(df) > 1000, "The split.csv dataframe seems too small"
    else:
        raise ValueError(f"No split.csv found in {ROOT_DIR}. Please run 01_make_data_split.py first.")

    print("Vectorizing and simplifying...")
    shards = 20
    print(f"Using {shards} shards, a thread for each of them.")
    df_portions = np.array_split(df, shards)

    with futures.ProcessPoolExecutor(max_workers=len(df_portions)) as executor:
        thread_indices = range(len(df_portions))
        preprocess_requests = [executor.submit(convert, thread_index, df_portion)
                               for thread_index, df_portion in zip(thread_indices, df_portions)]

        # Wait for all the tasks to complete
        for future in preprocess_requests:
            future.result()  # This blocks until the future is done
            
    print("Process completed. merging.")
    csv_files = [f for f in os.listdir(os.path.join(ROOT_DIR, "svgs_simplified")) if f.endswith("_thread.csv")]
    merged_df = pd.concat([pd.read_csv(os.path.join(ROOT_DIR, "svgs_simplified", csv_file)) for csv_file in csv_files],
                          ignore_index=True)
    
    save_pos = os.path.join(ROOT_DIR, "svg_file_paths.csv")
    merged_df.to_csv(save_pos, index=False)
    print("merged all svgs in ", save_pos)
