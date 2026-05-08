import os
import pandas as pd

root = "/mnt/oldssd/train/highway/"
section_ids = os.listdir(root)

dfs = []

for section_id in section_ids:
    path = os.path.join(
        root,
        section_id,
        "dynamic",
        "raw-revolutions"
    )

    frame_id = [
        str(f.split("_")[1].split(".")[0])
        for f in os.listdir(path)
        if f.startswith("frame_") and f.endswith(".laz")
    ]

    dfs.append(pd.DataFrame({
        "section_id": section_id,
        "frame_id": frame_id
    }))

df = pd.concat(dfs, ignore_index=True)

print(df)

df.to_csv("id_data.csv")
