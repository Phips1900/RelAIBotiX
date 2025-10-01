#!/usr/bin/env python3
"""
view_h5.py
Quick inspection tool for LeRobot HDF5 logs.
Hard-coded for rl and file lerobot_mj0.h5
"""

import h5py
import pandas as pd

# 🔒 Hard-coded name and file
AUTHOR_NAME = "name"
H5_FILE = "eval_act_mj2.h5"
NROWS = 10  # how many rows to preview


def load_h5(filename: str):
    with h5py.File(filename, "r") as f:
        # Extract datasets
        timestamps = f["timestamps"][:]
        features = f["features"][:]
        labels = f["labels"][:]

        # Handle feature names (bytes or str)
        raw_names = f["features"].attrs.get("feature_names", [])
        feature_names = [
            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
            for n in raw_names
        ]

    return timestamps, features, labels, feature_names


def main():
    timestamps, features, labels, feature_names = load_h5(H5_FILE)

    print(f"=== HDF5 Log Summary (by {AUTHOR_NAME}) ===")
    print(f"File: {H5_FILE}")
    print(f"Timestamps shape: {timestamps.shape}")
    print(f"Features shape:   {features.shape}")
    print(f"Labels shape:     {labels.shape}")
    print(f"Feature names:    {feature_names}")

    # Convert to DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    df["timestamp"] = timestamps
    df["label"] = labels

    print("\n=== First rows ===")
    print(df.head(NROWS))

    # Save to CSV for easier browsing
    csv_filename = H5_FILE.replace(".h5", f"_{AUTHOR_NAME}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved CSV preview to {csv_filename} (authored by {AUTHOR_NAME})")

    return df


if __name__ == "__main__":
    main()
