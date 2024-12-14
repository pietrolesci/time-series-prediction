import argparse
from pathlib import Path

import polars as pl
import polars.selectors as cs

# from datasets import Dataset as HFDataset, DatasetDict
from src.utilities import TOTAL_NUMBER_OF_LEVELS, create_memmap_from_polars

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/data.parquet")
    args = parser.parse_args()

    # Read L2 data
    data_path = Path(args.data_path)
    df = pl.read_parquet(data_path)

    # Rescale prices by 1000 (so that they are around 1)
    df = df.with_columns(cs.contains("Rate") / 1000, cs.contains("Size").log())

    # Fill nulls with 0
    df = df.with_columns((cs.contains("Rate") | cs.contains("Size")).fill_null(0))

    # Reorder cols
    cols = ["y"]
    for i in range(TOTAL_NUMBER_OF_LEVELS):
        for side in ["ask", "bid"]:
            cols += [f"{side}Rate{i}", f"{side}Size{i}"]

    df = df.select(cols)

    # train ~80%, val ~7% (250k), test ~8.5% (300k), ~4.5% is left out between train and test
    ds_dict: dict[str, pl.DataFrame] = {"train": df[:2_800_000], "val": df[2_900_000:3_150_000], "test": df[3_200_000:]}

    for k, v in ds_dict.items():
        create_memmap_from_polars(str(data_path.parent / f"{k}_memmap.npy"), v)
