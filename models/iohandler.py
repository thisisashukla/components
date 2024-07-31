import os
import pandas as pd
from typing import Union
import traceback


class IO_HANDLER:
    def __init__(self, base_path: str):

        self.base_path = base_path
        self.output_path = f"{base_path}/outputs"

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def read_data(self, name: str):

        df = pd.read_parquet(f"{self.base_path}/{name}.parquet")

        return df

    def write_data(self, df: pd.DataFrame, name: str):

        df.to_parquet(f"{self.output_path}/{name}.parquet", index=False)

    def read_data_sample(
        self, name: str, sample_size: Union[int, float], nsamples: int = 1
    ):

        df = self.read_data(name)

        sample_rows = 0
        if isinstance(sample_size, int):
            sample_rows = sample_size
        else:
            sample_rows = int(df.shape[0] * sample_size)

        samples = []
        for i in range(nsamples):
            samples.append(df.sample(sample_rows))

        return samples
