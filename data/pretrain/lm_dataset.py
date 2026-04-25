import os
import torch
import numpy as np

class LMDataset(torch.utils.data.Dataset):
    """ 
    Dataset only provides token sequences.
    The autoregressive shift is applied inside the model during loss computation, 
    keeping data loading independent from training objective.
    """
    def __init__(self, path, block_size=1024, dtype=np.uint16):
        self.block_size = block_size
        file_size = os.path.getsize(path)
        self.total_tokens = file_size // np.dtype(dtype).itemsize
        self.data = np.memmap(path, dtype=dtype, mode="r")

    def __len__(self):
        return self.total_tokens // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size

        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))

        return {"input_ids": chunk}