import operator
from functools import reduce
from multiprocessing import shared_memory
from typing import Dict, Optional, List
import dataclasses

import torch


@dataclasses.dataclass
class ModelSharer:
    theta: Dict
    owner: bool
    memory: Optional[shared_memory.SharedMemory] = None

    def __post_init__(self):
        self.offset = 0

    def __enter__(self):
        assert self.offset == 0, "cannot reuse ModelSharer instances"

        memory_kwargs = {}
        if self.owner:
            storage_size = 0
            for v in self.theta.values():
                storage_size = align_offset(storage_size + v.untyped_storage().nbytes())

            memory_kwargs["size"] = storage_size

        self.memory = shared_memory.SharedMemory(
            create=self.owner,
            name=f"bbwm-model-bytes",
            **memory_kwargs,
        )
        return self

    def serialize(self, k: str) -> None:
        v = self.theta[k]
        next_offset = self.offset + v.numpy().nbytes
        self.memory.buf[self.offset:next_offset] = v.numpy(force=True).tobytes()
        self.offset = align_offset(next_offset)

    def deserialize(self, shape: List[int], dtype: torch.dtype):
        count = reduce(operator.mul, shape, 1)
        res = torch.frombuffer(
            offset=self.offset,
            buffer=self.memory.buf,
            count=count,
            dtype=dtype,
        ).reshape(shape)
        self.offset = align_offset(self.offset + count * res.element_size())
        return res

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.memory is not None:
            self.memory.close()
            if self.owner:
                self.memory.unlink()


def align_offset(offset: int) -> int:
    return offset + (8 - offset % 8) % 8
