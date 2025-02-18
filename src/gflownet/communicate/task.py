import time
from pathlib import Path
import torch
from torch import Tensor

from typing import Any

from gflownet.config import Config
from gflownet import GFNTask, ObjectProperties

from gflownet.communicate.ipc import IPCModule, FileSystemIPC


class IPCTask(GFNTask):
    """The rewards of objects are calculated by different processs"""

    def __init__(self, config: Config):
        self.config = config
        self.setup_ipc_module(Path(config.log_dir) / "_ipc")
        self.setup_communication()

    def setup_ipc_module(self, workdir: str | Path):
        """Create the ipc module here"""
        self.ipc_module: IPCModule = FileSystemIPC(workdir, "sampler")

    def setup_communication(self):
        """Set the communication settings"""
        self._ipc_timeout = 600  # wait up to 10 min for oracle function
        self._ipc_tick = 0.1  # 0.1 sec

    def compute_obj_properties(self, objs: list[Any]) -> tuple[ObjectProperties, Tensor]:
        assert self.to_oracle(objs)
        self.wait_oracle()
        rewards, is_valid = self.from_oracle()
        assert len(objs) == is_valid.shape[0]
        assert rewards.shape[0] == is_valid.sum()
        return ObjectProperties(rewards), is_valid

    def to_oracle(self, objs: list[Any]) -> bool:
        """Send objects to Oracle Process

        Parameters
        ----------
        objs : list[Any]
            A list of n sampled objects

        Returns
        -------
        status: bool
        """
        flag = self.ipc_module.sampler_to_oracle(objs)
        self.ipc_module.sampler_unlock_oracle()
        return flag

    def wait_oracle(self) -> bool:
        """Wait Oracle Process"""
        tick_st = time.time()
        while self.ipc_module.oracle_wait_sampler():
            tick = time.time() - tick_st
            assert tick <= self._ipc_timeout, f"Timeout! ({tick} sec)"
            time.sleep(self._ipc_tick)
        return True

    def from_oracle(self) -> tuple[Tensor, Tensor]:
        """Receive rewards from Oracle Process
        Returns
        -------
        tuple[Tensor, Tensor]
            - rewards: FloatTensor [m, n_objectives]
                rewards of n<=m valid objects
            - is_valid: BoolTensor [n,]
                flags whether the objects are valid or not
        """
        fr, is_valid = self.ipc_module.sampler_from_oracle()
        return torch.tensor(fr, dtype=torch.float32), torch.tensor(is_valid, dtype=torch.bool)
