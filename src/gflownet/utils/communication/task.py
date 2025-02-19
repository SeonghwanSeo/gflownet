import time
import torch
from torch import Tensor
from typing import Any

from gflownet.config import Config
from gflownet import GFNTask, ObjectProperties

from gflownet.utils.communication.method import IPCModule, NetworkIPC, FileSystemIPC, FileSystemIPC_CSV


class IPCTask(GFNTask):
    """The rewards of objects are calculated by different processs"""

    def __init__(self, cfg: Config, ipc_module: IPCModule | None = None):
        self.cfg = cfg
        self.setup_ipc_module()
        self.setup_communication()

    def setup_ipc_module(self):
        ipc_cfg = self.cfg.communication
        if ipc_cfg.method == "network":
            ipc_module = NetworkIPC("sampler", ipc_cfg.network.host, ipc_cfg.network.port)
        elif ipc_cfg.method == "file":
            fs_cfg = ipc_cfg.filesystem
            ipc_module = FileSystemIPC("sampler", fs_cfg.workdir, fs_cfg.overwrite_existing_exp)
        elif ipc_cfg.method == "file-csv":
            fs_cfg = ipc_cfg.filesystem
            ipc_module = FileSystemIPC_CSV(
                "sampler", fs_cfg.workdir, fs_cfg.num_objectives, fs_cfg.overwrite_existing_exp
            )
        else:
            raise NotImplementedError
        self.ipc_module: IPCModule = ipc_module

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
        tick = lambda: time.time() - tick_st  # noqa
        while self.ipc_module.sampler_wait_oracle():
            assert tick() <= self._ipc_timeout, f"Timeout! ({tick()} sec)"
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

    def terminate_oracle(self):
        """Terminate Oracle Orocess"""
        self.ipc_module.sampler_terminate_oracle()
