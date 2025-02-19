from pathlib import Path
from typing import Dict
from collections import OrderedDict

import torch
import numpy as np
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch_geometric.data import Data

from gflownet.config import Config, init_empty
from gflownet.models import bengio2021flow
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward
from gflownet import LogScalar, ObjectProperties

from gflownet.tasks.seh_frag import SEHFragTrainer

from gflownet.communicate.ipc import IPCModule, FileSystemIPC
from gflownet.communicate.task import IPCTask
from gflownet.communicate.oracle import OracleModule


class LogitGFN_IPCTask(IPCTask):
    """IPCTask for temperature-conditioned GFlowNet (LogitGFN)"""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def setup_ipc_module(self, workdir: str | Path):
        """Create the ipc module here"""
        self.ipc_module: IPCModule = FileSystemIPC(workdir, "sampler")

    def setup_communication(self):
        """Set the communication settings"""
        self._ipc_timeout = 60  # wait up to 1 min for oracle function
        self._ipc_tick = 0.1  # 0.1 sec

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(obj_props)))


class SEHOracle(OracleModule):
    """Oracle Module which communicates with trainer running on the other process."""

    def __init__(self, gfn_log_dir, verbose: bool = True, device: str = "cuda"):
        super().__init__(gfn_log_dir, verbose)
        self.model = bengio2021flow.load_original_model().to(device)
        self.device = device

    def setup_ipc_module(self, workdir: str | Path):
        """Create the ipc module here"""
        self.ipc_module: IPCModule = FileSystemIPC(workdir, "oracle")

    def setup_communication(self):
        """Set the communication settings"""
        self._ipc_timeout = 60  # wait up to 1 min for gflownet
        self._ipc_tick = 0.1  # 0.1 sec

    @property
    def num_objectives(self) -> int:
        return 1

    def convert_object(self, obj: RDMol) -> Data:
        return bengio2021flow.mol2graph(obj)

    def filter_object(self, obj: Data) -> bool:
        return obj is not None

    def compute_reward_batch(self, objs: list[Data]) -> list[list[float]]:
        """Modify here if parallel computation is required

        Parameters
        ----------
        objs : list[Graph]
            A list of valid graphs

        Returns
        -------
        rewards_list: list[list[float]]
            Each item of list should be list of reward for each objective
            assert len(rewards_list) == len(objs)
        """
        batch = gd.Batch.from_data_list(objs)
        batch.to(self.device)
        preds = self.model(batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        preds = preds.clip(1e-4, 100).reshape(-1, 1)
        return preds.tolist()

    def log(self, objs: list[Data], rewards: list[list[float]], is_valid: list[bool]):
        info = OrderedDict()
        info["num_objects"] = len(is_valid)
        info["invalid_trajectories"] = 1 - np.mean(is_valid)
        info["sampled_rewards_avg"] = np.mean(rewards) if len(rewards) > 0 else 0.0
        self.logger.info(f"iteration {self.oracle_idx} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))


class SEHFragTrainer_IPC(SEHFragTrainer):
    task: LogitGFN_IPCTask

    def setup_task(self):
        self.task = LogitGFN_IPCTask(cfg=self.cfg)


def main_gfn(log_dir: str):
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = log_dir
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    config.num_training_steps = 1_00
    config.validate_every = 20
    config.num_final_gen_steps = 10
    config.num_workers = 1
    config.opt.lr_decay = 20_000
    config.algo.sampling_tau = 0.99
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64.0]

    trial = SEHFragTrainer_IPC(config)
    trial.run()
    trial.task.terminate_oracle()


def main_oracle(log_dir: str):
    """Example of how this oracle function can be run."""
    oracle = SEHOracle(log_dir, device="cuda")
    oracle.run()


if __name__ == "__main__":
    import sys

    process = sys.argv[1]
    assert process in ("gfn", "oracle")

    log_dir = "./logs/debug_run_seh_frag_ipc"
    if process == "gfn":
        main_gfn(log_dir)
    else:
        main_oracle(log_dir)
