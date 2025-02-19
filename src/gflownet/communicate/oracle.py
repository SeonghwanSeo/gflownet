import sys
import numpy as np
import time
import logging
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf

from gflownet.communicate.ipc import IPCModule, FileSystemIPC


class OracleModule:
    """Oracle Module which communicates with trainer running on the other process.
    To avoid the dependency issues, gflownet sources are not imported.
    Therefore, it is available to run without torch library.
    If you want to run reward function on your own dependencies or environments, copy this class.
    """

    def __init__(self, gfn_log_dir: str | Path, num_oracles: int, verbose_level: int = 1):
        # gflownet log dir
        self.gfn_log_dir = Path(gfn_log_dir)
        self.config_path = self.gfn_log_dir / "config.yaml"

        # Wait while the gflownet start
        for _ in range(60):  # 1 min
            if not self.config_path.exists():
                time.sleep(1)
        self.cfg = OmegaConf.load(self.config_path)

        # logging
        if verbose_level == 0:
            loglevel = logging.WARNING
        elif verbose_level == 1:
            loglevel = logging.INFO
        else:
            loglevel = logging.DEBUG
        self.logger: logging.Logger = self.create_logger("oracle", loglevel)
        self.oracle_idx = 0

        # create ipc module
        self.setup_ipc_module(self.gfn_log_dir / "_ipc")
        self.setup_communication()

    def setup_ipc_module(self, workdir: str | Path):
        self.ipc_module: IPCModule = FileSystemIPC(workdir, "oracle")

    def setup_communication(self):
        """Write here the communication settings"""
        self._ipc_timeout = 60  # wait up to 1 min for gflownet
        self._ipc_tick = 0.1  # 0.1 sec

    """Running API"""

    def run(self):
        while self.wait_sampler():
            # communication: receive objects
            objs = self.from_sampler()
            # compute rewards
            rewards, is_valid = self.compute_obj_properties(objs)
            # communication: send rewards
            self.to_sampler(rewards, is_valid)
            # logging (async)
            self.log(objs, rewards, is_valid)
            self.oracle_idx += 1

    def compute_obj_properties(self, objs: list[Any]) -> tuple[list[list[float]], list[bool]]:
        self.logger.debug(f"receive {len(objs)} objects")
        # convert and filter the objects before reward calculation
        st = time.time()
        converted_objs = [self.convert_object(obj) for obj in objs]
        is_valid = [self.filter_object(obj) for obj in converted_objs]
        valid_objs = [obj for flag, obj in zip(is_valid, converted_objs, strict=True) if flag]
        tick = time.time() - st
        self.logger.debug(f"get the {len(valid_objs)} valid objects (filter: {tick:.3f} sec)")

        # reward calculation
        st = time.time()
        rewards = self._compute_rewards(valid_objs)
        tick = time.time() - st
        self.logger.debug(f"finish reward calculation! (reward: {tick:.3f} sec)")
        return rewards, is_valid

    """Implement following methods!"""

    @property
    def num_objectives(self) -> int:
        raise NotImplementedError

    def convert_object(self, obj: Any) -> Any | None:
        """Implement the conversion function if required

        Parameters
        ----------
        obj : Any
            A valid object(graph, seq, mol, ...)

        Returns
        -------
        obj: Any
            return a converted object
        """
        return obj

    def filter_object(self, obj: Any) -> bool:
        """Implement the constraint here if required

        Parameters
        ----------
        obj : Any
            A valid object(graph, seq, mol, ...)

        Returns
        -------
        is_valid: bool
            return whether the object is valid or not
        """
        return obj is not None

    def compute_reward_batch(self, objs: list[Any]) -> list[list[float]]:
        """Modify here if parallel computation is required

        Parameters
        ----------
        objs : list[Any]
            A list of valid objects(graphs, seqs, mols, ...)

        Returns
        -------
        rewards_list: list[list[float]]
            Each item of list should be list of reward for each objective
            assert len(rewards_list) == len(objs)
        """
        return [self.compute_reward_single(obj) for obj in objs]

    def compute_reward_single(self, obj: Any) -> list[float]:
        """Implement the reward function which calculates the reward of each object individually

        Parameters
        ----------
        obj : Any
            A valid object(graph, seq, mol, ...)

        Returns
        -------
        rewards: list[float]
            It shoule be list of the property for each objective
            The negative value would be clipped to 0 because GFlowNets require a non-negative reward.
            assert len(reward) == self.num_objectives
        """
        raise NotImplementedError

    def log(self, objs: list[Any], rewards: list[list[float]], is_valid: list[bool]):
        """Log Hook

        Parameters
        ----------
        objs : list[Any]
            A list of objects
        rewards : list[list[float]]
            A list of reward for each object
        is_valid : list[bool]
            A list of valid flag of each objects
        """
        pass

    """Inner function"""

    def wait_sampler(self) -> bool:
        tick_st = time.time()
        tick = lambda: time.time() - tick_st  # noqa
        while self.ipc_module.oracle_wait_sampler():
            if self.ipc_module.oracle_is_terminated():
                self.logger.info("Sampler is terminated")
                return False
            if tick() > self._ipc_timeout:
                self.logger.warning(f"Timeout! ({tick()} sec)")
                return False
            time.sleep(self._ipc_tick)
        return True

    def from_sampler(self) -> list[Any]:
        """Receive objects from GFlowNet process
        if the gflownet process is terminated, it will return None"""
        return self.ipc_module.oracle_from_sampler()

    def to_sampler(self, rewards: list[list[float]], is_valid: list[bool]):
        """Send reward to GFlowNet process"""
        self.ipc_module.oracle_to_sampler(rewards, is_valid)
        self.ipc_module.oracle_unlock_sampler()

    def _compute_rewards(self, objs: list[Any]) -> list[list[float]]:
        """To prevent unsafe actions"""
        if len(objs) == 0:
            return []
        else:
            rs = self.compute_reward_batch(objs)
            for r in rs[1:]:
                assert (
                    len(r) == self.num_objectives
                ), f"The length of reward ({len(r)}) should be same to the number of objectives ({self.num_objectives})"
            assert len(rs) == len(
                objs
            ), f"The number of outputs {len(rs)} should be same to the number of samples ({len(objs)})"
            return rs

    def create_logger(self, name="logger", loglevel=logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(loglevel)
        while len([logger.removeHandler(i) for i in logger.handlers]):
            pass  # Remove all handlers (only useful when debugging)
        formatter = logging.Formatter(
            fmt=f"%(asctime)s - %(levelname)s - {name} - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
