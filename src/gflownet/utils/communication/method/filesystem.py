import os
import pickle
from pathlib import Path
from typing import Any

from gflownet.utils.communication.method.base import IPCModule


class FileSystemIPC(IPCModule):
    """IPCModule using FileSystem"""

    def __init__(self, process_type: str, workdir: str | Path, overwrite_existing_exp: bool = False):
        """Initialization of IPC Module

        Parameters
        ----------
        workdir : str | Path
            working directory of IPCModule
        process_type : str
            - GFNTask: 'sampler'
            - OracleModule: 'oracle'

        """
        super().__init__(process_type)
        self.root_dir = Path(workdir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.root_dir / "wait.lock"
        self.terminate_file = self.root_dir / "termiate"
        self.sampler_to_oracle_file = self.root_dir / "objects.pkl"
        self.oracle_to_sampler_file = self.root_dir / "rewards.pkl"

        # Only one process for a single gflownet process
        process_lock = self.root_dir / f"process_{self.process_type}.lock"

        # Overwrite workdir, unsafe
        if overwrite_existing_exp:
            if process_lock.exists():
                os.remove(process_lock)
            if self.terminate_file.exists():
                os.remove(self.terminate_file)
            if self.is_sampler:
                if self.sampler_to_oracle_file.exists():
                    os.remove(self.sampler_to_oracle_file)
                if self.lock_file.exists():
                    os.remove(self.lock_file)
            if self.oracle_to_sampler_file.exists():
                os.remove(self.oracle_to_sampler_file)

        assert not process_lock.exists(), "there is existing process"
        process_lock.touch()

    ### GFN ###
    def sampler_from_oracle(self) -> tuple[list[list[float]], list[bool]]:
        self.assert_is_sampler()
        with open(self.oracle_to_sampler_file, "rb") as f:
            rewards, is_valid = pickle.load(f)
        os.remove(self.oracle_to_sampler_file)
        return rewards, is_valid

    def sampler_to_oracle(self, objs: list[Any]) -> bool:
        self.assert_is_sampler()
        with self.sampler_to_oracle_file.open("wb") as w:
            pickle.dump(objs, w)
        return True

    def sampler_wait_oracle(self) -> bool:
        self.assert_is_sampler()
        return self.lock_file.exists()

    def sampler_unlock_oracle(self):
        self.assert_is_sampler()
        self.lock_file.touch()

    def sampler_terminate_oracle(self):
        self.assert_is_sampler()
        self.terminate_file.touch()

    ### Oracle Module ###
    def oracle_from_sampler(self) -> list[Any]:
        self.assert_is_oracle()
        with self.sampler_to_oracle_file.open("rb") as f:
            objs = pickle.load(f)
        os.remove(self.sampler_to_oracle_file)
        return objs

    def oracle_to_sampler(self, rewards: list[list[float]], is_valid: list[bool]):
        self.assert_is_oracle()
        with self.oracle_to_sampler_file.open("wb") as w:
            pickle.dump((rewards, is_valid), w)

    def oracle_wait_sampler(self) -> bool:
        self.assert_is_oracle()
        """Wait to receive signal from Sampler. (wait during True)"""
        return not self.lock_file.exists()

    def oracle_unlock_sampler(self):
        self.assert_is_oracle()
        """Send signal to Sampler."""
        os.remove(self.lock_file)

    def oracle_is_terminated(self):
        self.assert_is_oracle()
        return self.terminate_file.exists()


class FileSystemIPC_CSV(FileSystemIPC):
    """message passing using text(csv) file instead of pkl
    it would be useful for non-python rewards (e.g., experimental reward, non-python reward)
    """

    def __init__(
        self,
        process_type: str,
        workdir: str | Path,
        num_objectives: int,
        overwrite_existing_exp: bool = False,
    ):
        super().__init__(process_type, workdir, overwrite_existing_exp)
        self.num_objectives = num_objectives
        # override
        self.sampler_to_oracle_file = self.root_dir / "objects.csv"
        self.oracle_to_sampler_file = self.root_dir / "rewards.csv"

    ### Implement Here ###
    def object_to_str_repr(self, obj: Any) -> str:
        """Convert an Object to a string representation
        e.g., rdkit.Chem.Mol
            return Chem.MolToSmiles(obj)
        """
        raise NotImplementedError

    def str_repr_to_object(self, obj_repr: str) -> Any:
        """Convert an Object to a string representation
        e.g., rdkit.Chem.Mol
            return Chem.MolFromSmiles(obj_repr)
        """
        return obj_repr

    ### GFN ###
    def sampler_from_oracle(self) -> tuple[list[list[float]], list[bool]]:
        self.assert_is_sampler()
        rewards = []
        is_valid = []
        with self.oracle_to_sampler_file.open() as f:
            lines = f.readlines()
        for ln in lines[1:]:
            splits = ln.split(",")
            valid = splits[1].lower() == "true"
            is_valid.append(valid)
            if valid:
                rs = list(map(float, splits[2:]))
                rewards.append(rs)
        os.remove(self.oracle_to_sampler_file)
        return rewards, is_valid

    def sampler_to_oracle(self, objs: list[Any]) -> bool:
        self.assert_is_sampler()
        with self.sampler_to_oracle_file.open("w") as w:
            w.write(",object\n")
            for i, obj in enumerate(objs):
                obj_repr = self.object_to_str_repr(obj)
                w.write(f"{i},{obj_repr}\n")
        return True

    ### Oracle Module ###
    def oracle_from_sampler(self) -> list[Any]:
        self.assert_is_oracle()
        with self.sampler_to_oracle_file.open() as f:
            lines = f.readlines()[1:]
        objs = [self.str_repr_to_object(ln.split(",")[1].strip()) for ln in lines]
        os.remove(self.sampler_to_oracle_file)
        return objs

    def oracle_to_sampler(self, rewards: list[list[float]], is_valid: list[bool]):
        self.assert_is_oracle()
        columns = ["", "is_valid"] + [f"r{i}" for i in range(self.num_objectives)]
        with self.oracle_to_sampler_file.open("w") as w:
            w.write(",".join(columns) + "\n")
            t = 0
            for i in range(len(is_valid)):
                valid = is_valid[i]
                if valid:
                    rs = [str(r) for r in rewards[t]]
                    t += 1
                else:
                    rs = ["0.0"] * self.num_objectives
                w.write(f"{i},{valid}," + ",".join(rs) + "\n")
