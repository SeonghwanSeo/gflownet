from typing import Any


class IPCModule:
    """Inter-Process Communication Module between GFN sampler and Oracle Module"""

    def __init__(self, process_type: str):
        """Initialization of IPC Module

        Parameters
        ----------
        workdir : str | Path
            working directory of IPCModule
        process_type : str
            - GFNTask: 'sampler'
            - OracleModule: 'oracle'

        """
        self.process_type: str = process_type
        assert self.process_type in (
            "sampler",
            "oracle",
        ), f"process ({self.process_type}) should be 'sampler' or 'oracle"

    @property
    def is_sampler(self):
        return self.process_type == "sampler"

    @property
    def is_oracle(self):
        return self.process_type == "oracle"

    def assert_is_sampler(self):
        assert self.is_sampler

    def assert_is_oracle(self):
        assert self.is_oracle

    ### GFN ###
    def sampler_from_oracle(self) -> tuple[list[list[float]], list[bool]]:
        """Response from Oracle Module for n sampled objects

        Returns
        -------
        Tuple[list[list[float]], list[bool]
            - rewards: [m, n_objectives]
                rewards of n<=m valid objects
            - is_valid: [n,]
                flags whether the objects are valid or not
        """
        raise NotImplementedError

    def sampler_to_oracle(self, objs: list[Any]) -> bool:
        """Request to Oracle Module for rewarding

        Parameters
        ----------
        objs : list[Any]
            A List of n sampled objects

        Returns
        -------
        status: bool
        """
        raise NotImplementedError

    def sampler_wait_oracle(self) -> bool:
        """Wait to receive signal from Oracle."""
        raise NotImplementedError

    def sampler_unlock_oracle(self):
        """Send signal to Oracle."""
        raise NotImplementedError

    def sampler_terminate_oracle(self):
        """Send signal to Oracle to be terminate"""
        raise NotImplementedError

    ### Oracle Module ###
    def oracle_from_sampler(self) -> list[Any]:
        """Receive objects from Sampler process
        if the gflownet process is termiated, it will return None"""
        raise NotImplementedError

    def oracle_to_sampler(self, rewards: list[list[float]], is_valid: list[bool]) -> None:
        """Send reward to Sampler process

        Parameters
        ----------
        rewards: list[list[float]], [m, n_objectives]
            rewards of n<=m valid objects
        is_valid: list[bool], [n,]
            flags whether the objects are valid or not
        """
        raise NotImplementedError

    def oracle_wait_sampler(self) -> bool:
        """Wait to receive signal from Sampler.
        if it return False, the rewarding is finished
        """
        raise NotImplementedError

    def oracle_unlock_sampler(self):
        """Send signal to Sampler."""
        raise NotImplementedError

    def oracle_is_terminated(self):
        """Check the sampler is termiated or not"""
        raise NotImplementedError
