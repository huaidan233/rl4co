# Base environment
from rl4co.envs.common.base import RL4COEnvBase

# EDA
from rl4co.envs.eda import DPPEnv, MDPPEnv

# Graph
from rl4co.envs.graph import FLPEnv, MCPEnv

# Routing
from rl4co.envs.routing import (
    ATSPEnv,
    CVRPEnv,
    CVRPMVCEnv,
    CVRPTWEnv,
    DenseRewardTSPEnv,
    MDCPDPEnv,
    MTSPEnv,
    MTVRPEnv,
    OPEnv,
    PCTSPEnv,
    PDPEnv,
    PDPRuinRepairEnv,
    SDVRPEnv,
    SHPPEnv,
    SPCTSPEnv,
    SVRPEnv,
    TSPEnv,
    TSPkoptEnv,
)

# Scheduling
from rl4co.envs.scheduling import FFSPEnv, FJSPEnv, JSSPEnv, SMTWTPEnv

# Urban planning
from rl4co.envs.urbanplan.MAcityplan.env import MAlanduseOptEnv
from rl4co.envs.urbanplan.cityplan.env import landuseOptEnv

# Register environments
ENV_REGISTRY = {
    "atsp": ATSPEnv,
    "cvrp": CVRPEnv,
    "cvrptw": CVRPTWEnv,
    "cvrpmvc": CVRPMVCEnv,
    "dpp": DPPEnv,
    "ffsp": FFSPEnv,
    "jssp": JSSPEnv,
    "fjsp": FJSPEnv,
    "mdpp": MDPPEnv,
    "mtsp": MTSPEnv,
    "op": OPEnv,
    "pctsp": PCTSPEnv,
    "pdp": PDPEnv,
    "pdp_ruin_repair": PDPRuinRepairEnv,
    "sdvrp": SDVRPEnv,
    "shpp": SHPPEnv,
    "svrp": SVRPEnv,
    "spctsp": SPCTSPEnv,
    "tsp": TSPEnv,
    "smtwtp": SMTWTPEnv,
    "mdcpdp": MDCPDPEnv,
    "mtvrp": MTVRPEnv,
    "tsp_kopt": TSPkoptEnv,
    "lop": landuseOptEnv,
    "MAlop": MAlanduseOptEnv,
}


def get_env(env_name: str, *args, **kwargs) -> RL4COEnvBase:
    """Get environment by name.

    Args:
        env_name: Environment name
        *args: Positional arguments for environment
        **kwargs: Keyword arguments for environment

    Returns:
        Environment
    """
    env_cls = ENV_REGISTRY.get(env_name, None)
    if env_cls is None:
        raise ValueError(
            f"Unknown environment {env_name}. Available environments: {ENV_REGISTRY.keys()}"
        )
    return env_cls(*args, **kwargs)
