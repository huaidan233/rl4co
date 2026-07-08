from rl4co.envs.urbanplan.cityplan.env import landuseOptEnv


class MAlanduseOptEnv(landuseOptEnv):
    """Backward-compatible multi-agent LUOP entry point.

    Older code imported ``MAlop``/``MAOpt`` for a single-action environment.
    The project now uses one canonical joint type-parcel LUOP action surface,
    so this class intentionally inherits the canonical implementation.
    """

    name = "MAOpt"
