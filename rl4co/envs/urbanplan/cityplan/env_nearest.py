import torch

from rl4co.envs.urbanplan.cityplan import init
from rl4co.envs.urbanplan.cityplan.env import landuseOptEnv

class landuseOptNearestEnv(landuseOptEnv):
    """LUOP variant using nearest-facility accessibility.

    It shares the canonical joint type-parcel action surface and constraint
    masks with :class:`landuseOptEnv`; only the accessibility reward component
    differs.
    """

    name = "lop_nearest"

    def _get_reward(self, td, actions):
        current_plan = td["current_plan"]
        distances = td["distances"]
        neighbors = td["adjacency_list"]
        if self._is_adjacency_matrix_tensor(neighbors):
            neighbors = self.adj_matrix_to_list(neighbors)

        compatibility_table = torch.tensor(
            self.CompatibilityTable,
            dtype=torch.float32,
            device=current_plan.device,
        )
        if isinstance(neighbors, list):
            compatibility = self.calc_compatibility_tensor_real(
                current_plan, neighbors, compatibility_table
            )
        else:
            compatibility = self.calc_compatibility_tensor(
                current_plan, neighbors, compatibility_table
            )
        accessibility = self.calc_accessibility_nearest_tensor(
            current_plan, distances
        )

        norcompatibility = (compatibility - 6.0) / (8.8 - 6.0)
        noraccessibility = (accessibility - (-0.6)) / (-0.1 - (-0.6))
        components = torch.stack([norcompatibility, noraccessibility], dim=-1)
        weights = td.get("objective_weights", None)
        if weights is None:
            weights = self._default_objective_weights(
                current_plan.shape[:-1], current_plan.device
            )
        weights = self._prepare_objective_weights(
            weights,
            current_plan.shape[:-1],
            current_plan.device,
            dtype=components.dtype,
        )
        td.set("reward_components", components)
        td.set("objective_weights", weights)
        return self._scalarize_objectives(components, weights)

    def calc_accessibility_nearest_tensor(self, plan, distancelist):
        residential_types = ["Residential", "SOHO", "Residential&Commercial"]
        return (
            init.calInterAccessibility_Nearest_tensor(
                plan,
                residential_types,
                ["Commercial", "Residential&Commercial"],
                distancelist,
            )
            + init.calInterAccessibility_Nearest_tensor(
                plan, residential_types, ["Education"], distancelist
            )
            + init.calInterAccessibility_Nearest_tensor(
                plan, residential_types, ["Hospital"], distancelist
            )
            + init.calInterAccessibility_Nearest_tensor(
                plan, residential_types, ["Office", "SOHO"], distancelist
            )
            + init.calInterAccessibility_Nearest_tensor(
                plan, residential_types, ["Green Space"], distancelist
            )
        )
