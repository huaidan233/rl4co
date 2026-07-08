import torch

from rl4co.envs.urbanplan.cityplan.env import landuseOptEnv


class landuseOptCompatibilityEnv(landuseOptEnv):
    """LUOP variant that optimizes the compatibility component only.

    The state transition, joint type-parcel action, fixed-parcel protection,
    masks, grouped constraints, and embedding features are inherited from the
    canonical LUOP environment. Only the terminal reward is specialized here.
    """

    name = "lop_compatibility"

    def _default_objective_weights(self, batch_size, device):
        if self.sample_objective_weights or self.objective_weights is not None:
            return super()._default_objective_weights(batch_size, device)
        weights = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
        return self._prepare_objective_weights(weights, batch_size, device)

    def _get_reward(self, td, actions):
        current_plan = td["current_plan"]
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

        normalized = (compatibility - 6.0) / (8.8 - 6.0)
        components = torch.stack(
            [normalized, torch.zeros_like(normalized)],
            dim=-1,
        )
        weights = td.get("objective_weights", None)
        if weights is None:
            weights = self._default_objective_weights(
                current_plan.shape[:-1],
                current_plan.device,
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
