import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.env_embeddings.init import prepare_lop_objective_weights
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def env_dynamic_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment dynamic embedding. The dynamic embedding is used to modify query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": StaticEmbedding,
        "atsp": StaticEmbedding,
        "cvrp": StaticEmbedding,
        "cvrptw": StaticEmbedding,
        "ffsp": StaticEmbedding,
        "svrp": StaticEmbedding,
        "sdvrp": SDVRPDynamicEmbedding,
        "pctsp": StaticEmbedding,
        "spctsp": StaticEmbedding,
        "op": StaticEmbedding,
        "dpp": StaticEmbedding,
        "mdpp": StaticEmbedding,
        "pdp": StaticEmbedding,
        "mtsp": StaticEmbedding,
        "smtwtp": StaticEmbedding,
        "jssp": JSSPDynamicEmbedding,
        "fjsp": JSSPDynamicEmbedding,
        "mtvrp": StaticEmbedding,
        "luop": LOPDynamicEmbedding,
        "lop": LOPDynamicEmbedding,
        "lop_nearest": LOPDynamicEmbedding,
        "lop_compatibility": LOPDynamicEmbedding,
        "MAlop": LOPDynamicEmbedding,
        "MAOpt": LOPDynamicEmbedding,
    }

    if env_name not in embedding_registry:
        log.warning(
            f"Unknown environment name '{env_name}'. Available dynamic embeddings: {embedding_registry.keys()}. Defaulting to StaticEmbedding."
        )
    return embedding_registry.get(env_name, StaticEmbedding)(**config)


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, td):
        return 0, 0, 0


class LOPDynamicEmbedding(nn.Module):
    """Dynamic embedding for the Landuse Optimization Problem (LOP)."""

    def __init__(self, embed_dim, linear_bias=False):
        super(LOPDynamicEmbedding, self).__init__()
        self.num_types = 8
        self.projection = nn.Linear(36, 3 * embed_dim, bias=linear_bias)

    def forward(self, td):
        current_plan = td["current_plan"]
        areas = td["areas"]
        unassigned = current_plan < 0
        safe_plan = current_plan.clamp(min=0)
        current_plan_onehot = F.one_hot(safe_plan, num_classes=self.num_types).float()
        current_plan_onehot = current_plan_onehot * (~unassigned).unsqueeze(-1).float()

        objective_weights = prepare_lop_objective_weights(td, areas)
        objective_weights = objective_weights.unsqueeze(-2).expand(
            *current_plan.shape, objective_weights.size(-1)
        )
        node_idx = torch.arange(current_plan.size(-1), device=areas.device).view(1, -1)
        selected_parcel = node_idx.eq(td["current_node"].unsqueeze(-1))
        selected_parcel = selected_parcel & td["i"].squeeze(-1).gt(0).unsqueeze(-1)
        pending_type = td.get("pending_type_action", None)
        if pending_type is None:
            pending_type_onehot = areas.new_zeros((*current_plan.shape, self.num_types))
        else:
            pending_type = pending_type.long().clamp(0, self.num_types - 1)
            pending_type_onehot = F.one_hot(
                pending_type, num_classes=self.num_types
            ).to(dtype=areas.dtype)
            pending_type_onehot = pending_type_onehot.unsqueeze(-2).expand(
                *current_plan.shape, self.num_types
            )

        target_ratios = td.get("target_ratios", None)
        if target_ratios is None:
            target_ratios = areas.new_zeros((*td.batch_size, self.num_types))
        target_ratios = target_ratios.to(device=areas.device, dtype=areas.dtype)
        area_by_type = (current_plan_onehot * areas.unsqueeze(-1)).sum(dim=-2)
        total_area = areas.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        area_ratio_by_type = area_by_type / total_area
        remaining_deficit = (target_ratios - area_ratio_by_type).clamp_min(0)
        remaining_deficit = remaining_deficit.unsqueeze(-2).expand(
            *current_plan.shape, self.num_types
        )
        constraint_pressure = td.get("constraint_pressure", None)
        if constraint_pressure is None:
            constraint_pressure = areas.new_zeros((*td.batch_size, self.num_types))
        constraint_pressure = constraint_pressure.to(
            device=areas.device, dtype=areas.dtype
        )
        constraint_pressure = constraint_pressure / total_area
        constraint_pressure = constraint_pressure.unsqueeze(-2).expand(
            *current_plan.shape, self.num_types
        )

        dynamic_features = torch.cat(
            [
                current_plan_onehot,
                unassigned.unsqueeze(-1).float(),
                objective_weights,
                selected_parcel.unsqueeze(-1).to(dtype=areas.dtype),
                pending_type_onehot,
                remaining_deficit,
                constraint_pressure,
            ],
            dim=-1,
        )
        return self.projection(dynamic_features).chunk(3, dim=-1)


class SDVRPDynamicEmbedding(nn.Module):
    """Dynamic embedding for the Split Delivery Vehicle Routing Problem (SDVRP).
    Embed the following node features to the embedding space:
        - demand_with_depot: demand of the customers and the depot
    The demand with depot is used to modify the query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    """

    def __init__(self, embed_dim, linear_bias=False):
        super().__init__()
        self.projection = nn.Linear(1, 3 * embed_dim, bias=linear_bias)

    def forward(self, td):
        demands_with_depot = td["demand_with_depot"][..., None].clone()
        demands_with_depot[..., 0, :] = 0
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            demands_with_depot
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class JSSPDynamicEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=False, scaling_factor: int = 1000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.project_node_step = nn.Linear(2, 3 * embed_dim, bias=linear_bias)
        self.project_edge_step = nn.Linear(1, 3, bias=linear_bias)
        self.scaling_factor = scaling_factor

    def forward(self, td, cache):
        ma_emb = cache.node_embeddings["machine_embeddings"]
        bs, _, emb_dim = ma_emb.shape
        num_jobs = td["next_op"].size(1)
        # updates
        updates = ma_emb.new_zeros((bs, num_jobs, 3 * emb_dim))

        lbs = torch.clip(td["lbs"] - td["time"][:, None], 0) / self.scaling_factor
        update_feat = torch.stack((lbs, td["is_ready"]), dim=-1)
        job_update_feat = gather_by_index(update_feat, td["next_op"], dim=1)
        updates = updates + self.project_node_step(job_update_feat)

        ma_busy = td["busy_until"] > td["time"][:, None]
        # mask machines currently busy
        masked_proc_times = td["proc_times"].clone() / self.scaling_factor
        # bs, ma, ops
        masked_proc_times[ma_busy] = 0.0
        # bs, ops, ma, 3
        edge_feat = self.project_edge_step(masked_proc_times.unsqueeze(-1)).transpose(1, 2)
        job_edge_feat = gather_by_index(edge_feat, td["next_op"], dim=1)
        # bs, nodes, 3*emb
        edge_upd = torch.einsum("ijkl,ikm->ijlm", job_edge_feat, ma_emb).view(
            bs, num_jobs, 3 * emb_dim
        )
        updates = updates + edge_upd

        # (bs, nodes, emb)
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = updates.chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
