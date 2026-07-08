import random

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict.tensordict import TensorDict
try:
    from torchrl.data import Bounded, Composite, Unbounded
except ImportError:  # torchrl < 0.6 compatibility
    from torchrl.data import (  # type: ignore
        BoundedTensorSpec as Bounded,
        CompositeSpec as Composite,
        UnboundedContinuousTensorSpec as Unbounded,
    )

BoundedTensorSpec = Bounded
CompositeSpec = Composite
UnboundedContinuousTensorSpec = Unbounded
UnboundedDiscreteTensorSpec = Unbounded

from tensordict.tensordict import TensorDict
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.urbanplan.cityplan import init
from rl4co.utils.pylogger import get_pylogger
from typing import Optional
import numpy as np
import torch
from rl4co.envs.urbanplan.cityplan.generator import landuseOptGenerator
from rl4co.envs.urbanplan.cityplan.render import render
from rl4co.utils.multi_objective import normalize_weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = get_pylogger(__name__)


landtype = ['Commercial', 'Residential', 'Office', 'Residential&Commercial', 'Green Space', 'Education', 'Hospital',
            'SOHO']
class landuseOptEnv(RL4COEnvBase):
    """Landuse Optimization Problem as done in the CityPlan paper:
    The environment is a grid with locations containing different land types.
    The goal is to optimize the land use distribution to maximize the utility.
    The land use distribution is constrained by the land type of each location.

    Observations:
        - locations of the nodes
        - landuse types of the nodes
        - current land use type
        - remaining land use

    Constraints:
        - landuse cannot less than the limit area for each land type
        - fixed nodes cannot be changed

    Finish Condition:
    //TODO
        - the land use distribution is optimized

    reward:
        - the accessbility and compatibility mix of the land use distribution

    Args:
        generator: landuseOptGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = 'lop'

    def __init__(
        self,
        generator: landuseOptGenerator = None,
        generator_params: dict = {},
        tios: Optional[list] = None,
        min_type_ratios: Optional[list] = None,
        objective_weights: Optional[list] = None,
        eval_objective_weights: Optional[list] = None,
        objective_scalarization: str = "linear",
        objective_ideal: Optional[list] = None,
        sample_objective_weights: bool = False,
        sample_eval_objective_weights: bool = False,
        objective_weight_min: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("allow_done_after_reset", True)
        super().__init__(**kwargs)
        if generator is None:
            generator = landuseOptGenerator(**generator_params)
        self.generator = generator
        self.generator_params = generator_params
        self.num_types = 8
        self.tios = tios
        self.min_type_ratios = min_type_ratios
        self.objective_weights = objective_weights
        self.eval_objective_weights = eval_objective_weights
        self.objective_scalarization = objective_scalarization
        self.objective_ideal = objective_ideal
        self.sample_objective_weights = sample_objective_weights
        self.sample_eval_objective_weights = sample_eval_objective_weights
        if objective_weight_min < 0 or objective_weight_min >= 0.5:
            raise ValueError("objective_weight_min must be in [0, 0.5)")
        self.objective_weight_min = objective_weight_min
        self.max_steps = 0
        self.CompatibilityTable = [[8.8, 7.3, 7.9, 7.5, 7.4, 4.1, 4.2, 7.4],
                                  [7.3, 8.7, 6.0, 7.1, 8.4, 7.8, 6.2, 7.0],
                                  [7.9, 6.8, 8.6, 6.3, 7.1, 5.6, 5.0, 7.1],
                                  [7.5, 7.1, 6.3, 8.6, 7.3, 4.7, 4.7, 7.4],
                                  [7.4, 8.4, 7.1, 7.3, 8.2, 7.7, 6.9, 7.1],
                                  [4.1, 7.8, 5.6, 4.7, 7.7, 8.4, 5.3, 5.0],
                                  [4.2, 6.2, 5.0, 4.7, 6.9, 5.3, 8.1, 4.7],
                                  [7.4, 7.0, 7.1, 7.4, 7.1, 5.0, 4.7, 8.6]]
        self._make_spec(self.generator)

    def _set_seed(self, seed: Optional[int]):
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)

    def reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        if td is not None and not td.is_empty():
            td = self._exclude_transient_action_keys(td)
        return super().reset(td, batch_size=batch_size)

    def dataset(self, batch_size=[], phase="train", filename=None):
        dataset = super().dataset(batch_size=batch_size, phase=phase, filename=filename)
        if not self._dataset_has_key(dataset, "objective_weights"):
            weights = self._dataset_objective_weights(
                torch.Size([len(dataset)]), phase=phase, device="cpu"
            )
            if weights is None:
                return dataset
            dataset = dataset.add_key("objective_weights", weights.cpu())
        return dataset

    def _dataset_objective_weights(self, batch_size, phase, device):
        if phase == "train" or self.sample_eval_objective_weights:
            if self.sample_objective_weights:
                return self._default_objective_weights(batch_size, device).detach()
            return None

        if self.eval_objective_weights is not None:
            weights = torch.tensor(
                self.eval_objective_weights, dtype=torch.float32, device=device
            )
        elif self.objective_weights is not None:
            weights = torch.tensor(
                self.objective_weights, dtype=torch.float32, device=device
            )
        else:
            weights = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
        return self._prepare_objective_weights(weights, batch_size, device).detach()

    @staticmethod
    def _exclude_transient_action_keys(td):
        return td.exclude(
            "action",
            "type_action",
            "parcel_action",
            "pending_type_action",
            "reward_components",
            inplace=False,
        )

    @staticmethod
    def _dataset_has_key(dataset, key):
        if len(dataset) == 0:
            return False
        sample = dataset[0]
        return key in sample.keys() if isinstance(sample, TensorDict) else key in sample

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        td = self._exclude_transient_action_keys(td)
        # Initialize locations
        device = td.device
        init_locs = td["locs"]
        areas = td["areas"]
        num_loc = init_locs.shape[-2]
        fixed_mask = td["fixed_mask"]

        self.num_types = 8
        self.max_steps = num_loc
        # Non-fixed parcels are intentionally unassigned; the policy now chooses
        # both land-use type and parcel at every step.
        current_plan = td["init_plan"].clone()
        current_plan = torch.where(
            fixed_mask,
            torch.full_like(current_plan, -1),
            current_plan,
        )

        current_types_onehot = torch.zeros(
            (*batch_size, self.num_types), dtype=torch.bool, device=device
        )

        # initialize the current node
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        selected_type = torch.zeros((*batch_size,), dtype=torch.int64, device=device)

        if 'adjacency_list' in td.keys() and 'distances' in td.keys():
            adjacency_list = td['adjacency_list']
            distances = td['distances']
        else:
            num_neighbors = min(4, max(num_loc - 1, 0))
            adjacency_list = torch.zeros(
                (*batch_size, num_loc, num_neighbors + 1),
                dtype=torch.int64,
                device=device,
            )
            distances = torch.cdist(init_locs, init_locs)  # Calculate pairwise distances for all batches
            # Sort distances and get the indices of the closest neighbors (excluding self)
            sorted_indices = distances.argsort(dim=-1)[
                ..., :, 1 : num_neighbors + 1
            ]
            # Set the first element of each row in adjacency_list to neighbor count
            adjacency_list[..., :, 0] = num_neighbors
            # Set the remaining elements to the indices of the closest neighbors
            if num_neighbors > 0:
                adjacency_list[..., :, 1:] = sorted_indices

        # Other variables
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        parcel_action_mask = current_plan < 0
        action_mask, type_action_mask, type_parcel_action_mask = self._build_action_masks(
            current_plan, areas, parcel_action_mask
        )
        done = ~parcel_action_mask.any(dim=-1, keepdim=True)
        self._assert_active_rows_have_valid_action(action_mask, done)
        (
            action_mask,
            parcel_action_mask,
            type_action_mask,
            type_parcel_action_mask,
        ) = self._ensure_done_rows_have_dummy_action(
            action_mask,
            parcel_action_mask,
            type_action_mask,
            type_parcel_action_mask,
            done,
        )
        reward = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)
        objective_weights = td.get("objective_weights", None)
        if objective_weights is None:
            objective_weights = self._default_objective_weights(batch_size, device)
        else:
            objective_weights = self._prepare_objective_weights(
                objective_weights,
                batch_size,
                device,
            )
        target_ratios = self._target_ratios(device).expand(*batch_size, -1)
        constraint_ratios = self._constraint_ratios(device).expand(*batch_size, -1)
        constraint_pressure = self._constraint_pressure_by_type(
            current_plan, areas
        )
        return TensorDict(
            {
                "locs": init_locs,
                "areas": areas,
                "init_plan": td["init_plan"],
                "fixed_mask": fixed_mask,
                "current_node": current_node,
                "current_plan": current_plan,
                "current_type": selected_type,
                "selected_type": selected_type,
                "current_types_onehot": current_types_onehot,
                "adjacency_list": adjacency_list,
                "distances": distances,
                "i": i,
                "parcel_action_mask": parcel_action_mask,
                "type_action_mask": type_action_mask,
                "type_parcel_action_mask": type_parcel_action_mask,
                "action_mask": action_mask,
                "objective_weights": objective_weights,
                "target_ratios": target_ratios,
                "constraint_ratios": constraint_ratios,
                "constraint_pressure": constraint_pressure,
                "reward": reward,
                "done": done,
            },
            batch_size=batch_size,
        )
    def _step(self, td: TensorDict) -> TensorDict:
        current_plan = td["current_plan"].clone()
        areas = td["areas"]
        num_loc = current_plan.shape[-1]
        action, selected_type, current_node = self._resolve_joint_action(td, num_loc)
        td.set("action", action)
        done = td["done"]
        active = ~done.squeeze(-1)

        if active.any():
            chosen_is_feasible = td["action_mask"].gather(
                -1, td["action"].unsqueeze(-1)
            ).squeeze(-1)
            invalid = active & ~chosen_is_feasible
            if invalid.any():
                rows = invalid.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                if not isinstance(rows, list):
                    rows = [rows]
                raise ValueError(
                    "infeasible LUOP joint action selected for active batch rows "
                    f"{rows}"
                )

        if active.any():
            stepped_plan = current_plan.scatter(
                -1,
                current_node.clamp(0, num_loc - 1).unsqueeze(-1),
                selected_type.clamp(0, self.num_types - 1).unsqueeze(-1),
            )
            current_plan = torch.where(active.unsqueeze(-1), stepped_plan, current_plan)

        current_types_onehot = self._type_one_hot_tensor(selected_type)
        parcel_action_mask = current_plan < 0
        action_mask, type_action_mask, type_parcel_action_mask = self._build_action_masks(
            current_plan, areas, parcel_action_mask
        )

        done = done | ~parcel_action_mask.any(dim=-1, keepdim=True)
        done = done | (td["i"] >= self.max_steps - 1)
        self._assert_active_rows_have_valid_action(action_mask, done)
        (
            action_mask,
            parcel_action_mask,
            type_action_mask,
            type_parcel_action_mask,
        ) = self._ensure_done_rows_have_dummy_action(
            action_mask,
            parcel_action_mask,
            type_action_mask,
            type_parcel_action_mask,
            done,
        )
        constraint_pressure = self._constraint_pressure_by_type(
            current_plan, areas
        )

        td.update(
            {
                "current_node": current_node,
                "current_plan" : current_plan,
                "current_type": selected_type,
                "selected_type": selected_type,
                "type_action": selected_type,
                "parcel_action": current_node,
                "current_types_onehot": current_types_onehot,
                "i": td["i"] + 1,
                "parcel_action_mask": parcel_action_mask,
                "type_action_mask": type_action_mask,
                "type_parcel_action_mask": type_parcel_action_mask,
                "action_mask": action_mask,
                "constraint_pressure": constraint_pressure,
                "done": done,
            }
        )
        return td
    # def _step(self, td: TensorDict) -> TensorDict:
    #     current_node = td["action"]
    #     current_plan = td["current_plan"]
    #     areas = td["areas"]
    #     # update the current plan
    #     batch_size = current_plan.shape[0]
    #     current_type = td["current_type"]
    #     done = td["done"]
    #     # 更新 current_plan
    #     device = current_plan.device  # 获取当前设备
    #     batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)  # 批次索引
    #     valid_indices = current_type != -1
    #     valid_indices = valid_indices.to(device)
    #     # 只更新current_type不为-1的情况
    #     current_plan[batch_indices[valid_indices], current_node[valid_indices]] = current_type[valid_indices]
    #     # current_plan[batch_indices, current_node] = current_type
    #     #set the current type to the next type
    #     # current_types_onehot = torch.zeros((batch_size, self.num_types))
    #     # for i in range(batch_size):
    #     #     next_type = self.calc_next_type(current_plan[i], landtype, areas[i], current_type[i].item())
    #     #     if next_type is None or done[i]:
    #     #         done[i] = True
    #     #         continue
    #     #     current_type[i] = torch.tensor(next_type, dtype=torch.int64)
    #     #     current_types_onehot[i] = self._type_one_hot(next_type)
    #     next_types = self.calc_next_type_tensor(current_plan, areas, current_type)
    #     done |= next_types == None
    #     valid_indices = ~done
    #     valid_indices = valid_indices.squeeze()
    #     current_type[valid_indices] = next_types[valid_indices]
    #     current_types_onehot = self._type_one_hot_tensor(next_types)
    #     # Set available to 0 (i.e., already placed) if the current node is the first node
    #     available = td["action_mask"].scatter(
    #         -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
    #     )
    #
    #     # Set done if i is greater than max_steps
    #     # done = td["i"] >= self.max_steps - 1
    #     done = done | (td["i"] >= self.max_steps - 1)
    #
    #     # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
    #     reward = torch.zeros_like(done)
    #
    #     td.update(
    #         {
    #             "current_node": current_node,
    #             "current_plan" : current_plan,
    #             "current_type": current_type,
    #             "current_types_onehot": current_types_onehot,
    #             "i": td["i"] + 1,
    #             "action_mask": available,
    #             "reward": reward,
    #             "done": done,
    #         }
    #     )
    #     return td
    def _get_reward(self, td, actions):
        """
        We call the reward function with the final sequence of actions to get the reward
        reward = accessibility + compatibility
        """
        current_plan = td["current_plan"]
        distances = td["distances"]
        neighbors = td["adjacency_list"]
        if self._is_adjacency_matrix_tensor(neighbors):
            neighbors = self.adj_matrix_to_list(neighbors)
        else:
            neighbors = neighbors

        MinMaxCompatibility = [6.0, 8.8]
        MinMaxAccessibility = [-3.0, -1.5]
        if isinstance(neighbors, list):
            CompatibilityTable = torch.tensor(self.CompatibilityTable, dtype=torch.float32, device=current_plan.device)
            accessibility = self.calc_accessibility_tensor(current_plan, distances)
            compatibility = self.calc_compatibility_tensor_real(current_plan, neighbors, CompatibilityTable)
            norcompatibility = (compatibility - MinMaxCompatibility[0]) / (
                    MinMaxCompatibility[1] - MinMaxCompatibility[0])
            # 可达性归一化
            noraccessibility = (accessibility - MinMaxAccessibility[0]) / (
                    MinMaxAccessibility[1] - MinMaxAccessibility[0])
            components = torch.stack([norcompatibility, noraccessibility], dim=-1)
            # for i in range(batch_size):
            #     compatibility = self.calc_compatibility(current_plan[i].tolist(), landtype, neighbors[i], self.CompatibilityTable)
            #     # accessibility = self.calc_accessibility(current_plan[i].tolist(), landtype, distances[i].tolist())
            #     norcompatibility = (compatibility - MinMaxCompatibility[0]) / (
            #                 MinMaxCompatibility[1] - MinMaxCompatibility[0])
            #     noraccessibility = (accessibility[i] - MinMaxAccessibility[0]) / (
            #                 MinMaxAccessibility[1] - MinMaxAccessibility[0])
            #     # if self.isPlanValid(current_plan[i], areas[i]):
            #     if True:
            #         rewards[i] = norcompatibility + noraccessibility
            #     else:
            #         rewards[i] = norcompatibility + noraccessibility
        else:
            #neighbors = torch.tensor(neighbors, dtype=torch.long, device=device)  # Shape: [batch_size, num_centers, num_neighbours + 1]
            CompatibilityTable = torch.tensor(self.CompatibilityTable, dtype=torch.float32, device=current_plan.device)  # Shape: [num_landuse_types, num_landuse_types]
            # We do the operation in a batch
            # Ensure to calculate reward for each batch
            compatibility = self.calc_compatibility_tensor(current_plan, neighbors, CompatibilityTable)
            accessibility = self.calc_accessibility_tensor(current_plan, distances)
            # 兼容性归一化
            norcompatibility = (compatibility - MinMaxCompatibility[0]) / (
                    MinMaxCompatibility[1] - MinMaxCompatibility[0])
            # 可达性归一化
            noraccessibility = (accessibility - MinMaxAccessibility[0]) / (
                    MinMaxAccessibility[1] - MinMaxAccessibility[0])
            components = torch.stack([norcompatibility, noraccessibility], dim=-1)
        # CompatibilityTable = torch.tensor(self.CompatibilityTable, dtype=torch.float32, device=device)  # Shape: [num_landuse_types, num_landuse_types]
        # # We do the operation in a batch
        # # Ensure to calculate reward for each batch
        # compatibility = self.calc_compatibility_tensor(current_plan, neighbors, CompatibilityTable)
        # accessibility = self.calc_accessibility_tensor(current_plan, distances)
        # # 兼容性归一化
        # norcompatibility = (compatibility - MinMaxCompatibility[0]) / (
        #         MinMaxCompatibility[1] - MinMaxCompatibility[0])
        # # 可达性归一化
        # noraccessibility = (accessibility - MinMaxAccessibility[0]) / (
        #         MinMaxAccessibility[1] - MinMaxAccessibility[0])
        # for i in range(batch_size):
        #     compatibility = self.calc_compatibility(current_plan[i].tolist(), landtype, neighbors[i], self.CompatibilityTable)
        #     accessibility = self.calc_accessibility(current_plan[i].tolist(), landtype, distances[i].tolist())
        #     norcompatibility = (compatibility - MinMaxCompatibility[0]) / (
        #                 MinMaxCompatibility[1] - MinMaxCompatibility[0])
        #     noraccessibility = (accessibility - MinMaxAccessibility[0]) / (
        #                 MinMaxAccessibility[1] - MinMaxAccessibility[0])
        #     # if self.isPlanValid(current_plan[i], areas[i]):
        #     if True:
        #         rewards[i] = norcompatibility + noraccessibility
        #     else:
        #         rewards[i] = norcompatibility + noraccessibility
        #     print(rewards[i], compatibility, accessibility)
        #print(init.map_to_strings(current_plan[0].tolist(), landtype))
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
        rewards = self._scalarize_objectives(components, weights)
        td.set("reward_components", components)
        td.set("objective_weights", weights)
        return rewards
    # def getlastplan(self, actions):
    #
    #     for action in actions:
    #         if action is not None and select_type is not None:
    #             latest_plan[action] = select_type.item()
    #             select_type = calc_next_type(latest_plan, landtype, areaslist)
    #         else:
    #             break
    def isPlanValid(self, current_plan, areas):
        strstate = init.map_to_strings(current_plan.tolist(), landtype)
        #area_ratios = init.landuse_ratio_tensor(current_plan, areas, current_type)
        Residential_ratio = init.landuse_ratio(strstate, areas, 'Residential')
        Commercial_ratio = init.landuse_ratio(strstate, areas, 'Commercial')
        Education_ratio = init.landuse_ratio(strstate, areas, 'Education')
        Office_ratio = init.landuse_ratio(strstate, areas, 'Office')
        SOHO_ratio = init.landuse_ratio(strstate, areas, 'SOHO')
        RC_ratio = init.landuse_ratio(strstate, areas, 'Residential&Commercial')
        # 对每个批次进行计算
        if Education_ratio < 0.02:
            print("Education")
            return False
        if Residential_ratio < 0.2:
            print("Residential")
            return False
        if Commercial_ratio + RC_ratio < 0.1:
            print("RC_ratio")
            return False
        if Commercial_ratio < 0.05:
            print("Commercial")
            return False
        if Office_ratio < 0.05:
            print("Office")
            return False
        if Office_ratio + SOHO_ratio < 0.15:
            return False
        # if Residential_ratio + SOHO_ratio + RC_ratio < 0.5:
        #     return False
        return True

    def adj_matrix_to_list(self, adj_matrix):
        # 初始化邻接表，使用列表的列表存储每个批次的邻接列表
        batch_shape = adj_matrix.shape[:-2]
        if len(batch_shape) != 1:
            adj_matrix = adj_matrix.reshape(-1, *adj_matrix.shape[-2:])
        adj_list = []
        # 遍历邻接矩阵的每个批次
        for i in range(adj_matrix.shape[0]):
            # 初始化当前批次的邻接列表
            batch_adj_list = []
            # 遍历当前批次的每个节点
            for j in range(adj_matrix.shape[1]):
                # 找出与当前节点j相连的所有节点索引
                neighbors = []
                for k, is_connected in enumerate(adj_matrix[i, j]):
                    if is_connected:
                        neighbors.append(k)
                # 在邻居列表的[0]位置插入邻居节点的数量
                neighbors.insert(0, len(neighbors))
                # 将当前节点的邻接列表添加到批次的邻接列表中
                batch_adj_list.append(neighbors)
            # 将当前批次的邻接列表添加到总的邻接表中
            adj_list.append(batch_adj_list)
        return adj_list

    @staticmethod
    def _is_adjacency_matrix_tensor(neighbors):
        if neighbors.shape[-2] != neighbors.shape[-1]:
            return False
        if neighbors.dtype == torch.bool:
            return True
        if not torch.is_floating_point(neighbors) and not torch.is_complex(neighbors):
            binary_values = ((neighbors == 0) | (neighbors == 1)).all()
            if not binary_values:
                return False
            diagonal = neighbors.diagonal(dim1=-2, dim2=-1)
            return diagonal.eq(0).all() or diagonal.eq(1).all()
        return False
    def _make_spec(self, generator: landuseOptGenerator):
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            areas=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(generator.num_loc,),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            init_plan=BoundedTensorSpec(
                low=0,
                high=7,
                shape=(generator.num_loc,),
                dtype=torch.int64,
            ),
            current_plan=BoundedTensorSpec(
                low=-1,
                high=7,
                shape=(generator.num_loc,),
                dtype=torch.int64,
            ),
            current_type=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            selected_type=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_types_onehot=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(8,),
                dtype=torch.bool,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            parcel_action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc,),
                dtype=torch.bool,
            ),
            type_action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_types,),
                dtype=torch.bool,
            ),
            type_parcel_action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_types, generator.num_loc),
                dtype=torch.bool,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_types * generator.num_loc,),
                dtype=torch.bool,
            ),
            type_action=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            parcel_action=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            objective_weights=UnboundedContinuousTensorSpec(shape=(2,)),
            reward_components=UnboundedContinuousTensorSpec(shape=(2,)),
            target_ratios=UnboundedContinuousTensorSpec(shape=(self.num_types,)),
            constraint_ratios=UnboundedContinuousTensorSpec(shape=(self.num_types,)),
            constraint_pressure=UnboundedContinuousTensorSpec(shape=(self.num_types,)),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.num_types * generator.num_loc - 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)
    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        current_plan = td["current_plan"]
        areas = td["areas"]

        if not (current_plan >= 0).all():
            raise ValueError("All parcels must be assigned before reward")
        if not (current_plan < self.num_types).all():
            raise ValueError("All parcels must use a known land-use type")

        if "init_plan" in td.keys():
            fixed_parcels = ~td["fixed_mask"].bool()
            fixed_unchanged = current_plan[fixed_parcels] == td["init_plan"][
                fixed_parcels
            ]
            if not fixed_unchanged.all():
                raise ValueError("fixed parcels must keep their initial type")

        area_by_type = self._area_by_type(current_plan, areas)
        total_area = areas.sum(dim=-1)
        if self._uses_default_group_constraints():
            group_deficit = self._constraint_group_deficits(area_by_type, total_area)
            if not (group_deficit <= 1e-6).all():
                raise ValueError("Plan violates grouped minimum land-use ratios")
        else:
            required = td.get("constraint_ratios", None)
            if required is None:
                required = self._constraint_ratios(current_plan.device).expand(
                    *current_plan.shape[:-1], -1
                )
            ratios = area_by_type / total_area.unsqueeze(-1).clamp_min(1e-8)
            if not (ratios + 1e-6 >= required).all():
                raise ValueError("Plan violates minimum land-use ratios")
    def calc_compatibility(self, plan, landtypes, neighbourlist, CompatibilityTable):
        # 计算兼容性
        strstate = init.map_to_strings(plan, landtypes)
        res = init.calCompatibility(landtypes, neighbourlist, strstate, CompatibilityTable)
        return res
    def calc_compatibility_tensor(self, landuselists, neighbourlists, CompatibilityTable):
        '''Calculate compatibility for batches of land use lists and neighbour lists using PyTorch tensors.
        Each batch element contains multiple centers with their respective neighbours.
        '''
        # Remove the neighbour count column and get neighbour indices
        neighbour_indices = neighbourlists[..., :, 1:]
        num_neighbors = neighbour_indices.size(-1)
        neighbor_counts = neighbourlists[..., :, 0].clamp(min=0, max=num_neighbors)

        source_landuses = landuselists.unsqueeze(-2).expand(
            *landuselists.shape[:-1],
            neighbourlists.shape[-2],
            landuselists.shape[-1],
        )
        neighbour_landuses = source_landuses.gather(-1, neighbour_indices)

        # Get the compatibility scores using advanced indexing
        # CompatibilityTable should be indexed with the current land use and its neighbours' land use types
        compatibility_scores = CompatibilityTable[
            landuselists.unsqueeze(-1), neighbour_landuses
        ]
        valid_neighbors = (
            torch.arange(num_neighbors, device=landuselists.device)
            < neighbor_counts.unsqueeze(-1)
        )
        compatibility_scores = compatibility_scores * valid_neighbors.float()

        # Sum compatibility scores across neighbours for each center
        compatibility_sums = compatibility_scores.sum(dim=-1)
        neighbor_denominator = neighbor_counts.clamp_min(1).to(compatibility_scores.dtype)
        average_compatibility = compatibility_sums / neighbor_denominator

        # Compute average compatibility per batch element
        total_compatibility = average_compatibility.mean(dim=-1)

        return total_compatibility

    def calc_accessibility(self, plan, landtypes, distancelist):
        strstate = init.map_to_strings(plan, landtypes)
        # 计算可及性
        Accessibility = (init.calInterAccessibility_Average(strstate,
                                                           'Residential' or 'SOHO' or 'Residential&Commercial',
                                                           'Commercial' or 'Residential&Commercial', distancelist)
                         + init.calInterAccessibility_Average(strstate,
                                                              'Residential' or 'SOHO' or 'Residential&Commercial',
                                                              'Education', distancelist)
                         + init.calInterAccessibility_Average(strstate,
                                                            'Residential' or 'SOHO' or 'Residential&Commercial',
                                                            'Hospital', distancelist)
                         + init.calInterAccessibility_Average(strstate,
                                                            'Residential' or 'SOHO' or 'Residential&Commercial',
                                                            'Office' or 'SOHO', distancelist)
                         + init.calInterAccessibility_Average(strstate,
                                                            'Residential' or 'SOHO' or 'Residential&Commercial',
                                                            'Green Space', distancelist))
        return Accessibility
    def calc_accessibility_tensor(self, plan, distancelist):
        # 计算可及性
        resid = ['Residential', 'SOHO', 'Residential&Commercial']
        Accessibility = (init.calInterAccessibility_Average_tensor(plan, resid,
                                                           ['Commercial', 'Residential&Commercial'], distancelist)
                         + init.calInterAccessibility_Average_tensor(plan, resid,
                                                              ['Education'], distancelist)
                         + init.calInterAccessibility_Average_tensor(plan, resid,
                                                            ['Hospital'], distancelist)
                         + init.calInterAccessibility_Average_tensor(plan, resid,
                                                            ['Office', 'SOHO'], distancelist)
                         + init.calInterAccessibility_Average_tensor(plan, resid,
                                                            ['Green Space'], distancelist))
        return Accessibility
    def calc_next_type(self, current_plan, landtypes, areas, current_type):
        strstate = init.map_to_strings(current_plan.tolist(), landtypes)
        #tios = [0.051420568227290915, 0.20078031212484992, 0.05712284913965586, 0.42276910764305725,0.046318527410964386,0.03221288515406162,0.06562625050020007,0.12374949979991996]
        tios = [0.11556944, 0.2579062, 0.07018196, 0.10833395, 0.10989285, 0.02113334, 0.1810, 0.13518225]
        if init.landuse_ratio(strstate, areas, landtypes[current_type]) < tios[current_type]:
            return current_type
        else:
            if current_type == 7:
                return None
            return current_type + 1
    def calc_compatibility_tensor_real(self, landuselists, neighbourlists, CompatibilityTable):
        '''
        Calculate compatibility for batches of land use lists and neighbour lists using PyTorch tensors.
        Each batch element contains multiple centers with their respective neighbours.

        Parameters:
        - landuselists: Tensor of shape [batch_size, num_centers]
        - neighbourlists: List of length batch_size, where each element is a list of length num_centers,
                          and each center's neighbors are a list starting with the neighbor count followed by neighbor indices.
                          Example:
                          [
                              [ [2, 1, 2], [3, 0, 2, 1], [1, 1] ],
                              [ [1, 0], [2, 2, 3], [4, 0, 1, 3, 4] ]
                          ]
        - CompatibilityTable: Tensor or list of shape [num_land_use_types, num_land_use_types]

        Returns:
        - total_compatibility: Tensor of shape [batch_size]
        '''
        # Step 0: Ensure CompatibilityTable is a PyTorch tensor
        if not isinstance(CompatibilityTable, torch.Tensor):
            CompatibilityTable = torch.tensor(CompatibilityTable, dtype=torch.float32, device=landuselists.device)
        else:
            CompatibilityTable = CompatibilityTable.to(device=landuselists.device, dtype=torch.float32)

        batch_shape = landuselists.shape[:-1]
        num_centers = landuselists.shape[-1]
        landuselists = landuselists.reshape(-1, num_centers)
        batch_size = landuselists.shape[0]

        # Step 1: Determine the maximum number of neighbors across all centers in all batches
        max_num_neighbors = max(
            (len(neighbors) - 1) for batch in neighbourlists for neighbors in batch
        )

        # Step 2: Initialize a tensor for neighbor indices with padding (using -1 as the padding index)
        # Shape: [batch_size, num_centers, max_num_neighbors]
        padded_neighbour_indices = torch.full(
            (batch_size, num_centers, max_num_neighbors),
            fill_value=-1,  # Assuming -1 is not a valid land use index
            dtype=torch.long,
            device=landuselists.device
        )

        # Create a mask to identify valid neighbors
        mask = torch.zeros(
            (batch_size, num_centers, max_num_neighbors),
            dtype=torch.bool,
            device=landuselists.device
        )

        # Step 3: Populate the padded_neighbour_indices and mask
        for b, batch in enumerate(neighbourlists):
            for c, neighbors in enumerate(batch):
                if len(neighbors) == 0:
                    raise ValueError(f"Batch {b}, Center {c} has an empty neighbor list.")
                num_valid = neighbors[0]  # The first element is the number of neighbors
                neighbor_indices = neighbors[1:]

                if num_valid != len(neighbor_indices):
                    raise ValueError(
                        f"Batch {b}, Center {c}: Declared {num_valid} neighbors but got {len(neighbor_indices)}.")

                if num_valid > max_num_neighbors:
                    raise ValueError(
                        f"Batch {b}, Center {c}: Number of neighbors ({num_valid}) exceeds max_num_neighbors ({max_num_neighbors}).")

                if num_valid > 0:
                    padded_neighbour_indices[b, c, :num_valid] = torch.tensor(
                        neighbor_indices, dtype=torch.long, device=landuselists.device
                    )
                    mask[b, c, :num_valid] = 1  # Mark as valid

        # Step 4: Gather neighbor land uses, handling padding
        # Replace -1 with 0 to prevent indexing errors; these will be masked out later
        safe_neighbour_indices = padded_neighbour_indices.clamp(min=0)
        # landuselists is [batch_size, num_centers]
        # safe_neighbour_indices.view(batch_size, num_centers * max_num_neighbors) is [batch_size, num_centers * max_num_neighbors]
        # gather along dim=1
        neighbour_landuses = landuselists.gather(1, safe_neighbour_indices.view(batch_size, -1))
        # Reshape to [batch_size, num_centers, max_num_neighbors]
        neighbour_landuses = neighbour_landuses.view(batch_size, num_centers, max_num_neighbors)

        # Step 5: Get the current land use types for each center
        # Shape: [batch_size, num_centers, 1]
        current_land_use = landuselists.unsqueeze(2)

        # Step 6: Lookup compatibility scores
        # CompatibilityTable shape: [num_land_use_types, num_land_use_types]
        # current_land_use: [batch_size, num_centers, 1]
        # neighbour_landuses: [batch_size, num_centers, max_num_neighbors]
        # Using advanced indexing to get [batch_size, num_centers, max_num_neighbors]
        compatibility_scores = CompatibilityTable[current_land_use, neighbour_landuses]

        # Step 7: Apply the mask to zero out compatibility scores from padded neighbors
        compatibility_scores = compatibility_scores * mask.float()

        # Step 8: Sum compatibility scores across neighbors for each center
        compatibility_sums = compatibility_scores.sum(dim=2)  # Shape: [batch_size, num_centers]

        # Step 9: Count the number of valid neighbors per center
        num_valid_neighbors = mask.sum(dim=2).clamp(min=1).float()  # Avoid division by zero

        # Step 10: Compute average compatibility per center
        average_compatibility = compatibility_sums / num_valid_neighbors  # Shape: [batch_size, num_centers]

        # Step 11: Compute average compatibility per batch element
        total_compatibility = average_compatibility.mean(dim=1).reshape(batch_shape)

        return total_compatibility
    def calc_next_type_tensor(self, current_plan, areas, current_type):
        #tios = [0.051420568227290915, 0.20078031212484992, 0.05712284913965586, 0.42276910764305725,0.046318527410964386,0.03221288515406162,0.06562625050020007,0.12374949979991996]
        #tios = [0.34765893, 0.22549535, 0.10341608, 0.13914989, 0.01110495, 0.0232858, 0.01065741, 0.13923156]
        #tios = [0.39447598232042846, 0.2445364166666667, 0.0532621387334182, 0.15519095666666669, 0.02144777529193919, 0.020156663654213657, 0.01065741, 0.1002726266666666]
        #TODO
        #tios = [0.11556944, 0.2579062, 0.07018196, 0.10833395, 0.10989285, 0.02113334, 0.1810, 0.13518225]
        if self.tios is not None:
            tios = torch.tensor(self.tios).to(current_plan.device)
        else:
            #tioslist = [0.39447598232042846, 0.2445364166666667, 0.0532621387334182, 0.15519095666666669, 0.02144777529193919, 0.020156663654213657, 0.01065741, 0.1002726266666666]
            #tioslist = [0.11556944, 0.2579062, 0.07018196, 0.10833395, 0.10989285, 0.02113334, 0.1810, 0.13518225]
            tioslist = [0.13593291, 0.22783339,0.051964402, 0.11997096,0.11003187,0.020120958, 0.18186955,0.15227593]
            tios = torch.tensor(tioslist).to(current_plan.device)
        #tios = torch.tensor(tios).to(current_plan.device)
        area_ratios = init.landuse_ratio_tensor(current_plan, areas, current_type)
        area_exceeds_mask = area_ratios > tios[current_type]
        type_is_7_mask = current_type == 7
        next_types = current_type.where(~area_exceeds_mask, current_type + 1)

        # 对于面积比例超过阈值且类型为7的情况，将 next_types 设置为一个特定的值，例如 -1
        next_types[area_exceeds_mask & type_is_7_mask] = -1
        next_types = torch.where(next_types == 6, torch.tensor(7, device=device), next_types)
        #next_types = torch.where(next_types == 1, torch.tensor(2, device=device), next_types)

        return next_types
    def find_type_index(self,search_type):
        landtype_array = np.array(landtype)
        index = np.where(landtype_array == search_type)[0]
        return index

    @staticmethod
    def encode_action(selected_type, parcel, num_loc):
        return selected_type * num_loc + parcel

    @staticmethod
    def decode_action(action, num_loc):
        return action // num_loc, action % num_loc

    def _resolve_joint_action(self, td, num_loc):
        has_flat = "action" in td.keys()
        has_explicit = "type_action" in td.keys() and "parcel_action" in td.keys()

        if has_flat:
            action = td["action"].long()
            self._validate_flat_action_range(action, num_loc)
            selected_type, current_node = self.decode_action(action, num_loc)
            if has_explicit:
                explicit_type = td["type_action"].long()
                explicit_node = td["parcel_action"].long()
                self._validate_component_action_ranges(
                    explicit_type,
                    explicit_node,
                    num_loc,
                )
                explicit_action = self.encode_action(explicit_type, explicit_node, num_loc)
                stale_explicit = torch.zeros_like(action, dtype=torch.bool)
                if (
                    "selected_type" in td.keys()
                    and "current_node" in td.keys()
                    and "i" in td.keys()
                ):
                    stale_explicit = (
                        (td["i"].squeeze(-1) > 0)
                        & (explicit_type == td["selected_type"].long())
                        & (explicit_node == td["current_node"].long())
                    )
                valid_explicit = stale_explicit | (action == explicit_action)
                if not valid_explicit.all():
                    mismatch = ~valid_explicit
                    rows = mismatch.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
                    raise ValueError(
                        "mismatched flat and explicit LUOP joint actions for "
                        f"batch rows {rows}"
                    )
            return action, selected_type, current_node

        if has_explicit:
            selected_type = td["type_action"].long()
            current_node = td["parcel_action"].long()
            self._validate_component_action_ranges(
                selected_type,
                current_node,
                num_loc,
            )
            explicit_action = self.encode_action(selected_type, current_node, num_loc)
            return explicit_action, selected_type, current_node

        raise KeyError("LUOP step requires action or type_action+parcel_action")

    def _validate_flat_action_range(self, action, num_loc):
        num_actions = self.num_types * num_loc
        invalid = (action < 0) | (action >= num_actions)
        if invalid.any():
            rows = invalid.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            raise ValueError(
                "LUOP flat actions must be in valid joint type-parcel range; "
                f"got range [0, {num_actions - 1}] violations at batch rows {rows}"
            )

    def _validate_component_action_ranges(
        self, selected_type, current_node, num_loc
    ):
        invalid = (
            (selected_type < 0)
            | (selected_type >= self.num_types)
            | (current_node < 0)
            | (current_node >= num_loc)
        )
        if invalid.any():
            rows = invalid.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            raise ValueError(
                "LUOP component actions must be in valid type and parcel ranges; "
                f"got type range [0, {self.num_types - 1}] and parcel range "
                f"[0, {num_loc - 1}] violations at batch rows {rows}"
            )

    def _target_ratios(self, device):
        if self.tios is not None:
            values = self.tios
        else:
            values = [
                0.13593291,
                0.22783339,
                0.051964402,
                0.11997096,
                0.11003187,
                0.020120958,
                0.18186955,
                0.15227593,
            ]
        return torch.tensor(values, dtype=torch.float32, device=device)

    def _constraint_ratios(self, device):
        if self.min_type_ratios is not None:
            values = self.min_type_ratios
        elif self.tios is not None:
            values = self.tios
        else:
            # Singleton lower bounds from the original grouped feasibility rules.
            values = [0.05, 0.20, 0.05, 0.0, 0.0, 0.02, 0.0, 0.0]
        return torch.tensor(values, dtype=torch.float32, device=device)

    def _uses_default_group_constraints(self):
        return self.min_type_ratios is None and self.tios is None

    def _constraint_group_matrix_and_ratios(self, device):
        if self._uses_default_group_constraints():
            matrix = torch.zeros((6, self.num_types), dtype=torch.float32, device=device)
            matrix[0, 5] = 1.0  # Education
            matrix[1, 1] = 1.0  # Residential
            matrix[2, [0, 3]] = 1.0  # Commercial + Residential&Commercial
            matrix[3, 0] = 1.0  # Commercial
            matrix[4, 2] = 1.0  # Office
            matrix[5, [2, 7]] = 1.0  # Office + SOHO
            ratios = torch.tensor(
                [0.02, 0.20, 0.10, 0.05, 0.05, 0.15],
                dtype=torch.float32,
                device=device,
            )
        else:
            matrix = torch.eye(self.num_types, dtype=torch.float32, device=device)
            ratios = self._constraint_ratios(device)
        return matrix, ratios

    def _default_objective_weights(self, batch_size, device):
        if self.sample_objective_weights:
            rng = getattr(self, "rng", None)
            sample_device = device if rng is None else "cpu"
            first_weight = torch.rand(
                batch_size,
                dtype=torch.float32,
                device=sample_device,
                generator=rng,
            )
            if self.objective_weight_min > 0:
                width = 1.0 - 2.0 * self.objective_weight_min
                first_weight = self.objective_weight_min + width * first_weight
            weights = torch.stack([first_weight, 1.0 - first_weight], dim=-1)
            weights = weights.to(device)
            return self._prepare_objective_weights(weights, batch_size, device)
        if self.objective_weights is None:
            weights = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
        else:
            weights = torch.tensor(self.objective_weights, dtype=torch.float32, device=device)
        return self._prepare_objective_weights(weights, batch_size, device)

    @staticmethod
    def _prepare_objective_weights(
        weights,
        batch_size,
        device,
        dtype=torch.float32,
    ):
        weights = torch.as_tensor(weights, device=device, dtype=dtype)
        weights = normalize_weights(weights, name="objective_weights")
        if weights.size(-1) != 2:
            raise ValueError("objective_weights must have two objective dimensions")
        if weights.dim() == 1:
            weights = weights.expand(*batch_size, -1)
        elif weights.shape[:-1] != torch.Size(batch_size):
            try:
                weights = weights.expand(*batch_size, -1)
            except RuntimeError as exc:
                raise ValueError(
                    "objective_weights batch shape must match the environment batch"
                ) from exc
        return weights

    def _objective_ideal_tensor(self, components):
        if self.objective_ideal is None:
            return components.new_ones(components.size(-1))
        ideal = torch.as_tensor(
            self.objective_ideal, device=components.device, dtype=components.dtype
        )
        if ideal.shape[-1] != components.shape[-1]:
            raise ValueError(
                "objective_ideal must have the same number of entries as reward_components"
            )
        return ideal

    def _scalarize_objectives(self, components, weights):
        if self.objective_scalarization == "linear":
            return (components * weights).sum(dim=-1)
        if self.objective_scalarization == "chebyshev":
            ideal = self._objective_ideal_tensor(components)
            shortfall = (ideal - components).clamp_min(0)
            return -(weights * shortfall).max(dim=-1).values
        raise ValueError(
            "objective_scalarization must be one of {'linear', 'chebyshev'}"
        )

    def _area_by_type(self, current_plan, areas):
        safe_plan = current_plan.clamp(min=0)
        assigned = current_plan >= 0
        one_hot = F.one_hot(safe_plan, num_classes=self.num_types).float()
        one_hot = one_hot * assigned.unsqueeze(-1).float()
        return (one_hot * areas.unsqueeze(-1)).sum(dim=-2)

    def _constraint_group_deficits(self, area_by_type, total_area):
        group_matrix, group_ratios = self._constraint_group_matrix_and_ratios(
            area_by_type.device
        )
        group_area = torch.matmul(area_by_type, group_matrix.t())
        required_area = total_area.unsqueeze(-1) * group_ratios
        while required_area.dim() < group_area.dim():
            required_area = required_area.unsqueeze(-2)
        return (required_area - group_area).clamp_min(0)

    def _constraint_bucket_deficits(self, group_deficits):
        if self._uses_default_group_constraints():
            education = group_deficits[..., 0]
            residential = group_deficits[..., 1]
            commercial_family = torch.maximum(
                group_deficits[..., 2], group_deficits[..., 3]
            )
            office_family = torch.maximum(
                group_deficits[..., 4], group_deficits[..., 5]
            )
            bucket_deficits = torch.stack(
                [education, residential, commercial_family, office_family],
                dim=-1,
            )
        else:
            bucket_deficits = group_deficits
        return bucket_deficits

    def _future_constraint_needs(self, group_deficits):
        bucket_deficits = self._constraint_bucket_deficits(group_deficits)
        required_area = bucket_deficits.sum(dim=-1)
        required_count = bucket_deficits.gt(1e-8).sum(dim=-1)
        return required_area, required_count

    def _max_remaining_area_after_candidate(self, areas, parcel_action_mask):
        remaining_areas = areas * parcel_action_mask.to(dtype=areas.dtype)
        k = min(2, remaining_areas.size(-1))
        top_values = remaining_areas.topk(k=k, dim=-1).values
        top1 = top_values[..., 0]
        if k == 1:
            top2 = torch.zeros_like(top1)
        else:
            top2 = top_values[..., 1]
        candidate_is_top1 = parcel_action_mask & areas.eq(top1.unsqueeze(-1))
        return torch.where(candidate_is_top1, top2.unsqueeze(-1), top1.unsqueeze(-1))

    def _top_remaining_cumsum_after_candidate(
        self, areas, parcel_action_mask, max_count=8
    ):
        num_loc = areas.size(-1)
        k_limit = min(max_count, max(num_loc - 1, 0))
        if k_limit == 0:
            return areas.new_zeros((*areas.shape[:-1], num_loc, 0))

        remaining_areas = areas * parcel_action_mask.to(dtype=areas.dtype)
        top_count = min(num_loc, k_limit + 1)
        top_values, top_indices = remaining_areas.topk(k=top_count, dim=-1)
        top_cumsum = top_values.cumsum(dim=-1)
        base_cumsum = top_cumsum[..., :k_limit]
        replacement_cumsum = top_cumsum[..., 1 : k_limit + 1]

        parcel_idx = torch.arange(num_loc, device=areas.device).view(
            *([1] * len(areas.shape[:-1])), num_loc, 1
        )
        candidate_in_top_prefix = (
            top_indices.unsqueeze(-2).eq(parcel_idx)[..., :k_limit]
            .cumsum(dim=-1)
            .bool()
        )
        candidate_area = areas.unsqueeze(-1)
        return torch.where(
            candidate_in_top_prefix,
            replacement_cumsum.unsqueeze(-2) - candidate_area,
            base_cumsum.unsqueeze(-2),
        )

    def _future_constraint_min_parcel_count(
        self, group_deficits, top_remaining_cumsum_after
    ):
        bucket_deficits = self._constraint_bucket_deficits(group_deficits)
        effective_deficits = (bucket_deficits - 1e-8).clamp_min(0)
        k_limit = top_remaining_cumsum_after.size(-1)
        if k_limit == 0:
            required_count = torch.where(
                effective_deficits.gt(0),
                torch.ones_like(effective_deficits),
                torch.zeros_like(effective_deficits),
            )
            return required_count.sum(dim=-1)

        available_by_count = top_remaining_cumsum_after.unsqueeze(-3).unsqueeze(-1)
        meets_deficit = available_by_count >= effective_deficits.unsqueeze(-2)
        has_enough_area = meets_deficit.any(dim=-2)
        first_count = meets_deficit.float().argmax(dim=-2) + 1
        topk_area = top_remaining_cumsum_after[..., -1].unsqueeze(-1)
        if k_limit == 1:
            kth_area = topk_area
        else:
            kth_area = (
                top_remaining_cumsum_after[..., -1]
                - top_remaining_cumsum_after[..., -2]
            ).unsqueeze(-1)
        extra_area = (effective_deficits - topk_area.unsqueeze(-3)).clamp_min(0)
        extra_count = torch.ceil(extra_area / kth_area.unsqueeze(-3).clamp_min(1e-8))
        required_count = torch.where(
            effective_deficits.gt(0),
            torch.where(
                has_enough_area,
                first_count.to(dtype=effective_deficits.dtype),
                k_limit + extra_count,
            ),
            torch.zeros_like(effective_deficits),
        )
        return required_count.sum(dim=-1)

    def _constraint_pressure_by_type(self, current_plan, areas):
        assigned_area = self._area_by_type(current_plan, areas)
        total_area = areas.sum(dim=-1)
        group_deficits = self._constraint_group_deficits(
            assigned_area, total_area
        )
        group_matrix, _ = self._constraint_group_matrix_and_ratios(
            current_plan.device
        )
        return torch.matmul(group_deficits, group_matrix)

    def _build_action_masks(self, current_plan, areas, parcel_action_mask=None):
        if parcel_action_mask is None:
            parcel_action_mask = current_plan < 0
        group_matrix, _ = self._constraint_group_matrix_and_ratios(current_plan.device)
        assigned_area = self._area_by_type(current_plan, areas)
        total_area = areas.sum(dim=-1)
        current_group_deficit = self._constraint_group_deficits(
            assigned_area, total_area
        )
        has_deficit = current_group_deficit.gt(1e-8).any(dim=-1)
        unassigned_area = (areas * parcel_action_mask.float()).sum(dim=-1)
        unassigned_count = parcel_action_mask.sum(dim=-1)

        batch_shape = current_plan.shape[:-1]
        num_loc = areas.size(-1)
        candidate_area = areas.unsqueeze(-2).unsqueeze(-1)
        remaining_after = (unassigned_area.unsqueeze(-1) - areas).clamp_min(0)
        remaining_count_after = (unassigned_count - 1).clamp_min(0)
        after_area = assigned_area.unsqueeze(-2).unsqueeze(-2).expand(
            *batch_shape, self.num_types, num_loc, self.num_types
        )
        type_eye = torch.eye(
            self.num_types,
            dtype=areas.dtype,
            device=current_plan.device,
        ).view(*([1] * len(batch_shape)), self.num_types, 1, self.num_types)
        after_area = after_area + candidate_area * type_eye

        after_group_deficit = self._constraint_group_deficits(after_area, total_area)
        area_deficit, positive_deficit_count = self._future_constraint_needs(
            after_group_deficit
        )
        top_remaining_cumsum_after = self._top_remaining_cumsum_after_candidate(
            areas, parcel_action_mask
        )
        min_required_parcel_count = self._future_constraint_min_parcel_count(
            after_group_deficit,
            top_remaining_cumsum_after,
        )
        can_still_meet_targets = (
            area_deficit <= (remaining_after.unsqueeze(-2) + 1e-8)
        ) & (
            positive_deficit_count
            <= remaining_count_after.unsqueeze(-1).unsqueeze(-1)
        ) & (
            min_required_parcel_count
            <= remaining_count_after.unsqueeze(-1).unsqueeze(-1)
        )
        reduces_active_deficit = (
            current_group_deficit.gt(1e-8).float() @ group_matrix
        ).gt(0)
        deficit_type_mask = torch.where(
            has_deficit.unsqueeze(-1),
            reduces_active_deficit,
            torch.ones_like(reduces_active_deficit, dtype=torch.bool),
        )
        joint_mask = (
            can_still_meet_targets
            & parcel_action_mask.unsqueeze(-2)
            & deficit_type_mask.unsqueeze(-1)
        )
        type_action_mask = joint_mask.any(dim=-1)
        return (
            joint_mask.reshape(*batch_shape, -1),
            type_action_mask,
            joint_mask,
        )

    def get_parcel_action_mask(self, td):
        type_parcel_mask = td.get("type_parcel_action_mask", None)
        if type_parcel_mask is None:
            type_parcel_mask = td["action_mask"].view(
                *td["action_mask"].shape[:-1], self.num_types, -1
            )
        if "pending_type_action" in td.keys():
            type_key = "pending_type_action"
        else:
            type_key = "type_action" if "type_action" in td.keys() else "selected_type"
        selected_type = td[type_key].long()
        invalid = (selected_type < 0) | (selected_type >= self.num_types)
        if invalid.any():
            rows = invalid.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            raise ValueError(
                "LUOP type-conditioned parcel mask requires type ids in "
                f"[0, {self.num_types - 1}]; invalid batch rows {rows}"
            )
        gather_index = selected_type.unsqueeze(-1).unsqueeze(-1).expand(
            *selected_type.shape, 1, type_parcel_mask.shape[-1]
        )
        return type_parcel_mask.gather(-2, gather_index).squeeze(-2)

    @staticmethod
    def _ensure_done_rows_have_dummy_action(
        action_mask, parcel_action_mask, type_action_mask, type_parcel_action_mask, done
    ):
        safe_mask = action_mask.clone()
        done_rows = done.squeeze(-1)
        if done_rows.any():
            parcel_action_mask = parcel_action_mask.clone()
            type_action_mask = type_action_mask.clone()
            type_parcel_action_mask = type_parcel_action_mask.clone()
            safe_mask = torch.where(
                done_rows.unsqueeze(-1),
                torch.zeros_like(safe_mask),
                safe_mask,
            )
            parcel_action_mask = torch.where(
                done_rows.unsqueeze(-1),
                torch.zeros_like(parcel_action_mask),
                parcel_action_mask,
            )
            type_action_mask = torch.where(
                done_rows.unsqueeze(-1),
                torch.zeros_like(type_action_mask),
                type_action_mask,
            )
            type_parcel_action_mask = torch.where(
                done_rows.unsqueeze(-1).unsqueeze(-1),
                torch.zeros_like(type_parcel_action_mask),
                type_parcel_action_mask,
            )
            dummy_action_mask = torch.zeros_like(safe_mask)
            dummy_action_mask[..., 0] = done_rows
            dummy_parcel_mask = torch.zeros_like(parcel_action_mask)
            dummy_parcel_mask[..., 0] = done_rows
            dummy_type_mask = torch.zeros_like(type_action_mask)
            dummy_type_mask[..., 0] = done_rows
            dummy_type_parcel_mask = torch.zeros_like(type_parcel_action_mask)
            dummy_type_parcel_mask[..., 0, 0] = done_rows
            safe_mask = safe_mask | dummy_action_mask
            parcel_action_mask = parcel_action_mask | dummy_parcel_mask
            type_action_mask = type_action_mask | dummy_type_mask
            type_parcel_action_mask = type_parcel_action_mask | dummy_type_parcel_mask
        return safe_mask, parcel_action_mask, type_action_mask, type_parcel_action_mask

    @staticmethod
    def _assert_active_rows_have_valid_action(action_mask, done):
        active = ~done.squeeze(-1)
        invalid = active & ~action_mask.any(dim=-1)
        if invalid.any():
            rows = torch.nonzero(invalid, as_tuple=False).flatten().detach().cpu().tolist()
            raise ValueError(
                "no valid LUOP joint action for active batch rows "
                f"{rows}; constraints are infeasible for the remaining parcels"
            )

    def _type_one_hot(self, type_value):
        one_hot = torch.zeros(self.num_types, dtype=torch.bool)
        one_hot[type_value] = 1
        return one_hot
    def _type_one_hot_tensor(self, type_value):
        """
            将 type_value 转换为 one-hot 编码张量。

            参数:
            - type_value: 一个包含类型索引的一维整数张量。

            返回:
            - one_hot: 一个形状为 (batch_size, self.num_types) 的布尔类型 one-hot 编码张量。
            """
        num_types = self.num_types  # 获取类型总数

        safe_type = type_value.clamp(min=0, max=num_types - 1)
        return F.one_hot(safe_type, num_classes=num_types).bool()
    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None, planout=None, reward=0):
        return render(td, actions, ax, planout, reward)

    def close(self) -> None:
        pass
