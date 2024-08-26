import random

from typing import Optional
import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = landuseOptGenerator(**generator_params)
        self.generator = generator
        self.generator_params = generator_params
        self.num_types = 8
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

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        # Initialize locations
        device = td.device
        init_locs = td["locs"]
        areas = td["areas"]
        num_loc = init_locs.shape[-2]
        action_mask = td["fixed_mask"]

        self.num_types = 8
        self.max_steps = num_loc - 3
        # initialize the current plan
        current_plan = td["init_plan"]
        #batch_size = current_plan.shape[0]


        # initialize the land use type
        # TODO: how to initialize the land use type
        current_types_onehot = torch.zeros((*batch_size, 8), dtype=torch.bool, device=device)
        vector = self._type_one_hot(5) #shape = (8,)
        current_types_onehot = vector.expand(*batch_size, -1)

        # initialize the current node
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)

        # initialize the action mask (all nodes are available except the fixed nodes)
        # action_mask = torch.ones(
        #     (*batch_size, num_loc), dtype=torch.bool, device=device
        # )

        # Other variables
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        done = torch.zeros((*batch_size, 1), dtype=torch.bool, device=device)
        reward = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)
        return TensorDict(
            {
                "locs": init_locs,
                "areas": areas,
                "current_node": current_node,
                "current_plan": current_plan,
                "current_type": torch.zeros(*batch_size, dtype=torch.int64),
                "current_types_onehot": current_types_onehot,
                "i": i,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            },
            batch_size=batch_size,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]
        current_plan = td["current_plan"]
        areas = td["areas"]
        # update the current plan
        batch_size = current_plan.shape[0]
        current_type = td["current_type"]
        done = td["done"]
        # 更新 current_plan
        device = current_plan.device  # 获取当前设备
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)  # 批次索引
        valid_indices = current_type != -1
        valid_indices = valid_indices.to(device)
        # 只更新current_type不为-1的情况
        current_plan[batch_indices[valid_indices], current_node[valid_indices]] = current_type[valid_indices]
        # current_plan[batch_indices, current_node] = current_type
        #set the current type to the next type
        # current_types_onehot = torch.zeros((batch_size, self.num_types))
        # for i in range(batch_size):
        #     next_type = self.calc_next_type(current_plan[i], landtype, areas[i], current_type[i].item())
        #     if next_type is None or done[i]:
        #         done[i] = True
        #         continue
        #     current_type[i] = torch.tensor(next_type, dtype=torch.int64)
        #     current_types_onehot[i] = self._type_one_hot(next_type)
        next_types = self.calc_next_type_tensor(current_plan, areas, current_type)
        done |= next_types == None
        valid_indices = ~done
        valid_indices = valid_indices.squeeze()
        current_type[valid_indices] = next_types[valid_indices]
        current_types_onehot = self._type_one_hot_tensor(next_types)
        # Set available to 0 (i.e., already placed) if the current node is the first node
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # Set done if i is greater than max_steps
        # done = td["i"] >= self.max_steps - 1
        done = done | (td["i"] >= self.max_steps - 1)

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "current_plan" : current_plan,
                "current_type": current_type,
                "current_types_onehot": current_types_onehot,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            }
        )
        return td
    def _get_reward(self, td, actions):
        """
        We call the reward function with the final sequence of actions to get the reward
        reward = accessibility + compatibility
        """
        current_plan = td["current_plan"]
        distances = td["distances"]
        neighbors = td["adjacency_list"]
        if neighbors.shape[-2] == neighbors.shape[-1]:
            neighbors = self.adj_matrix_to_list(neighbors)
        else:
            neighbors = neighbors

        MinMaxCompatibility = [6.0, 8.8]
        MinMaxAccessibility = [-3.0, -1.5]
        neighbors = torch.tensor(neighbors, dtype=torch.long, device=device)  # Shape: [batch_size, num_centers, num_neighbours + 1]
        CompatibilityTable = torch.tensor(self.CompatibilityTable, dtype=torch.float32, device=device)  # Shape: [num_landuse_types, num_landuse_types]
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
        rewards = norcompatibility + noraccessibility
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
                # 将当前节点的邻接列表添加到批次的邻接列表中
                batch_adj_list.append(neighbors)
            # 将当前批次的邻接列表添加到总的邻接表中
            adj_list.append(batch_adj_list)
        return adj_list
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
            current_plan=BoundedTensorSpec(
                low=0,
                high=7,
                shape=(generator.num_loc,),
                dtype=torch.int64,
            ),
            current_types_onehot=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(8,),
                dtype=torch.float32,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc,),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)
    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        pass
    def calc_compatibility(self, plan, landtypes, neighbourlist, CompatibilityTable):
        # 计算兼容性
        strstate = init.map_to_strings(plan, landtypes)
        res = init.calCompatibility(landtypes, neighbourlist, strstate, CompatibilityTable)
        return res
    def calc_compatibility_tensor(self, landuselists, neighbourlists, CompatibilityTable):
        '''Calculate compatibility for batches of land use lists and neighbour lists using PyTorch tensors.
        Each batch element contains multiple centers with their respective neighbours.
        '''
        # Assuming:
        # landuselists shape: [batch_size, num_centers]
        # batch_neighbourlists shape: [batch_size, num_centers, num_neighbours_per_center + 1] (including neighbour count)

        # Remove the neighbour count column and get neighbour indices
        neighbour_indices = neighbourlists[:, :, 1:]
        # Create batch indices that match the shape of neighbour_indices
        # Expand the batch_indices to match the dimensions of neighbour_indices
        batch_indices = torch.arange(landuselists.shape[0], device=landuselists.device).unsqueeze(1).unsqueeze(2)
        batch_indices = batch_indices.expand(-1, landuselists.shape[1], 4)

        # Using the indices to gather the land use types of the neighbours
        neighbour_landuses = landuselists[batch_indices, neighbour_indices]

        # Get the compatibility scores using advanced indexing
        # CompatibilityTable should be indexed with the current land use and its neighbours' land use types
        compatibility_scores = CompatibilityTable[landuselists.unsqueeze(2), neighbour_landuses]

        # Sum compatibility scores across neighbours for each center
        compatibility_sums = compatibility_scores.sum(dim=2)

        # Compute average compatibility per batch element
        total_compatibility = compatibility_sums.mean(dim=1) / 4  # Division by 4 might be for normalization

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
        Accessibility = (init.calInterAccessibility_Average_tensor(plan,
                                                           ['Residential'],
                                                           ['Commercial'], distancelist)
                         + init.calInterAccessibility_Average_tensor(plan,
                                                              ['Residential'],
                                                              ['Education'], distancelist)
                         + init.calInterAccessibility_Average_tensor(plan,
                                                            ['Residential'],
                                                            ['Hospital'], distancelist)
                         + init.calInterAccessibility_Average_tensor(plan,
                                                            ['Residential'],
                                                            ['Office'], distancelist)
                         + init.calInterAccessibility_Average_tensor(plan,
                                                            ['Residential'],
                                                            ['Green Space'], distancelist))
        # Accessibility = (init.calInterAccessibility_Average_tensor(plan,
        #                                                            ['Residential', 'SOHO', 'Residential&Commercial'],
        #                                                            ['Commercial', 'Residential&Commercial'],
        #                                                            distancelist)
        #                  + init.calInterAccessibility_Average_tensor(plan,
        #                                                              ['Residential', 'SOHO', 'Residential&Commercial'],
        #                                                              ['Education'], distancelist)
        #                  + init.calInterAccessibility_Average_tensor(plan,
        #                                                              ['Residential', 'SOHO', 'Residential&Commercial'],
        #                                                              ['Hospital'], distancelist)
        #                  + init.calInterAccessibility_Average_tensor(plan,
        #                                                              ['Residential', 'SOHO', 'Residential&Commercial'],
        #                                                              ['Office', 'SOHO'], distancelist)
        #                  + init.calInterAccessibility_Average_tensor(plan,
        #                                                              ['Residential', 'SOHO', 'Residential&Commercial'],
        #                                                              ['Green Space'], distancelist))
        return Accessibility
    # def calc_next_type(self, current_plan, landtypes, areas):
    #     strstate = init.map_to_strings(current_plan.tolist(), landtypes)
    #     Residential_ratio = init.landuse_ratio(strstate, areas, 'Residential')
    #     Commercial_ratio = init.landuse_ratio(strstate, areas, 'Commercial')
    #     Education_ratio = init.landuse_ratio(strstate, areas, 'Education')
    #     Office_ratio = init.landuse_ratio(strstate, areas, 'Office')
    #     SOHO_ratio = init.landuse_ratio(strstate, areas, 'SOHO')
    #     RC_ratio = init.landuse_ratio(strstate, areas, 'Residential&Commercial')
    #     # 对每个批次进行计算
    #     if Education_ratio < 0.02:
    #         type = 'Education'
    #         return self.find_type_index(type)
    #     if Residential_ratio < 0.2:
    #         type = 'Residential'
    #         return self.find_type_index(type)
    #     if Commercial_ratio + RC_ratio < 0.1:
    #         type = 'Residential&Commercial'
    #         return self.find_type_index(type)
    #     if Commercial_ratio < 0.05:
    #         type = 'Commercial'
    #         return self.find_type_index(type)
    #     if Office_ratio < 0.05:
    #         type = 'Office'
    #         return self.find_type_index(type)
    #     if Office_ratio + SOHO_ratio < 0.15:
    #         type = 'SOHO'
    #         return self.find_type_index(type)
    #     if Residential_ratio + SOHO_ratio + RC_ratio < 0.5:
    #         type = random.choices(['Residential', 'SOHO', 'Residential&Commercial'], [0.3, 0.3, 0.4])[0]
    #         return self.find_type_index(type)
    #     else:
    #         type = random.choices(['Hospital', 'Green Space'], [0.5, 0.5])[0]
    #         return self.find_type_index(type)
    def calc_next_type(self, current_plan, landtypes, areas, current_type):
        strstate = init.map_to_strings(current_plan.tolist(), landtypes)
        tios = [0.051420568227290915, 0.20078031212484992, 0.05712284913965586, 0.42276910764305725,0.046318527410964386,0.03221288515406162,0.06562625050020007,0.12374949979991996]
        if init.landuse_ratio(strstate, areas, landtypes[current_type]) < tios[current_type]:
            return current_type
        else:
            if current_type == 7:
                return None
            return current_type + 1
    def calc_next_type_tensor(self, current_plan, areas, current_type):
        tios = [0.051420568227290915, 0.20078031212484992, 0.05712284913965586, 0.42276910764305725,0.046318527410964386,0.03221288515406162,0.06562625050020007,0.12374949979991996]
        tios = torch.tensor(tios).to(current_plan.device)
        area_ratios = init.landuse_ratio_tensor(current_plan, areas, current_type)
        area_exceeds_mask = area_ratios > tios[current_type]
        type_is_7_mask = current_type == 7
        next_types = current_type.where(~area_exceeds_mask, current_type + 1)

        # 对于面积比例超过阈值且类型为7的情况，将 next_types 设置为一个特定的值，例如 -1
        next_types[area_exceeds_mask & type_is_7_mask] = -1

        return next_types
    def find_type_index(self,search_type):
        landtype_array = np.array(landtype)
        index = np.where(landtype_array == search_type)[0]
        return index

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
        # 假设 self.num_types 是一个定义了的类属性，表示类型总数
        batch_size = type_value.size(0)  # 获取批次大小
        num_types = self.num_types  # 获取类型总数

        # 创建一个形状为 (batch_size, num_types) 的零张量，数据类型为布尔
        one_hot = torch.zeros((batch_size, num_types), dtype=torch.bool)

        # 使用 torch.arange 创建一个索引张量，其形状与 type_value 相同
        indices = torch.arange(batch_size).long().unsqueeze(1)  # 增加维度以匹配 type_value 的形状

        # 使用索引将 one-hot 张量的对应位置设置为 True
        one_hot[indices, type_value] = True

        return one_hot
    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None, planout=None):
        return render(td, actions, ax, planout)

    def close(self) -> None:
        pass