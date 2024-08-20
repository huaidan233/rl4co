
from typing import Callable, Union
import torch
from tensordict.tensordict import TensorDict

from rl4co.envs.common.utils import Generator, get_sampler
from torch.distributions import Exponential, Normal, Poisson, Uniform
from rl4co.utils.pylogger import get_pylogger

class landuseOptGenerator(Generator):
    """Data generator for the landuseOpt problem.

    Args:
        num_loc: number of locations (customers) in the landuseOpt
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        num_landtype: number of land types
        landtype: list of land types
        fixed_landtype: fixed land type for each location
        init_sol_type: the method type used for generating initial solutions (random or greedy)
        loc_distribution: distribution for the location coordinates

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        num_landtype: int = 8,
        num_fixed: int = 3,
        landtype: list = [0, 1, 2, 3, 4, 5, 6, 7],
        init_sol_type: str = "random",
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        area_distribution: Union[int, float, str, type, Callable] = Uniform,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.num_types = num_landtype
        self.max_loc = max_loc
        self.init_sol_type = init_sol_type
        self.num_fixed = num_fixed
        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

        # area distribution
        if kwargs.get("area_sampler", None) is not None:
            self.area_sampler = kwargs["area_sampler"]
        else:
            self.area_sampler = get_sampler(
                "area", area_distribution, 0, 1, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        # Generate random areas in the range [0, 1]
        areas = self.area_sampler.sample((*batch_size, self.num_loc))

        # Normalize areas to make their sum equal to 1
        areas = areas / areas.sum(dim=-1, keepdim=True)

        # init plans
        init_plan = torch.ones((*batch_size, self.num_loc), dtype=torch.int64)

        fixed_node = torch.randint(0, self.num_loc, (*batch_size, self.num_fixed))
        # Randomly select 'num_fixed' points to be inaccessible for each batch
        fixed_mask = torch.ones((*batch_size, self.num_loc), dtype=torch.bool)
        for i in range(batch_size[0]):

            # Assign fixed nodes to be "Green Space" or "Hospital"
            init_plan[i, fixed_node[i, :self.num_fixed // 2]] = 4 # "Green Space"
            init_plan[i, fixed_node[i, self.num_fixed // 2:]] = 6 # "Hospital"
            fixed_indices = torch.randperm(self.num_loc)[:self.num_fixed]
            fixed_mask[i, fixed_indices] = False
        # Calculate adjacency list for the nearest 4 neighbors
        # adjacency_list[0] is the count of neighbors
        # Calculate distances between all locations

        adjacency_list = torch.zeros((*batch_size, self.num_loc, 5), dtype=torch.int64)
        distances = torch.zeros((*batch_size, self.num_loc, self.num_loc))
        for i in range(batch_size[0]):
            distances[i] = torch.cdist(locs[i], locs[i])
            for j in range(self.num_loc):
                sorted_indices = torch.argsort(distances[i, j])[1:5]  # Skip the first one (distance to itself)
                adjacency_list[i, j, 0] = 4
                adjacency_list[i, j, 1:] = sorted_indices

        return TensorDict(
            {
                "locs": locs,
                "areas": areas,
                "init_plan": init_plan,  # "init_plans": "Green Space" or "Hospital"
                "fixed_mask": fixed_mask,
                "adjacency_list": adjacency_list,
                "distances": distances,
            },
            batch_size=batch_size,
        )

    def _get_initial_solutions(self, coordinates):
        """
        Generate initial solutions for the landuseOpt problem.
        """
        batch_size = coordinates.size(0)

        if self.init_sol_type == "random":
            set = torch.rand(batch_size, self.num_loc).argsort().long()
            rec = torch.zeros(batch_size, self.num_loc).long()
            index = torch.zeros(batch_size, 1).long()

            for i in range(self.num_loc - 1):
                rec.scatter_(1, set.gather(1, index + i), set.gather(1, index + i + 1))

            rec.scatter_(1, set[:, -1].view(-1, 1), set.gather(1, index))
