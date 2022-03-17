import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StatePDP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
#     demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
#     used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    to_delivery: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
#                 used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                to_delivery=self.to_devliery[key],
            )
        return super(StatePDP, self).__getitem__(key)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
#         demand = input['demand']

        batch_size, n_loc, _ = loc.size()
        to_delivery=torch.cat([torch.ones(batch_size, 1, n_loc // 2 + 1, dtype=torch.uint8, device=loc.device),
            torch.zeros(batch_size, 1, n_loc // 2, dtype=torch.uint8, device=loc.device)], dim=-1)  # [batch_size, 1, graph_size+1], [1,1...1, 0...0]
        return StatePDP(
            coords=torch.cat((depot[:, None, :], loc), -2),
#             demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
#             used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            to_delivery=to_delivery
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

   
        # Update the state
        batch_size,_, n_loc = self.to_delivery.size()  # number of customers
        n_loc -= 1
        # the pair node of selected node
        
        pair_node = ((selected -1 + n_loc // 2) % (n_loc) + 1)* (selected>0)
        pair_node = pair_node[:, None]
        
        pickup_node = torch.arange(1, n_loc//2+1, device = self.to_delivery.device).view(1, -1).expand(batch_size, 1, n_loc//2)
        prev_pickup_flag = self.to_delivery[:,:,:n_loc//2]
        # self.to_delivery = torch.cat([self.to_delivery[:,:,:n_loc // 2],
        #                               torch.zeros(batch_size, 1, n_loc // 2, dtype=torch.uint8, device=self.to_delivery.device)], 
        #                               -1)  # [batch_size, 1, graph_size+1], [1,1...1, 0...0] *(selected >= n_loc//2).float()
         
        
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
#         n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        # selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]


        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            to_delivery = self.to_delivery.scatter(-1, pair_node[:, :, None], 1)
            
            # load_with_node = torch.full_like(dataset['load'][:, :n_loc//2], ),
            # if last visited node is delivery or last last visited is delivery 
            pickup_flag = ((prev_a > n_loc//2)&(prev_a <= n_loc)|(prev_a==0)).expand(batch_size, n_loc//2)[:,None,:].to(torch.uint8)
                          # |((prev_pickup_flag).bool()))
            
            to_delivery = to_delivery.scatter(-1, pickup_node, pickup_flag)
            
            # print(prev_a[159])
            # print(to_delivery[159])
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1, to_delivery=to_delivery,
        )

    def all_finished(self):
        return self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """


        visited_loc = self.visited_

        n_loc = visited_loc.size(-1) - 1  # num of customers
        batch_size = visited_loc.size(0)

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
#         exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(self.to_delivery.device) | (1 - self.to_delivery)

        # Cannot visit the depot if just visited and still unserved nodes
#         mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)

        if self.i == 0:  # [0,1...1]
            return torch.cat([torch.zeros(batch_size, 1, 1, dtype=torch.uint8, device=mask_loc.device),
                            torch.ones(batch_size, 1, n_loc, dtype=torch.uint8, device=mask_loc.device)], dim=-1) > 0

        return mask_loc > 0  # return true/false

    def construct_solutions(self, actions):
        return actions
