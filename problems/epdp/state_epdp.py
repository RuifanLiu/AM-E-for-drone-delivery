import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
from utils.energy_cost_fun import energy_cost_fun


class StateEPDP(NamedTuple):
    
    STATION_NO = 3
    
    ## Calculation of transfer coefficient from distance to state of the charge
    ## with the energy consumption model when airspeed = 20m/s:
    ##   E = 1933.2* distance(m)/ground_speed(m/s)
    ## aftering uniformalize the distance with the area length (10km) and speed (20m/s)
    ##  E = 1933.2*10*1000*Dis/(20*Vel)
    ## Also, assume the battery capacity is 1.5kWh, the state of charge is:
    ##  SOC = E/(1.5*1000*3600)
    ## so:  SOC = 1933.2*10*1000*Dis/(20*Vel)/(1.5*1000*3600)
    # DIS2SOC = 0.179 # linear coefficient transfer distance to state of charge
    # DIS2SOC = 10/60*(0.7567)/0.7333
    BATTERY_CAPACITY = 1.0
    
    PENALITY = 1.0
    MEAN_WIND = 0.0 # mean value for weibull distribution
    
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    load: torch.Tensor
    wind_mag: torch.Tensor
    wind_dir: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    total_cost: torch.Tensor
    failed_flag: torch.Tensor
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
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                total_cost=self.total_cost[key],
                failed_flag=self.failed_flag[key],
                cur_coord=self.cur_coord[key],
                to_delivery=self.to_devliery[key],
            )
        return super(StateEPDP, self).__getitem__(key)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        station = input['station']
        loc = input['loc']
        load = input['load']
        wind_mag = input['wind_mag']*StateEPDP.MEAN_WIND
        wind_dir = input['wind_dir']

        batch_size, n_loc, _ = loc.size()
        _, n_station, _ = station.size()
        ''' Set mask for task node: 1 - feasible, 0-infeasible'''
        to_delivery=torch.cat([torch.ones(batch_size, 1, n_loc // 2, dtype=torch.uint8, device=loc.device),
            torch.zeros(batch_size, 1, n_loc // 2, dtype=torch.uint8, device=loc.device)], dim=-1)  # [batch_size, 1, graph_size+1], [1,1...1, 0...0]
        
        return StateEPDP(
            coords=torch.cat(( loc, station, depot[:, None, :]), -2),
            load=load,
            wind_mag=wind_mag,
            wind_dir=wind_dir,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=(n_loc+n_station)*torch.ones(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=load.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + n_station + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            total_cost=torch.zeros(batch_size, 1, device=loc.device),
            failed_flag=torch.zeros(batch_size, 1, dtype=torch.bool, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            to_delivery=to_delivery
        )

    def get_final_cost(self):
        
        # assert self.all_finished()
        # final_cost = self.total_cost + StateEPDP.DIS2SOC*(self.coords[self.ids, -1, :] - self.cur_coord).norm(p=2, dim=-1)
        batch_size, n_loc = self.load.size()
        final_cost = self.total_cost + energy_cost_fun((self.coords[self.ids, -1, :], self.cur_coord), self.wind_mag,\
                                                   self.wind_dir, command='airspeed', command_speed=15)
        if not self.all_finished():
            visited_loc = self.visited_[:, :, :n_loc]
            final_cost += StateEPDP.PENALITY*(visited_loc==0).sum(-1)

        return final_cost
    
    def get_failure_rate(self):
        
        failed_rate = torch.mean(self.failed_flag.float(),-1)
        return failed_rate

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        batch_size,_, n_loc = self.to_delivery.size()  # number of customers
        
        delivery_node = (selected + n_loc // 2) % (n_loc)*(selected//n_loc==0) # the pair node of selected node
        delivery_node = delivery_node[:, None]
        
        pickup_node = torch.arange(0, n_loc//2, device = self.to_delivery.device).view(1, -1).expand(batch_size, 1, n_loc//2)
        pickup_old = self.to_delivery[:,:,:n_loc//2]
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
        energy_cost = energy_cost_fun((cur_coord, self.cur_coord), self.wind_mag,\
                                                   self.wind_dir, command='airspeed', command_speed=15)
            
        
        total_cost = self.total_cost + energy_cost + \
            StateEPDP.PENALITY*(self.used_capacity + energy_cost > StateEPDP.BATTERY_CAPACITY)
        
        
        failed_flag = self.failed_flag | (self.used_capacity + energy_cost > StateEPDP.BATTERY_CAPACITY)
        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        # selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + energy_cost) * (prev_a < n_loc).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            to_delivery = self.to_delivery.scatter(-1, delivery_node[:, :, None], 1)
            
            # load_with_node = torch.full_like(dataset['load'][:, :n_loc//2], ),
            pickup_new = (((prev_a >= n_loc//2)&(prev_a < n_loc)).expand(batch_size, n_loc//2)[:,None,:]
                          |((prev_a >= n_loc).expand(batch_size, n_loc//2)[:,None,:]&(pickup_old).bool())).to(torch.uint8)
            
            to_delivery = to_delivery.scatter(-1, pickup_node, pickup_new)
            
            # print(prev_a[159])
            # print(to_delivery[159])
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            total_cost=total_cost, cur_coord=cur_coord, i=self.i + 1, to_delivery=to_delivery,
            failed_flag = failed_flag
        )

    def all_finished(self):
        batch_size, n_loc = self.load.size()
        return self.visited[:,:,:n_loc].all()

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


#         visited_loc = self.visited_

#         n_loc = visited_loc.size(-1) - 1  # num of customers
#         batch_size = visited_loc.size(0)

#         # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
# #         exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
#         # Nodes that cannot be visited are already visited or too much demand to be served now
#         mask_loc = visited_loc.to(self.to_delivery.device) | (1 - self.to_delivery)

#         # Cannot visit the depot if just visited and still unserved nodes
# #         mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)

#         if self.i == 0:  # [0,1...1]
#             return torch.cat([torch.zeros(batch_size, 1, 1, dtype=torch.uint8, device=mask_loc.device),
#                             torch.ones(batch_size, 1, n_loc, dtype=torch.uint8, device=mask_loc.device)], dim=-1) > 0

#         return mask_loc > 0  # return true/false
        batch_size, n_loc = self.load.size()
    
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, :n_loc]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.load.size(-1))
        mask_loc = visited_loc.to(self.to_delivery.dtype) | (1 - self.to_delivery)

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a >= n_loc) & ((mask_loc == 0).int().sum(-1) > 0)
        
        mask_all = torch.cat((mask_loc, mask_depot[:,:,None].expand(batch_size, 1, 1+StateEPDP.STATION_NO)), -1)
        
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        energy_required = energy_cost_fun((self.cur_coord, self.coords[self.ids, :].squeeze(1)), \
                                          torch.zeros_like(self.wind_mag),  torch.zeros_like(self.wind_dir),\
                                              command='airspeed', command_speed=15)
        energy_to_station = energy_cost_fun((self.coords[:,:,None,:], self.coords[:,n_loc:,:][:,None,:,:]), \
                                          torch.zeros_like(self.wind_mag), torch.zeros_like(self.wind_dir),\
                                              command='airspeed', command_speed=15)
            
        # energy_required = energy_cost_fun((self.cur_coord, self.coords[self.ids, :].squeeze(1)), \
        #                                   self.wind_mag, self.wind_dir,\
        #                                       command='airspeed', command_speed=15)
        # energy_to_station = energy_cost_fun((self.coords[:,:,None,:], self.coords[:,n_loc:,:][:,None,:,:]), \
        #                                   self.wind_mag, self.wind_dir,\
        #                                       command='airspeed', command_speed=15)
            
        energy_to_station = torch.min(energy_to_station,dim=-1).values
        exceeds_cap = (energy_required[:,None,:] + self.used_capacity[:, :, None] + energy_to_station[:,None,:] > self.BATTERY_CAPACITY)
        exceeds_cap[:,:,n_loc:] = 0
        # # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_all = mask_all.to(exceeds_cap.dtype) | exceeds_cap
        return mask_all

    def construct_solutions(self, actions):
        return actions
    
    
        #     energy_required = (self.cur_coord - self.coords[self.ids, :].squeeze(1)).norm(p=2, dim=-1)*StateEPDP.DIS2SOC
        # energy_to_station = (self.coords[:,:,None,:] - self.coords[:,n_loc:,:][:,None,:,:]).norm(p=2, dim=-1)*StateEPDP.DIS2SOC
