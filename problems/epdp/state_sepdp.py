import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
from problems.epdp.state_epdp import StateEPDP


class StateSEPDP(StateEPDP):
     # std of ground speed when airspeed = 20m/s, mean of wind speed = 5m/s
     SPEED_STD = 3.59/20
     SPEED_MEAN = 20/20
     
     MEAN_WIND = 5.0 # mean value for weibull distribution
     
     @staticmethod
     def initialize(input, visited_dtype=torch.uint8):
    
        depot = input['depot']
        station = input['station']
        loc = input['loc']
        load = input['load']
        wind_mag = input['wind_mag']*StateSEPDP.MEAN_WIND
        wind_dir = input['wind_dir']

        batch_size, n_loc, _ = loc.size()
        _, n_station, _ = station.size()
        ''' Set mask for task node: 1 - feasible, 0-infeasible'''
        to_delivery=torch.cat([torch.ones(batch_size, 1, n_loc // 2, dtype=torch.uint8, device=loc.device),
            torch.zeros(batch_size, 1, n_loc // 2, dtype=torch.uint8, device=loc.device)], dim=-1)  # [batch_size, 1, graph_size+1], [1,1...1, 0...0]
        
        return StateSEPDP(
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
        '''
        
    def get_final_cost(self):
        
        batch_size, n_loc = self.load.size()
        final_cost = self.total_cost + StateEPDP.DIS2SOC*(self.coords[self.ids, -1, :] - self.cur_coord).norm(p=2, dim=-1)
        
        if not self.all_finished():
            visited_loc = self.visited_[:, :, :n_loc]
            final_cost += StateSEPDP.PENALITY*(visited_loc==0).sum(-1)

        return final_cost
    
     def get_failure_rate(self):
        
        failed_rate = torch.mean(self.failed_flag.float())
        
        return failed_rate
     def update(self, selected):
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        batch_size,_, n_loc = self.to_delivery.size()  # number of customers
        
        delivery_node = (selected + n_loc // 2) % (n_loc)*(selected//n_loc==0) # the pair node of selected node
        delivery_node = delivery_node[:, None]
        
        pickup_node = torch.arange(0, n_loc//2, device = self.to_delivery.device).view(1, -1).expand(batch_size, 1, n_loc//2)
        pickup_old = self.to_delivery[:,:,:n_loc//2]
        
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # energy consumption 
        distance = (cur_coord - self.cur_coord).norm(p=2, dim=-1)
        if StateSEPDP.SPEED_STD == 0:
            speed = StateSEPDP.SPEED_MEAN
        else:
            speed = torch.FloatTensor(batch_size,1).normal_(StateSEPDP.SPEED_MEAN, StateSEPDP.SPEED_STD).to(selected.device)
        energy_cost = distance/speed*StateSEPDP.DIS2SOC

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (self.used_capacity + energy_cost
                         ) * (prev_a < n_loc).float()
        
        # add energy cost to total cost
        # extra penalty if exceed the battery capacity
        total_cost = self.total_cost + energy_cost + \
            StateSEPDP.PENALITY*(self.used_capacity+energy_cost > StateSEPDP.BATTERY_CAPACITY)# (batch_dim, 1)
        failed_flag = self.failed_flag | (self.used_capacity+energy_cost > StateSEPDP.BATTERY_CAPACITY) 
        # total_cost = self.total_cost + energy_cost

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            to_delivery = self.to_delivery.scatter(-1, delivery_node[:, :, None], 1)
            
            # load_with_node = torch.full_like(dataset['load'][:, :n_loc//2], ),
            pickup_new = (((prev_a >= n_loc//2)&(prev_a < n_loc)).expand(batch_size, n_loc//2)[:,None,:]
                          |((prev_a >= n_loc).expand(batch_size, n_loc//2)[:,None,:]&(pickup_old).bool())).to(torch.uint8)
            
            to_delivery = to_delivery.scatter(-1, pickup_node, pickup_new)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            total_cost=total_cost, cur_coord=cur_coord, i=self.i + 1, to_delivery=to_delivery,
            failed_flag = failed_flag
        )
    


     def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
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
        speed_boundary = StateSEPDP.SPEED_MEAN - 1.96*StateSEPDP.SPEED_STD # 95 internal confidence
        # speed_boundary = 1
        
        
        energy_required = (self.cur_coord - self.coords[self.ids, :].squeeze(1)).norm(p=2, dim=-1)*StateEPDP.DIS2SOC/speed_boundary
        energy_to_station = (self.coords[:,:,None,:] - self.coords[:,n_loc:,:][:,None,:,:]).norm(p=2, dim=-1)*StateEPDP.DIS2SOC/speed_boundary
        energy_to_station = torch.min(energy_to_station,dim=-1).values
        exceeds_cap = (energy_required[:,None,:] + self.used_capacity[:, :, None] + energy_to_station[:,None,:] > self.BATTERY_CAPACITY)
        # # Nodes that cannot be visited are already visited or too much demand to be served now
        
        
        # unmask the station that exceed the capacity
        exceeds_cap[:,:,n_loc:] = 0
        mask_all = mask_all.to(exceeds_cap.dtype) | exceeds_cap
        return mask_all
'''






