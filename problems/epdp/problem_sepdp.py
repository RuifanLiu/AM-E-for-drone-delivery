from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.epdp.state_sepdp import StateSEPDP
from problems.epdp.problem_epdp import EPDP, EPDPDataset
from utils.beam_search import beam_search


class SEPDP(EPDP):
    NAME = 'sepdp'  # Capacitated Vehicle Routing Problem

    
    
    @staticmethod
    def get_costs(dataset, pi):  # pi:[batch_size, graph_size]
        # assert (pi[:, 0]==0).all(), "not starting at depot"
        # assert (
        #     torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) == pi.data.sort(1)[0]).all(), "not visiting all nodes"


        batch_size, graph_size = dataset['load'].size()
        # Check that tours are valid, i.e. contain 0 to n - 1
        # assert (pi[:, 0]==graph_size).all(), "not starting at depot"
        

        # Sorting it should give all zeros at front and then 1...n
        sorted_pi = pi.data.sort(1)[0]
        assert (
            torch.arange(0, graph_size, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, :graph_size]
        ).all(), "Not visit all nodes"
        
        # Pickup and delivery precedent constraint
        load_with_node = torch.ones_like(dataset['load'][:, 0]) # first pickup task
                
        load_with_node = torch.cat(
            [torch.full_like(dataset['load'][:, :graph_size//2], 1),
             torch.full_like(dataset['load'][:, :graph_size//2], -1)],
            dim = 1)  # [batch_size, 1, graph_size+1], [1,1...1, 0...0]
        load_with_node = torch.cat(
            (load_with_node,
             torch.full_like(dataset['load'][:, :EPDP.STATION_NO+1], 0)),
            dim = 1)
        
        load_d = load_with_node.gather(1, pi)
        load_nb = torch.zeros_like(dataset['load'][:, 0])
        for i in range(pi.size(1)):
            load_nb += load_d[:,i]
            assert (load_nb<=1+1e-5).all() and (load_nb>=0-1e-5).all(), "Violate delivery precedent constraint"
        
        
        # visited_time = torch.argsort(pi, 1)  # pickup的index < 对应的delivery的index
        # assert (visited_time[:, 1:pi.size(1) // 2 + 1] < visited_time[:, pi.size(1) // 2 + 1:]).all(), "deliverying without pick-up"
        # dataset['depot']: [batch_size, 2], dataset['loc']: [batch_size, graph_size, 2]
        
        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        # Get recharged in depot and stations
        recharge_with_station = torch.cat(
            (
                torch.full_like(dataset['load'], 0),
                torch.full_like(dataset['load'][:, :EPDP.STATION_NO+1], -StateSEPDP.BATTERY_CAPACITY),
            ),
            1
        )
        recharge_d = recharge_with_station.gather(1, pi)


        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['loc'],dataset['station'],dataset['depot'][:, None, :]), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        used_energy = torch.zeros_like(dataset['load'][:, 0])
        energy_cost = torch.zeros_like(dataset['load'][:, 0])
        for i in range(pi.size(1)-1):
            edge_dist = (d[:, i] - d[:, i+1]).norm(p=2, dim=1)
            
            if StateSEPDP.SPEED_STD == 0:
                speed = StateSEPDP.SPEED_MEAN
            else:
                speed = torch.FloatTensor(batch_size,1).normal_(StateSEPDP.SPEED_MEAN, StateSEPDP.SPEED_STD).to(edge_dist.device)
            edge_cost = edge_dist/speed*StateSEPDP.DIS2SOC
            
            edge_cost = (d[:, i] - d[:, i+1]).norm(p=2, dim=1)*StateSEPDP.DIS2SOC
            used_energy += edge_cost
            used_energy += recharge_d[:,i] # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_energy[used_energy < 0] = 0
            energy_cost += edge_cost 
            energy_cost += StateSEPDP.PENALITY(used_energy <= EPDP.BATTERY_CAPACITY)
            # assert (used_energy <= EPDP.BATTERY_CAPACITY + 1e-5).all(), "Used more than capacity"
            
        energy_cost += ((d[:, 0] - dataset['depot']).norm(p=2, dim=1) 
                        + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)
                        )*EPDP.DIS2SOC

        return energy_cost, None
    @staticmethod
    def make_state(*args, **kwargs):
        return StateSEPDP.initialize(*args, **kwargs)
        
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)
        
        state = SEPDP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
#            input, visited_dtype=torch.int64 if compress_mask else torch.bool
        )
        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        return beam_search(state, beam_size, propose_expansions)

