from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.epdp.state_epdp import StateEPDP
from utils.beam_search import beam_search
import numpy as np


class EPDP(object):

    NAME = 'epdp'  # Capacitated Vehicle Routing Problem
    STATION_NO = 3
    BATTERY_CAPACITY = 1.0 # (w.l.o.g. energy capacity is 1, energy consumption should be scaled)
    
    DIS2SOC = 0.179 # linear coefficient transfer distance to state of charge
    
    
    

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
                torch.full_like(dataset['load'][:, :EPDP.STATION_NO+1], -EPDP.BATTERY_CAPACITY),
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
            edge_cost = (d[:, i] - d[:, i+1]).norm(p=2, dim=1)*EPDP.DIS2SOC
            used_energy += edge_cost
            used_energy += recharge_d[:,i] # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_energy[used_energy < 0] = 0
            energy_cost += edge_cost 
            assert (used_energy <= EPDP.BATTERY_CAPACITY + 1e-5).all(), "Used more than capacity"
            
        energy_cost += ((d[:, 0] - dataset['depot']).norm(p=2, dim=1) 
                        + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)
                        )*EPDP.DIS2SOC

        return energy_cost, None


    @staticmethod
    def make_dataset(*args, **kwargs):
        return EPDPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateEPDP.initialize(*args, **kwargs)
        
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)
        
        state = EPDP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
#            input, visited_dtype=torch.int64 if compress_mask else torch.bool
        )
        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        return beam_search(state, beam_size, propose_expansions)

def make_instance(args):
    loc, load, station, depot, wind_mag, wind_dir, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'load': torch.tensor(load, dtype=torch.float),
        'station': torch.tensor(station, dtype=torch.float) / grid_size,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'wind_mag': torch.tensor([wind_mag], dtype=torch.float),
        'wind_dir': torch.tensor([wind_dir], dtype=torch.float),
    }



class EPDPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=None, distribution=None):
        super(EPDPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            # self.data = [make_instance(args) for args in data[offset:offset+num_samples]]
            if offset: 
                self.data = [make_instance(data[args]) for args in offset]
            else:
                self.data = [make_instance(args) for args in data]

        else:

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'load': torch.FloatTensor(size).uniform_(0, 4), # Weight of delivery package, Uniform 0-4
                    'station': torch.FloatTensor(3, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    'wind_mag': torch.Tensor(np.random.weibull(2.0, 1)),
                    'wind_dir': torch.FloatTensor(1).uniform_(0, 1)*2*np.pi,
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
    
    def data_gen(self):
        yield from self.data