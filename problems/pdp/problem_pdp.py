from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.pdp.state_pdp import StatePDP
from utils.beam_search import beam_search


class PDP(object):

    NAME = 'pdp'  # Capacitated Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, pi):  # pi:[batch_size, graph_size]
        '''
        # assert (pi[:, 0]==0).all(), "not starting at depot"
        assert (torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) == pi.data.sort(1)[0]).all(), "not visiting all nodes"

        visited_time = torch.argsort(pi, 1)  # pickup的index < 对应的delivery的index
        # assert (visited_time[:, 1:pi.size(1) // 2 + 1] < visited_time[:, pi.size(1) // 2 + 1:]).all(), "deliverying without pick-up"
        # dataset['depot']: [batch_size, 2], dataset['loc']: [batch_size, graph_size, 2]
        dataset = torch.cat([dataset['depot'].reshape(-1, 1, 2), dataset['loc']], dim = 1)  # [batch, graph_size+1, 2]
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))  # [batch, graph_size+1, 2]
        # d[:, :-1] do not include -1
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - dataset['depot']).norm(p=2, dim=1) \
                + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)
        '''
        batch_size, task_size, _ = dataset['loc'].size()
        graph_size = task_size + 1
        # Check that tours are valid, i.e. contain 0 to n - 1
        # assert (pi[:, 0]==graph_size).all(), "not starting at depot"
        
        
        # Sorting it should give all zeros at front and then 1...n
        sorted_pi = pi.data.sort(1)[0]
        assert (
            torch.arange(0, graph_size, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, :graph_size]
        ).all(), "Not visit all nodes"
        
        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['loc'],dataset['depot'][:, None, :]), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        cost = torch.zeros(batch_size).to(dataset['loc'].device)
        for i in range(pi.size(1)-1):
            cost += (d[:, i] - d[:, i+1]).norm(p=2, dim=1)
            
        cost += ((d[:, 0] - dataset['depot']).norm(p=2, dim=1) 
                        + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)
                        )
        return cost, None


    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDP.initialize(*args, **kwargs)
        
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)
        
        state = PDP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
#            input, visited_dtype=torch.int64 if compress_mask else torch.bool
        )
        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        return beam_search(state, beam_size, propose_expansions)

def make_instance(args):
    depot, loc, *args = args
    # grid_size = 1
    # if len(args) > 0:
    #     depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float),
        'depot': torch.tensor(depot, dtype=torch.float)
    }


class PDPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(PDPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
