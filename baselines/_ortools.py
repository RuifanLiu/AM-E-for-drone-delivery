from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import torch

def _solve_cp(input):
    
    
    AMP = 10000 # Amplify index, since or-tools only solve integer problems
    DIS2SOC = 0.179
    
    depot = input['depot']
    station = input['station']
    loc = input['loc']
    load = input['load']
    
    loc_all = torch.cat((loc, station, depot[None, :]), 0)
    
    node_count,_ = loc_all.size()
    stat_count = station.size(0)
    req_count = int((node_count-stat_count-1)/2)
    veh_count = 1
    
    pickup_node = np.arange(0,req_count)
    delivery_node = pickup_node+req_count
    pickup_delivery_pair = np.concatenate([pickup_node[:,None], delivery_node[:,None]],1)
    
    station_node = np.arange(req_count*2,node_count)
    
    
    x_loc = loc_all[:,None,:].expand(-1,node_count,-1)
    y_loc = loc_all[None,:,:].expand(node_count,-1,-1)
    
    dis_matrix = (x_loc - y_loc).norm(p=2, dim=-1)
    
    en_cost_matrix = dis_matrix*DIS2SOC*AMP
    
    cust_load = np.concatenate([np.ones(req_count), -1*np.ones(req_count), np.zeros(1+stat_count)])

    manager = pywrapcp.RoutingIndexManager(loc_all.size(0), veh_count, node_count-1)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        src = manager.IndexToNode(from_idx)
        dst = manager.IndexToNode(to_idx)
        return int(en_cost_matrix[src][dst])
    d_cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(d_cb_idx)
    
    routing.AddDimension(
        d_cb_idx,
        0,  # null capacity slack
        10000000,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Energy')
    energy_dimension = routing.GetDimensionOrDie('Energy')
    
    # Define Transportation Requests.
    for request in pickup_delivery_pair:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            energy_dimension.CumulVar(pickup_index) <=
            energy_dimension.CumulVar(delivery_index))
        
    def soc_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node in station_node:
            soc_status = -1*AMP
        else:
            soc_status = int(en_cost_matrix[from_node][to_node])
        return soc_status
    
    soc_callback_index = routing.RegisterTransitCallback(soc_callback)

    routing.AddDimension(
        soc_callback_index,
        1*AMP,
        10000000,
        True,
        'SOC'
        )
    SOC_dimension = routing.GetDimensionOrDie('SOC')
    
    for node_idx in range(len(en_cost_matrix)):
        index = manager.NodeToIndex(node_idx)
        SOC_dimension.CumulVar(index).SetRange(0, int(0.7*AMP))
            
    def load_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return cust_load[from_node]

    load_callback_index = routing.RegisterUnaryTransitCallback(load_callback)
    
    routing.AddDimension(
        load_callback_index,
        0,  # null capacity slack
        1,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Load')
    
    # Allow to drop nodes.
    penalty = 2*AMP
    for node in range(req_count*2):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    penalty = 0
    for node in station_node:
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    
    # if nodes.size(1) > 3:
    #     horizon = int(nodes[0,4])
    #     def time_cb(from_idx, to_idx):
    #         src = manager.IndexToNode(from_idx)
    #         dst = manager.IndexToNode(to_idx)
    #         return int(nodes[src, 5] + nodes[src, :2].sub(nodes[dst, :2]).pow(2).sum().pow(0.5) / veh_speed)
    #     t_cb_idx = routing.RegisterTransitCallback(time_cb)
    #     routing.AddDimension(t_cb_idx, horizon, 2*horizon, True, "Time")
    #     t_dim = routing.GetDimensionOrDie("Time")
    #     for j, (e,l) in enumerate(nodes[1:,3:5], start = 1):
    #         idx = manager.NodeToIndex(j)
    #         t_dim.CumulVar(idx).SetMin(int(e))
    #         t_dim.SetCumulVarSoftUpperBound(idx, int(l), late_cost)
    #     for i in range(veh_count):
    #         idx = routing.End(i)
    #         t_dim.SetCumulVarSoftUpperBound(idx, horizon, late_cost)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(1)
    
    solution = routing.SolveWithParameters(params)

    # routes = []
    # for i in range(veh_count):
    #     route = []
    #     idx = routing.Start(i)
    #     while not routing.IsEnd(idx):
    #         idx = assign.Value(routing.NextVar(idx))
    #         route.append( manager.IndexToNode(idx) )
    #     routes.append(route)
    index = routing.Start(0)
    pathes = []
    for i in range(veh_count):
        path = []
        index = routing.Start(i)
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))   
        path.append(manager.IndexToNode(index))
        pathes.append(path)
    total_cost = solution.ObjectiveValue()/AMP
        
    # soc_dimension = routing.GetDimensionOrDie('SOC')
    # print('Objective: {} miles'.format(solution.ObjectiveValue()/AMP))
    # index = routing.Start(0)
    # plan_output = 'Route for vehicle 0:\n'
    # soc_output = 'SOC for vehicle0:\n'
    # path = []
    # route_distance = 0
    # soc_var = 0
    # while not routing.IsEnd(index):
    #     # plan_output += ' {} ->'.format(manager.IndexToNode(index))
    #     soc_var = soc_dimension.CumulVar(index)
    #     plan_output += ' {},'.format(manager.IndexToNode(index))
    #     soc_output += ' {},'.format(solution.Min(soc_var)/AMP)
    #     path.append(manager.IndexToNode(index))
    #     previous_index = index
    #     index = solution.Value(routing.NextVar(index))
    #     route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    # plan_output += ' {}\n'.format(manager.IndexToNode(index))    
    # path.append(manager.IndexToNode(index))
    # soc_var = soc_dimension.CumulVar(index)
    # soc_output += ' {}\n'.format(solution.Min(soc_var)/AMP)
    
    # route_distance = route_distance/AMP
    # plan_output += 'Route distance: {}miles\n'.format(route_distance)
    
    # print(plan_output)
    # print(soc_output)

    return pathes, total_cost


def ort_solve(dataset, mp, late_cost = 1):

    if mp:
        n_process = 30
        with Pool(n_process) as p:
            with tqdm(desc = "Calling ORTools", total = dataset.size) as pbar:
                results = [p.apply_async(_solve_cp, ([data]),
                    callback = lambda _:pbar.update()) for data in dataset.data_gen()]
                routes = [res.get()[0] for res in results]
                costs = [res.get()[1] for res in results]
        return routes, costs
    else:
        routes = []
        costs = []
        for data in tqdm(dataset.data_gen(),desc = "Calling ORTools", total = dataset.size):
            route, cost= _solve_cp(data)
            routes.append(route)
            costs.append(cost)

        return routes, costs
