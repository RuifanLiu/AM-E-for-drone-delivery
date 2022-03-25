#!/usr/bin/env python
# This Python file uses the following encoding: utf-8
# Copyright 2015 Tin Arm Engineering AB
# Copyright 2018 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Capacitated Vehicle Routing Problem (CVRP).

   This is a sample using the routing library python wrapper to solve a CVRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

from __future__ import print_function
from collections import namedtuple
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import math
import torch
import time
import numpy as np
from scipy.spatial import distance_matrix
from utils.energy_cost_fun import energy_cost_fun

###########################
# Problem Data Definition #
###########################
# Vehicle declaration
Vehicle = namedtuple('Vehicle', ['capacity'])


class Int_transfer():
    def __init__(self, AMP = 10000):
        self.AMP = AMP # Amplify index, since or-tools only solve integer problems
        
    def Float2Int(self, v):
        return (v * self.AMP + 0.5).astype(np.int32)
    
    def Int2Float(self, v):
        return v/self.AMP



def solve_epdp_ortools(loc, load, station, depot, wind_mag, wind_dir, sec_local_search=0, battery_margin=0):
    AMP = 10000000
    Tran = Int_transfer(AMP)
    DIS2SOC = 0.179
    # DIS2SOC = 10/60*(0.7567)/0.7333
    
    station = np.array(station)
    loc = np.array(loc)
    depot = np.array(depot)
    
    stat_count = station.shape[0]
    node_count = loc.shape[0] + stat_count + 1
    req_count = int((node_count-stat_count-1)/2)
    veh_count = 1
    
    wind_mag = np.zeros_like([wind_mag])
    wind_dir = np.zeros_like([wind_dir])
    
    loc_all = np.concatenate((loc, station, depot[None,:]), 0)
    
    dis_matrix = distance_matrix(loc_all,loc_all)

    en_cost_matrix = energy_cost_fun((loc_all[None,:,None,:], loc_all[None,None,:,:]), wind_mag[None,:], wind_dir[None,:])
    en_cost_matrix = Tran.Float2Int(np.array(en_cost_matrix.squeeze(0).cpu()))
    
    station_copynum = int(req_count/10)-1
    stat_count0 = stat_count
    # Treat depot as an additional station
    stat_count = stat_count+1
    for _ in range(station_copynum):
        en_cost_matrix = np.column_stack((en_cost_matrix,en_cost_matrix[:,-stat_count:]))
        en_cost_matrix = np.row_stack((en_cost_matrix,en_cost_matrix[-stat_count:,:]))
    en_cost_matrix = np.column_stack((en_cost_matrix,en_cost_matrix[:,-1:]))
    en_cost_matrix = np.row_stack((en_cost_matrix,en_cost_matrix[-1:,:]))  
    stat_count = (stat_count)*(station_copynum+1)
    node_count = req_count*2 + stat_count + 1
    
    pickup_node = np.arange(0,req_count)
    delivery_node = pickup_node+req_count
    pickup_delivery_pair = np.concatenate([pickup_node[:,None], delivery_node[:,None]],1)
    
    station_node = np.arange(req_count*2,node_count-1)
    
    cust_load = np.concatenate([np.ones(req_count), -1*np.ones(req_count), np.zeros(1+stat_count)])

    manager = pywrapcp.RoutingIndexManager(node_count, veh_count, node_count-1)
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
        1000*AMP,  # vehicle maximum capacities
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
        AMP*1,
        1000*AMP,
        True,
        'SOC'
        )
    SOC_dimension = routing.GetDimensionOrDie('SOC')
    
    for node_idx in range(len(en_cost_matrix)):
        index = manager.NodeToIndex(node_idx)
        SOC_dimension.CumulVar(index).SetRange(0, int(AMP*(1-battery_margin)))
            
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
    penalty = 1*AMP
    for node in range(req_count*2):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    penalty = 0
    for node in station_node:
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
    params.time_limit.seconds = sec_local_search
    # params.solution_limit = 100

    solution = routing.SolveWithParameters(params)

    index = routing.Start(0)
    pathes = []
    cost_output = 'Energy cost for vehicle0:\n'
    for i in range(veh_count):
        path = []
        index = routing.Start(i)
        while not routing.IsEnd(index):
            next_node = manager.IndexToNode(index)
            while next_node>=node_count:
                next_node -= stat_count
            path.append(next_node)
            
            cost_var = energy_dimension.CumulVar(index)
            cost_output += ' {},'.format(Tran.Int2Float(solution.Min(cost_var)))
            
            index = solution.Value(routing.NextVar(index))   
        next_node = manager.IndexToNode(index)
        path.append(next_node)
        
        path_adjust = np.copy(path)
        for _ in range(station_copynum):
            for ii in range(len(path_adjust)):
                if path_adjust[ii] >= req_count*2 + stat_count0 + 1:
                    path_adjust[ii] -= stat_count0 + 1
                
        for ii in range(len(path_adjust)):
            if path_adjust[ii] >= req_count*2 + stat_count0 + 1:
                path_adjust[ii] -= 1
        
        pathes.append(path_adjust)
        
        cost_var = energy_dimension.CumulVar(index)
        cost_output += ' {}\n'.format(Tran.Int2Float(solution.Min(cost_var)))
        
    # print(pathes)
    #print(cost_output)
        
    total_cost = Tran.Int2Float(solution.ObjectiveValue())
    # print('Objetive: {}'.format(total_cost))
    return total_cost, pathes

    #print_solution(data, routing, assignment)



# class DataProblem():
#   """Stores the data for the problem"""

#   def __init__(self, depot, loc, prize, max_length):
#     """Initializes the data for the problem"""
#     # Locations in block unit
#     self._locations = [(float_to_scaled_int(l[0]), float_to_scaled_int(l[1])) for l in [depot] + loc]

#     self._prizes = [float_to_scaled_int(v) for v in prize]

#     self._max_length = float_to_scaled_int(max_length)

#   @property
#   def vehicle(self):
#     """Gets a vehicle"""
#     return Vehicle()

#   @property
#   def num_vehicles(self):
#     """Gets number of vehicles"""
#     return 1

#   @property
#   def locations(self):
#     """Gets locations"""
#     return self._locations

#   @property
#   def num_locations(self):
#     """Gets number of locations"""
#     return len(self.locations)

#   @property
#   def depot(self):
#     """Gets depot location index"""
#     return 0

#   @property
#   def prizes(self):
#     """Gets prizes at each location"""
#     return self._prizes

#   @property
#   def max_length(self):
#       """Gets prizes at each location"""
#       return self._max_length


# #######################
# # Problem Constraints #
# #######################
# def euclidian_distance(position_1, position_2):
#   """Computes the Euclidian distance between two points"""
#   return int(math.sqrt((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2) + 0.5)


# class CreateDistanceEvaluator(object):  # pylint: disable=too-few-public-methods
#   """Creates callback to return distance between points."""

#   def __init__(self, data):
#     """Initializes the distance matrix."""
#     self._distances = {}

#     # precompute distance between location to have distance callback in O(1)
#     for from_node in xrange(data.num_locations):
#       self._distances[from_node] = {}
#       for to_node in xrange(data.num_locations):
#         if from_node == to_node:
#           self._distances[from_node][to_node] = 0
#         else:
#           self._distances[from_node][to_node] = (
#               euclidian_distance(data.locations[from_node],
#                                  data.locations[to_node]))

#   def distance_evaluator(self, from_node, to_node):
#     """Returns the manhattan distance between the two nodes"""
#     return self._distances[from_node][to_node]


# class CreatePrizeEvaluator(object):  # pylint: disable=too-few-public-methods
#   """Creates callback to get prizes at each location."""

#   def __init__(self, data):
#     """Initializes the prize array."""
#     self._prizes = data.prizes

#   def prize_evaluator(self, from_node, to_node):
#     """Returns the prize of the current node"""
#     del to_node
#     return self._prizes[from_node]


# def add_capacity_constraints(routing, data, prize_evaluator):
#   """Adds capacity constraint"""
#   capacity = 'Capacity'
#   routing.AddDimension(
#       prize_evaluator,
#       0,  # null capacity slack
#       data.vehicle.capacity,
#       True,  # start cumul to zero
#       capacity)


# def add_distance_constraint(routing, distance_evaluator, maximum_distance):
#     """Add Global Span constraint"""
#     distance = "Distance"
#     routing.AddDimension(
#         distance_evaluator,
#         0, # null slack
#         maximum_distance, # maximum distance per vehicle
#         True, # start cumul to zero
#         distance)


###########
# Printer #
###########
# def print_solution(data, routing, assignment):
#   """Prints assignment on console"""
#   print('Objective: {}'.format(assignment.ObjectiveValue()))
#   total_distance = 0
#   total_load = 0
#   capacity_dimension = routing.GetDimensionOrDie('Capacity')
#   for vehicle_id in xrange(data.num_vehicles):
#     index = routing.Start(vehicle_id)
#     plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
#     distance = 0
#     while not routing.IsEnd(index):
#       load_var = capacity_dimension.CumulVar(index)
#       plan_output += ' {} Load({}) -> '.format(
#           routing.IndexToNode(index), assignment.Value(load_var))
#       previous_index = index
#       index = assignment.Value(routing.NextVar(index))
#       distance += routing.GetArcCostForVehicle(previous_index, index,
#                                                vehicle_id)
#     load_var = capacity_dimension.CumulVar(index)
#     plan_output += ' {0} Load({1})\n'.format(
#         routing.IndexToNode(index), assignment.Value(load_var))
#     plan_output += 'Distance of the route: {}m\n'.format(distance)
#     plan_output += 'Load of the route: {}\n'.format(assignment.Value(load_var))
#     print(plan_output)
#     total_distance += distance
#     total_load += assignment.Value(load_var)
#   print('Total Distance of all routes: {}m'.format(total_distance))
#   print('Total Load of all routes: {}'.format(total_load))


########
# Main #
########
# def main():
#   """Entry point of the program"""
#   # Instantiate the data problem.
#   data = DataProblem()

#   # Create Routing Model
#   routing = pywrapcp.RoutingModel(data.num_locations, data.num_vehicles,
#                                   data.depot)

#   # Define weight of each edge
#   distance_evaluator = CreateDistanceEvaluator(data).distance_evaluator
#   routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator)
#   # Add Capacity constraint
#   # prize_evaluator = CreatePrizeEvaluator(data).prize_evaluator
#   # add_capacity_constraints(routing, data, prize_evaluator)

#   # Setting first solution heuristic (cheapest addition).
#   search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
#   search_parameters.first_solution_strategy = (
#       routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)  # pylint: disable=no-member
#   # Solve the problem.
#   assignment = routing.SolveWithParameters(search_parameters)
#   print_solution(data, routing, assignment)


# if __name__ == '__main__':
#   main()
  