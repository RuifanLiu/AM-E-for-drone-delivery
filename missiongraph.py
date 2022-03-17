# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:54:50 2022

@author: s313488
"""

import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

class missiongraph():
    def __init__(self, instance): #initialize the graph viewer

        n_loc = instance['loc'].size(0)
        pickups = instance['loc'][:n_loc//2,:]
        deliverys = instance['loc'][n_loc//2:,:]
        
        stations = instance['station']
        depot = instance['depot']
        
        
        
        nodes_pickup = {'pickup'+str(i): pickup.tolist() for i, pickup in enumerate(pickups)}
        nodes_delivery = {'delivery'+str(i): delivery.tolist() for i, delivery in enumerate(deliverys)}
        nodes_stations = {'station'+str(i): station.tolist() for i, station in enumerate(stations)}
        node_depot = {'depot': depot.tolist()}

        nodes = {}
        nodes.update(nodes_pickup)
        nodes.update(nodes_delivery)
        nodes.update(nodes_stations)
        nodes.update(node_depot)
        
        self.nodes = nodes
        
        self.G = nx.Graph()
        for v in nodes:
            self.G.add_node(v)
        
        fig = plt.figure(figsize=(8, 6))
        self.ax = plt.subplot(111)
        
        self.node_color1 = "orangered"
        self.node_color2 = "paleturquoise"
        self.node_color3 = "deepskyblue"
        self.node_color4 = '#1f78b4'
        
        self.edge_color1 = "Darkgray"
        self.edge_color2 = "red"

        nx.draw_networkx(self.G, nodes, nodelist=nodes_pickup, node_color=self.node_color1, with_labels=True)
        nx.draw_networkx(self.G, nodes, nodelist=nodes_delivery, node_color=self.node_color2, with_labels=True)
        nx.draw_networkx(self.G, nodes, nodelist=nodes_stations, node_color=self.node_color3, with_labels=True)
        nx.draw_networkx(self.G, nodes, nodelist=node_depot, node_color=self.node_color4, with_labels=True)
        
    def add_tour(self, tour):
        
        
        
        edge_list = [(list(self.nodes)[tour[n]], list(self.nodes)[tour[n+1]]) 
                     for n in range(len(tour)-1)]
        
        edge_list = [('depot',list(self.nodes)[tour[0]])] + edge_list + \
                    [(list(self.nodes)[tour[-1]], 'depot',)]
        self.G.add_edges_from(edge_list)
        
        nx.draw_networkx_edges(self.G,self.nodes,
                                edgelist=edge_list,ax=self.ax,
                                width=2,alpha=0.1,edge_color=self.edge_color2,
                                arrows = True, arrowstyle='-|>', arrowsize = 200)

    def plot(self):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        plt.savefig('figures/Taskmap'+dt_string+'.png', dpi=300)