# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:03:06 2021

@author: s313488
"""

from gurobipy import *
import numpy as np
from scipy.spatial import distance_matrix

def solve_epdp_gurobi(loc, load, station, depot, battery_margin=0, threads=1, timeout=None, gap=None):
    M = 1e5
    E_max = 1
    e_max = 1-battery_margin
    e_min = 0
    v_e = 1.0
    
    
    # DIS2SOC = 10/60*(0.7567)/0.7333
    DIS2SOC = 0.179
    
    station = np.array(station)
    loc = np.array(loc)
    depot = np.array(depot)
    
    n_station = station.shape[0]
    n_node = loc.shape[0] + n_station + 1
    req_count = int((n_node-n_station-1)/2)
    veh_count = 1
    n_depot = 1
    
    loc_all = np.concatenate((loc, station, depot[None,:]), 0)

    dis_matrix = distance_matrix(loc_all,loc_all)
    E_matrix = dis_matrix*DIS2SOC
    
    station_copynum = int(req_count/10)-1
    # Treat depot as an additional station
    n_station = n_station + 1
    
    # extend copies of station to allow multiple visiting of stations
    for _ in range(station_copynum):
        E_matrix = np.column_stack((E_matrix,E_matrix[:,-n_station:]))
        E_matrix = np.row_stack((E_matrix,E_matrix[-n_station:,:]))
        # tao = np.column_stack((tao,tao[:,-n_station:]))
        # tao = np.row_stack((tao,tao[-n_station:,:]))
    E_matrix = np.column_stack((E_matrix,E_matrix[:,-1:]))
    E_matrix = np.row_stack((E_matrix,E_matrix[-1:,:]))
    E_matrix = np.column_stack((E_matrix,E_matrix[:,-1:]))
    E_matrix = np.row_stack((E_matrix,E_matrix[-1:,:]))
    n_station = int((station_copynum+1)*n_station)
    tao = E_matrix*100
    
    n_task =  req_count*2
    n_node = req_count*2 + n_station + n_depot*2
    
    m=Model("Vehiclerouting")
    x=m.addVars(n_node,n_node,vtype=GRB.BINARY, name='x')
    t=m.addVars(n_node,vtype=GRB.CONTINUOUS, name='t')
    e=m.addVars(n_node,vtype=GRB.CONTINUOUS,name='e')
    
    for i in range(n_node):
        m.addConstr(x[i,i],GRB.EQUAL,0)
        
    ind1 = []
    ind2 = []
    for i in range(n_node):
        ind1.append((i,n_node-2))
        ind2.append((n_node-1,i))
    con1_1=quicksum(x[i] for i in ind1)
    con1_2=quicksum(x[i] for i in ind2)
    m.addConstr(con1_1,GRB.EQUAL,0)
    m.addConstr(con1_2,GRB.EQUAL,0) # constraint ---1
    
    for i in range(n_task):
        ind=[]
        for j in range(n_node):
            ind.append((i,j))
        con2_1=quicksum(x[i] for i in ind)
        m.addConstr(con2_1,GRB.EQUAL,1)      #constraint--2
    
    ind=[]
    for j in range(n_task):
        ind.append((n_node-2,j))
    con2_2=quicksum(x[i] for i in ind)
    m.addConstr(con2_2,GRB.EQUAL,1)      #constraint--2
    
    for i in range(n_task):
        ind=[]
        for j in range(n_node):
            ind.append((j,i))
        con3_1=quicksum(x[i] for i in ind)
        m.addConstr(con3_1,GRB.EQUAL,1)      #constraint--3
    
    ind=[]
    for j in range(n_task):
        ind.append((j,n_node-1))
    con3_2=quicksum(x[i] for i in ind)
    m.addConstr(con3_2,GRB.EQUAL,1)      #constraint--3
    
    
    for i in range(n_task,n_node-2):
        ind1=[]
        ind2=[]
        for j in range(n_node):
            ind1.append((i,j))
            ind2.append((j,i))
        con4=quicksum(x[i] for i in ind1)-quicksum(x[i] for i in ind2)
        m.addConstr(con4,GRB.EQUAL,0)                       #constraint----4
        
    for i in range(0,req_count):
        for j in range(n_task,n_node-2):
            con5_1 = (x[j,i+req_count]-1) + M*(1-x[i,j])
            con5_2 = (x[j,i+req_count]-1) - M*(1-x[i,j])
            m.addConstr(con5_1,GRB.GREATER_EQUAL,0) 
            m.addConstr(con5_2,GRB.LESS_EQUAL,0)        #constraint----5
    
    # for i in range(0,n_task,2):
    #     ind1=[]
    #     for j in range(n_task+n_depot+1,n_node):
    #         ind1.append((i,j))
    #     exp = quicksum(x[i] for i in ind1)
    #     con6_1 = (x[i,i+1]-1) + M*(1-exp)
    #     con6_2 = (x[i,i+1]-1) - M*(1-exp)
    #     m.addConstr(con6_1,GRB.GREATER_EQUAL,0) 
    #     m.addConstr(con6_2,GRB.LESS_EQUAL,0)        #constraint----6
        
    for i in range(0,req_count):
        ind1=[]
        for j in range(n_task, n_node-2):
            ind1.append((i,j))
        exp = quicksum(x[i] for i in ind1)
        con6_1 = (exp-1) + M*(x[i,i+req_count])
        con6_2 = (exp-1) - M*(x[i,i+req_count])
        m.addConstr(con6_1,GRB.GREATER_EQUAL,0) 
        m.addConstr(con6_2,GRB.LESS_EQUAL,0)        #constraint----6
        
    for i in range(n_node):
        for j in range(n_task):
            exp = e[i]-e[j]-E_matrix[i,j]/E_max
            con7_1 = exp + M*(1-x[i,j])
            con7_2 = exp - M*(1-x[i,j])
            m.addConstr(con7_1,GRB.GREATER_EQUAL,0) 
            m.addConstr(con7_2,GRB.LESS_EQUAL,0)        #constraint----7
            
    for i in range(n_task, n_node-2):
        con8_1 = e[i]-e_max
        m.addConstr(con8_1,GRB.EQUAL,0)        #constraint----8
    con8_2 = e[n_node-2]-e_max
    m.addConstr(con8_2,GRB.EQUAL,0) 
    
    for i in range(n_node):
        con9_1 = e[i]-e_min
        con9_2 = e[i]-e_max
        m.addConstr(con9_1,GRB.GREATER_EQUAL,0) 
        m.addConstr(con9_2,GRB.LESS_EQUAL,0)        #constraint----9
    
    con = t[n_node-2]
    m.addConstr(con,GRB.EQUAL,0)        #constraint----9
    
    for i in range(n_node):
        for j in range(n_node):
            exp = t[i]+tao[i,j]-t[j]
            con10_1 = exp + M*(1-x[i,j])
            con10_2 = exp - M*(1-x[i,j])
            m.addConstr(con10_1,GRB.GREATER_EQUAL,0) 
            m.addConstr(con10_2,GRB.LESS_EQUAL,0)        #constraint----10
                
    # for i in range(n_node):
    #     for j in range(n_task,n_node-2):
    #         exp = t[i]+tao[i,j]-t[j]+(e_max-e[i]-E_matrix[i,j]/E_max)/v_e
    #         con11_1 = exp + M*(1-x[i,j])
    #         con11_2 = exp - M*(1-x[i,j])
    #         m.addConstr(con11_1,GRB.GREATER_EQUAL,0) 
    #         m.addConstr(con11_2,GRB.LESS_EQUAL,0)        #constraint----11
    
    minexp=quicksum(E_matrix[i,j]*x[i,j] for i in range(n_node) for j in range(n_node))
    m.setObjective(minexp,GRB.MINIMIZE)
    #m.Params.OutputFlag = 0
    if timeout:
        m.Params.timeLimit = timeout
    else:
        m.Params.timeLimit = 100.0
    # if gap:
    #     m.Params.mipGap = gap * 0.01  # Percentage
    #     m.Params.mipGap = 1e-5
    # m.Params.tolerance = 1e-4
    
    m.optimize()
    m.write('mymodel.lp')
              
    print('Obj: %g' % m.objVal)
    # print(m.getAttr("X", m.getVars()))
    
    output = m.getVars()
    output_x = np.zeros([n_node,n_node])
    output_e = np.zeros(n_node)
    output_t = np.zeros(n_node)
    row, col = 0, 0
    ind_e = 0
    ind_t = 0
    for v  in m.getVars():
        if v.varName[0] == 'x':
            output_x[row,col] = v.x
            col += 1
            if col==n_node:
                row += 1
                col = 0
        elif v.varName[0] == 'e':
            output_e[ind_e] = v.x
            ind_e += 1
        elif v.varName[0] == 't':
            output_t[ind_t] = v.x
            ind_t += 1
    
    
    tour_with_statcopy, tour, outpur_e_reorder = solution2path(output_x, output_e, n_task, n_station, n_depot, station_copynum)
    
    
    return m.objVal, [tour] 
        
def solution2path(sol, output_e, n_task, n_station, n_depot, station_copynum):
    
    path = []
    next_node = n_task + n_station + n_depot - 1
    for ii in range(int(round(sum(sum(sol))))):
        next_node = np.where(sol[next_node,:]>0.9)[0].tolist()[0]
        path.append(next_node)
        
        
    print('Tour: {}'.format(path))
    
    path_adjust = np.copy(path)
    for _ in range(station_copynum):
        for ii in range(len(path_adjust)):
            if path_adjust[ii] >= n_task+n_station/(station_copynum+1):
                path_adjust[ii] -= n_station/(station_copynum+1)
            
    for ii in range(len(path_adjust)):
        if path_adjust[ii]>=n_task+n_station/(station_copynum+1):
            path_adjust[ii] -= 1
    for ii in range(len(path_adjust)):
        if path_adjust[ii]>=n_task+n_station/(station_copynum+1):
            path_adjust[ii] -= 1
            
                    
    output_e_reorder = []
    for v in path:
        output_e_reorder.append(output_e[v])
    
    return path, path_adjust, output_e_reorder