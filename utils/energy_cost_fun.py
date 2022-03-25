# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:22:49 2022

@author: s313488
"""
import torch

def energy_cost_fun(locs, wind_mag=0.0, wind_dir=0.0, command='airspeed', command_speed=15):
    
    loc1, loc2 = locs
    
    relative_loc = loc2 - loc1
    
    try:
        relative_loc = torch.cuda.FloatTensor(relative_loc)
        wind_mag = torch.clamp(torch.cuda.FloatTensor(wind_mag), 0, 10) 
        wind_dir = torch.cuda.FloatTensor(wind_dir) 
    except:
        relative_loc = torch.FloatTensor(relative_loc)
        wind_mag = torch.clamp(torch.FloatTensor(wind_mag), 0, 10) 
        wind_dir = torch.FloatTensor(wind_dir) 
        
    batch_size = wind_mag.shape[0]
    shp_e = relative_loc.shape
    
    relative_loc = relative_loc.view(batch_size, -1, 2)
    
    windSpeed = wind_mag # batch_size*1
    windAngle = wind_dir # batch_size*1
    
    if command == 'airspeed':# costant air speed command
        airSpeed = command_speed
        
        courseAngle = torch.atan2(relative_loc[...,1], relative_loc[...,0]) # batch_size*size1*size2
        distance = relative_loc.norm(p=2, dim=-1) # batch_size*size1*size2

        # airAngle = courseAngle - torch.asin(torch.mul(torch.div(windSpeed/airSpeed), torch.sin(windAngle-courseAngle)))
        # groundSpeed = torch.sqrt(torch.square(airSpeed) + torch.square(windSpeed) + \
        #                          2*torch.mul(torch.mul(airSpeed,windSpeed),torch.cos(airAngle-windAngle)))
        airAngle = courseAngle - torch.asin(torch.mul(windSpeed/airSpeed, torch.sin(windAngle-courseAngle)))
        groundSpeed = torch.sqrt(airSpeed*airSpeed + torch.square(windSpeed) + \
                                 2*torch.mul(airSpeed*windSpeed,torch.cos(airAngle-windAngle)))
        t_flight = torch.div(distance, groundSpeed)
        ### need to work on
        # power = power_fun(airspeed)
        power = 1495.5436006 
        
        t_flight = 10*1000*t_flight
        energy_cost = power*t_flight/(1.5*1000*3600)
        
        #1933.2*10*1000*Dis/(20*Vel)/(1.5*1000*3600)
        # print(energy_cost.squeeze().shape)
    else:
        
        assert 0, 'Only support constant airspeed command yet'
        
        
    return energy_cost.view(shp_e[:-1])


