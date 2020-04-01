# -*- coding: utf-8 -*-
"""
Created on 15. September 2019

@author: Leon Haupt, Pit Sippel
"""

""" Import"""
"""Arbitrage Market"""
file_storage = 'input_storage_parameter.xlsx'
file_market = 'input_market_data.xlsx'


import pyomo.opt
from pyomo.environ import *
import pandas as pd
import numpy as np
import gurobipy
#import matplotlib.pyplot as plt
"""SRL"""


""" Read the storage inputs from Excel """
xl = pd.ExcelFile(file_storage)
storage_system = xl.parse('storage_system')

param_rated_energy = storage_system['rated_energy'].to_dict()[0]  # [MWh]
param_rated_power = storage_system['rated_power'].to_dict()[0] # [MW]
param_efficiency_ch =storage_system['efficiency_ch'].to_dict()[0]  # [%]
param_efficiency_dis = storage_system['efficiency_dis'].to_dict()[0]  # [%]
param_specific_power_cost =storage_system['specific_power_cost'].to_dict()[0]  # [€/MW]
param_specific_energy_cost = storage_system['specific_energy_cost'].to_dict()[0]  # [€/MWh]

""" Read the market inputs from Excel """
x2 = pd.ExcelFile(file_market)

market = x2.parse('market')

param_market_price = market['data_2018'].to_dict()  # [MW]
#timestep = pd.RangeIndex(stop=len(param_market_price))
timestep = market.index
""" Read the SRL market inputs from Excel """

""" Other manual parameters """
M = 1 #PTU == 15min =>  M = 0,25; Zeitschritt ist momentan an marktdaten gekoppelt.

""" Construct Optimization Model """
model = AbstractModel() #Klasse aus Pyomo

""" SETS """
model.t = Set(initialize=timestep)
model.t_restricted = Set(initialize=timestep[1:], within=model.t)


""" PARAMETER """
# Battery
model.param_rated_energy = Param(initialize=param_rated_energy, within=NonNegativeReals)
model.param_rated_power = Param(initialize=param_rated_power, within=NonNegativeReals)
model.param_efficiency_ch = Param(initialize=param_efficiency_ch, within=NonNegativeReals)
model.param_efficiency_dis = Param(initialize=param_efficiency_dis, within=NonNegativeReals)
model.param_specific_power_cost = Param(initialize=param_specific_power_cost, within=NonNegativeReals)
model.param_specific_energy_cost = Param(initialize=param_specific_energy_cost, within=NonNegativeReals)

# Market
model.param_market_price = Param(model.t, initialize=param_market_price, within=Reals)

""" VARIABLES """
model.var_cha = Var(model.t, bounds=(0,model.param_rated_power), within=NonNegativeReals)
model.var_dis = Var(model.t, bounds=(0,model.param_rated_power), within=NonNegativeReals)
model.var_soc = Var(model.t, bounds=(0,1), initialize=0.5, within=NonNegativeReals)
model.var_act = Var(model.t, within=Boolean)

""" OBJECTIVE FUNCTION """
def objectiv_function(model):
    operating_profit = sum((model.var_dis[t] * model.param_market_price[t])  - (model.var_cha[t] * model.param_market_price[t]) for t in model.t)
    return operating_profit
model.objective = Objective(rule=objectiv_function, sense=maximize)#sense kann auch 'minimize' sein
""" CONSTRAINTS """
#Storage Constraints

# Max Energy Constraint
def constr_capacity(model, t):
    return model.var_soc[t]*model.param_rated_energy <= model.param_rated_energy
model.constr_capacity = Constraint(model.t, rule=constr_capacity)

#SOC_Evolution
def soc_evolution(model, t):
    return model.var_soc[t]  ==  model.var_soc[t-1] + (M*model.var_cha[t-1]*model.param_efficiency_ch - (M*model.var_dis[t-1])/model.param_efficiency_dis)/model.param_rated_energy
model.soc_evolution = Constraint(model.t_restricted, rule=soc_evolution)

#End Constraints
def constr_ch_end(model):
    return model.var_cha[len(model.t)-1] == 0
model.constr_ch_end = Constraint(rule=constr_ch_end)

def constr_dis_end(model):
    return model.var_dis[len(model.t)-1] == 0
model.constr_dis_end = Constraint(rule=constr_dis_end)

def constr_soc_end(model):
    return model.var_soc[len(model.t)-1] == 0.5
model.constr_soc_end = Constraint(rule=constr_soc_end)

""" RUNNING MODEL """
instance = model.create_instance()

# Create a Solver
opt = pyomo.opt.SolverFactory("gurobi")

instance.write('junk.lp', io_options={'symbolic_solver_labels': True})
results = opt.solve(instance, tee=True)
results.write()
print(results)

results_soc = [0 for x in timestep]

for t in timestep:
    results_soc[t] = instance.var_soc[t].value
plt.plot(results_soc)

n=plt.figure()
plt.plot(list(param_market_price.values()))
plt.show()
plt.show()
