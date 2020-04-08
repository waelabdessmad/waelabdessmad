# -*- coding: utf-8 -*-
"""
Created on 5. April 2020

@author: Leon Haupt, Pit Sippel

Peak Shaving

########### ToDos ##############

- Iteration over storage sizes (power and energy)
- Prozentuale Lastspitzenkappung ausrechnen
- Dynamik des Speichers abbilden 
- Investition
- Vermiedene Netzentgelte
- Zusätzliche Erlöse durch Vermarktung
- Betriebskosten im ersten Jahr
- Amortisationszeit (statisch) ohne Fremdkapital
- Kapitalrendite p.a. ohne Fremdkapital *
- Amortisationszeit mit Fremdkapital
- Jahreshöchstlast
- Jährliche Energiemenge
- Jahresbenutzungsdauer
- Stromkosten vorher und nachher

"""
""" Import"""
data_input = 'data_input.xlsx'

import pyomo.opt
from pyomo.environ import *
import pandas as pd
import numpy as np
import gurobipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


""" Read the storage inputs from Excel """
xl = pd.ExcelFile(data_input)

industry_data = xl.parse('industry_data')
init_power_load = industry_data['Load1'].to_dict() # [kW] in Viertelstunden

timestep = industry_data.index # 365*24*4 = 35040 PTUs
M = 0.25 #PTU == 15min =>  M = 0,25; Zeitschritt ist momentan an marktdaten gekoppelt.

# pv_data = xl.parse('industry_data')
# init_power_load = industry_data['Load1'].to_dict() # [MWh] 

cost_demand_charge = 100.50 # [€/kW] SWM 
cost_energy_cost = 0.16 # [€/kWh] SWM 

cost_annualized_energy = 10 # [€/kWh]
cost_annualized_power =100 # [€/kW]

param_charging_efficiency = 0.8 #to be updated
param_discharging_efficiency = 0.8 #to be updated

param_soc_init = 0.9 #to be updated
param_soc_final = 1#to be updated
param_soc_min= 0.05#to be updated
param_soc_max = 1#to be updated

param_soc_self_discharge = 0.01 # [% per day] #to be updated
# param_oversizing_factor = 1.2 # aus Smart Power abgeleitet
# param_ch_dis_ratio = 1.2 

param_ess_size_charging_power = 1200 #[kW]
param_ess_size_discharging_power = 1500 #[kW]
param_ess_size_energy = 2000 #[kWh]

""" Construct Optimization Model """
model = AbstractModel() #Klasse aus Pyomo

""" SETS """
model.t = Set(initialize=timestep)
model.t_restricted = Set(initialize=timestep[1:], within=model.t)

model.w = RangeSet(52) #1 to 52 weeks
model.h = RangeSet(168) #1 to 168 hours
model.q = RangeSet(4) #1 to 4 quarterhours

""" PARAMETER """
model.cost_demand_charge = Param(initialize=cost_demand_charge, within=NonNegativeReals)
model.cost_energy_cost = Param(initialize=cost_energy_cost, within=NonNegativeReals)

model.cost_annualized_energy = Param(initialize=cost_annualized_energy, within=NonNegativeReals)
model.cost_annualized_power = Param(initialize=cost_annualized_power, within=NonNegativeReals)

model.param_charging_efficiency = Param(initialize=param_charging_efficiency, within=NonNegativeReals)
model.param_discharging_efficiency = Param(initialize=param_discharging_efficiency, within=NonNegativeReals)

model.param_soc_init = Param(initialize=param_soc_init, within=NonNegativeReals)
model.param_soc_max = Param(initialize=param_soc_max, within=NonNegativeReals)
model.param_soc_min = Param(initialize=param_soc_min, within=NonNegativeReals)

# Industry Load Curve
model.power_industry = Param(model.t, initialize=init_power_load, within=NonNegativeReals)

model.p_peak = Param(initialize=10.02, within=Reals)#[€/kWha]
model.p_BSS = Param(initialize=p_BSS, within=Reals)#price for BSS capacity [€/KWh]
model.T = Param(initialize=1, within=NonNegativeReals)#number of annuities [a]
model.i = Param(initialize=1, within=NonNegativeReals)#discount rate [%]
model.L_tcal = Param(initialize=L_tcal,within=NonNegativeReals)#calendar lifetime [years]
model.L_tcycl = Param(initialize=L_tcycl,within=NonNegativeReals)#cycle lifetime [cycles]

""" VARIABLES """
model.ess_ch = Var(model.t, initialize=0, within=NonNegativeReals)# [kW]  --> cannot charge within first period (t=0)
model.ess_dis = Var(model.t,initialize=0, within=NonNegativeReals)# [kW]--> cannot charge within first period (t=0)
model.ess_soc = Var(model.t, bounds=(0,1),initialize=model.param_soc_init, within=NonNegativeReals) # [p.u.] State of charge 
model.ess_act = Var(model.t, within=Boolean)#activity restriction to avoud simultanouesly charging and dischargig

model.grid_ch = Var(model.t, within=NonNegativeReals)# [kW] electricity drawn from the grid
model.grid_dis = Var(model.t, within=NonNegativeReals)# [kW] electricty fed into the grid
model.grid_max = Var(within=NonNegativeReals)# max value

model.C_peak = Var(within=NonNegativeReals)# cost for power capacity [€]
model.A_BSS = Var(within=NonNegativeReals) # annuity payment for BSS [€]
model.P_peak = Var(within=NonNegativeReals)# peak power from grid per year [MW]
model.cap_ASS_aged = Var(within=NonNegativeReals)#invested capacity of BSS with battery degradation [kWh]
model.cap_BSS = Var(within=NonNegativeReals)#installed capacity of BSS [kWh]
model.cap_add_l = Var(within=NonNegativeReals)#additional capacity invest due to storage level degradation [kWh]
model.cap_red_cycl = Var(within=NonNegativeReals)#reduced capacity invest due to unexploited cycle life [kWh]
model.l_BSS = Var(model.q*model.h*model.w,within=NonNegativeReals)#storage level of BSS [kWh]
model.x_BSS_tot = Var(within=Reals)#energy in-flow in BSS per year [kWh]
model.x_in =  Var(model.q*model.h,within=Reals)
model.P_PCR = Var(model.w,within=NonNegativeReals) 

""" OBJECTIVE FUNCTION """
def objectiv_function(model):
    operating_cost = model.C_peak + model.A_BSS    
    return operating_cost
model.objective = Objective(rule=objectiv_function, sense=minimize) #minimize cost 


""" CONSTRAINTS """
def Peak (model):
    return model.C_peak == model.P_peak * model.p_peak
model.peak = Constraint (model, rule=Peak)

def Annuity(model):
    return model.A_BSS == model.cap_ASS_aged * model.p_BSS * ((1/i)-1/i*((1+i)**T))
model.annuity = Constraint(model,rule=Annuity)
 
def aged_cap (model):
   return model.cap_ASS_aged == (model.cap_BSS/model.Eol) + model.cap_add_l - model.cap_red_cycl#declaration of Eol?
model.aged = Constraint(model,rule=aged_cap)

def add_cap(model):
   return model.cap_add_l == (1/3) * (sum(model.l_BSS[h][w][q] * (1/model.N_int) for (h,w,q) in model.h*model.w*model.q))#declaration of Ninterval?
model.add = Constraint(model,rule=add_cap)

def redu_cap(model):
   return model.cap_red_cycl == (1/3) * (model.cap_BSS-(model.x_BSS_tot * (model.L_tcal/model.L_tcycl)))
model.redu = Constraint(model,rule=redu_cap)
    
def ener_flow(model):
   return model.x_BSS_tot == sum(sum(model.x_in [h] [q] for (h,q) in model.h*model.q) + model.P_PCR [w] * model.E_PCR,mean for w in model.w)#Epcr,mean declaration?
model.ener = Constraint(model,rule=ener_flow)
def (model):
   return



    
# Energy Balance
def constraint_0(model, t):
    return 0 == model.power_industry[t] - model.grid_ch[t] - model.grid_dis[t] - model.ess_dis[t] + model.ess_ch[t]
model.constraint_0 = Constraint(model.t, rule=constraint_0)

#Demand Charge
def constraint_1(model, t):
    return model.grid_max >= model.grid_ch[t] #+ model.grid_dis[t]
model.constraint_1 = Constraint(model.t, rule=constraint_1)

#No Discharging into the grid
def constraint_2(model, t):
    return model.grid_dis[t] == 0 
model.constraint_2 = Constraint(model.t, rule=constraint_2)

# Storage Charging and Discharging Activity 
def constraint_3(model, t):
    return  model.ess_ch[t] <= (1-model.ess_act[t])* param_ess_size_charging_power
model.constraint_3 = Constraint(model.t, rule=constraint_3)
def constraint_4(model, t):
    return  model.ess_dis[t] <= (model.ess_act[t])* param_ess_size_discharging_power
model.constraint_4 = Constraint(model.t, rule=constraint_4)



#SoC Evolution
def constraint_6(model,t):
    return  model.ess_soc[t] == model.ess_soc[t-1] + ((M*model.ess_ch[t]*model.param_charging_efficiency - M*model.ess_dis[t]/model.param_discharging_efficiency)/param_ess_size_energy)
model.constraint_6 = Constraint(model.t_restricted, rule=constraint_6)

#SoC Evolution - Finale state 
def constraint_7(model):
    return  model.ess_soc[len(model.t)-1]  == param_soc_final
model.constraint_7 = Constraint( rule=constraint_7)





""" RUNNING MODEL """
instance = model.create_instance()

# Create a Solver
opt = pyomo.opt.SolverFactory("gurobi")

instance.write('junk.lp', io_options={'symbolic_solver_labels': True})
results = opt.solve(instance, tee=True)
results.write()
#print(results)


""" PLOT """

grid_power = [0 for x in timestep]
soc = [0 for x in timestep]
load_curv_list= [0 for x in timestep]

for t in timestep:
    grid_power[t] = instance.grid_ch[t].value
    soc[t] = instance.ess_soc[t].value
    load_curv_list[t] = init_power_load[t]
    
plt.plot(grid_power)
plt.plot(list(init_power_load.values()))
plt.show()


fig2 = plt.figure()
plt.plot(soc)
plt.show()

# peaks = [0 for x in timestep]

# for t in timestep:
#     if(init_power_load[t]-grid_power[t]>0):
#         peaks[t] = init_power_load[t]
#     else:
#         peaks[t] = 0
    


new_grid_power_3d = np.reshape(grid_power, (365, 96)).T
new_init_power_load_3d = np.reshape(load_curv_list, (365, 96)).T
# new_peak_3d = np.reshape(peaks, (365, 96)).T


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
Y = np.arange(0,96,1)
X= np.arange(0, 365,1)
X, Y = np.meshgrid(X, Y)
Z = new_grid_power_3d
Z1 = new_init_power_load_3d

# Plot the surface.
surf = ax.plot_surface(X, Y, Z,
                       linewidth=0, antialiased=True)
surf = ax.plot_surface(X, Y, Z1,
                       linewidth=0, antialiased=True)

# Customize the z axis.
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.

plt.show()

