# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:15:10 2017

@author: Sara
"""

# Edit this line to select input file!
file = '_escenarios_batteries.xlsx'

import pyomo.opt
from pyomo.environ import *
import pandas as pd
import xlwings as xw
from sys import argv

##READ THE INPUTS FROM EXCEL##

## Parsing input file ## Parse specified sheet(s) into a DataFrame. Equivalent to read_excel
xl = pd.ExcelFile(file)

dso_request=xl.parse('dso_request')
batteries = xl.parse('batteries')

# Taking the inputs from parsed data - they should be in dict
init_t = dso_request.index
init_bat = batteries.index

init_dso_down = dso_request['down'].to_dict()
init_dso_up = dso_request['up'].to_dict()

init_p_b_ch = batteries['cost_charging'].to_dict()
init_p_b_dis = batteries['cost_discharging'].to_dict()
init_q_ch = batteries['max_charging'].to_dict()
init_q_dis = batteries['max_discharging'].to_dict()
init_a_ch = batteries['efficiency_charge'].to_dict()
init_a_dis = batteries['efficiency_discharge'].to_dict()
init_o_min = batteries['initial_charge'].to_dict()
init_o_max = batteries['max_storage'].to_dict()
init_s_ch = batteries['s_ch'].to_dict()
init_s_dis = batteries['s_dis'].to_dict()

model = AbstractModel()

### SETS ###

# Sets of periods
model.t = Set(initialize=init_t)
model.t_restricted = Set(initialize=init_t[1:], within=model.t)

# Sets of batteries
model.b = Set(initialize=init_bat)


## PARAMETERS ##

# Regulation request
model.dso_down = Param(model.t, initialize=init_dso_down, within=NonNegativeReals)
model.dso_up = Param(model.t, initialize=init_dso_up, within=NonNegativeReals)

# Battery parameters
model.p_b_ch = Param(model.b, initialize=init_p_b_ch, within=NonNegativeReals)
model.p_b_dis = Param(model.b, initialize=init_p_b_dis, within=NonNegativeReals)
model.q_ch = Param(model.b, initialize=init_q_ch, within=NonNegativeReals)
model.q_dis = Param(model.b, initialize=init_q_dis, within=NonNegativeReals)
model.a_ch = Param(model.b, initialize=init_a_ch, within=NonNegativeReals)
model.a_dis = Param(model.b, initialize=init_a_dis, within=NonNegativeReals)
model.o_min = Param(model.b, initialize=init_o_min, within=NonNegativeReals) #aqui pone el inicial, no el minimo posible. Se asume pues que las baterias empiezan estando descargadas 
model.o_max = Param(model.b, initialize=init_o_max, within=NonNegativeReals)
model.s_ch = Param(model.b, initialize=init_s_ch, within=NonNegativeReals)
model.s_dis = Param(model.b, initialize=init_s_dis, within=NonNegativeReals)

### DECISION VARIABLES ###

# Batteries
model.sigma_ch = Var(model.t*model.b, within=NonNegativeReals)
model.sigma_dis = Var(model.t*model.b, within=NonNegativeReals)
model.sigma_soc = Var(model.t*model.b, within=NonNegativeReals)

### OBJECTIVE FUNCTION ###

def funcion_objetivo(model):
    battery_charging = sum(model.sigma_ch[i,j]*model.p_b_ch[j] for (i,j) in model.t * model.b)
    battery_discharging = sum(model.sigma_dis[i,j]*model.p_b_dis[j] for (i,j) in model.t * model.b)
    return battery_charging + battery_discharging
model.objective= Objective(rule=funcion_objetivo, sense=minimize)

### CONSTRAINTS ###

def regulation_request(model, t):
     total_battery_charging = sum(model.sigma_ch[i,j] for (i,j) in model.t * model.b if i == t)
     down_regulation = total_battery_charging
     
     total_battery_discharging = sum(model.sigma_dis[i,j] for (i,j) in model.t * model.b if i == t)
     up_regulation = total_battery_discharging
     if model.dso_down[t] > 0:
         return down_regulation-up_regulation >= model.dso_down[t]
     elif model.dso_up[t] > 0:
         return up_regulation-down_regulation >= model.dso_up[t]
     else:
        return Constraint.Skip
    
model.dso_constraint = Constraint(model.t, rule=regulation_request)


## CONSTRAINTS ###    
 # Battery specific  
def soc_evolution(model, t_restr, b):
    return model.sigma_soc[t_restr, b] == model.sigma_soc[t_restr-1, b] + model.a_ch[b] * model.sigma_ch[t_restr,b] - model.sigma_dis[t_restr,b]/model.a_dis[b]
model.soc_evolution = Constraint(model.t_restricted, model.b, rule=soc_evolution)

#SOC initial 
def soc_initial(model,b):
    return model.sigma_soc[0, b] == model.o_min[b] #♠aqui porque es 1 y no cero, el momento inicial
model.soc_evolution_init = Constraint(model.b, rule=soc_initial)


# Max power IN 
def bat_max_power_in(model, t, b):
    return model.sigma_ch[t,b] <= model.q_ch[b]
model.bat_max_power_in = Constraint(model.t, model.b, rule=bat_max_power_in)

# Max power OUT
def bat_max_power_out(model, t, b):
    return model.sigma_dis[t,b] <= model.q_dis[b]
model.bat_max_power_out = Constraint(model.t, model.b, rule=bat_max_power_out)

# Max battery capacity (max SOC)
def bat_max_soc(model, t, b):
    return model.sigma_soc[t,b] <= model.o_max[b]
model.bat_max_soc = Constraint(model.t, model.b, rule=bat_max_soc)

#Following constraint ensures that the energy charges to the battery b is linearly descreased
#from S_ch state of charge, typically 0,8 until zero at 100% SOC
def bat_s_ch(model, t, b):
    return model.sigma_ch[t,b] <= ((-model.q_ch[b])/(1-model.s_ch[b]))*((model.sigma_soc[t,b]/model.o_max[b])-1)
model.bat_s_ch = Constraint(model.t, model.b, rule=bat_s_ch)

#the same for discharging power sigma_out of battery unit b during period t. The lowerthreshold to limit the energy output is s_dis, tipically 0.1
def bat_s_dis(model, t, b):
    return model.sigma_dis[t,b] <= ((model.q_dis[b]/model.s_dis[b])*((model.sigma_soc[t,b])/(model.o_max[b])))
model.bat_s_dis= Constraint(model.t, model.b, rule=bat_s_dis)


### RUNNING MODEL ###
instance = model.create_instance()

#Create a Solver 
opt = pyomo.opt.SolverFactory("glpk")

instance.write('junk.lp',io_options={'symbolic_solver_labels':True})
results = opt.solve(instance, tee=True)
results.write()
print(results)

### CREATING OUTPUTS ###

bat_out_index = []
for bat in batteries.index:
    bat_out_index.append(bat + '_soc')
    bat_out_index.append(bat + '_charge_power')
    bat_out_index.append(bat + '_discharge_power')

final_batteries = pd.DataFrame(index=dso_request.index, columns=bat_out_index)

for bat in batteries.index:
    for period in dso_request.index:
        final_batteries[bat + '_soc'][period] = instance.sigma_soc[period, bat].value
        final_batteries[bat + '_charge_power'][period] = instance.sigma_ch[period, bat].value
        final_batteries[bat + '_discharge_power'][period] = instance.sigma_dis[period, bat].value

cost = pd.DataFrame([instance.objective()], index = ['Total cost'], columns = ['Total cost'])

if len(argv) > 1:
   wb = xw.Book(file)
   wb.sheets['battery_control'].clear()
   wb.sheets['battery_control'].range('A1').value = final_batteries
   wb.sheets['cost'].clear()
   wb.sheets['cost'].range('A1').value = cost
    
else:
    writer = pd.ExcelWriter(file[:-5] + '_output.xlsx')
    final_batteries.to_excel(writer,'batteries')
    cost.to_excel(writer,'cost')
    writer.save()

