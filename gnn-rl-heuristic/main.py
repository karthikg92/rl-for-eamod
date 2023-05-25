from __future__ import print_function
import argparse
import os
import gurobipy as gp
from tqdm import trange
import numpy as np
from src.algos.pax_flows_solver import PaxFlowsSolver
from src.algos.reb_flows_solver import RebalFlowSolver
import torch
import json
import os
import wandb
import pickle
import time

from src.envs.amod_env import Scenario, AMoD
from src.algos.a2c_gnn import A2C
# from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum

def create_scenario(json_file_path, energy_file_path, seed=10):
    f = open(json_file_path)
    energy_dist = np.load(energy_file_path)
    data = json.load(f)
    tripAttr = data['demand']
    reb_time = data['rebTime']
    total_acc = data['totalAcc']
    # additional_vehicles_peak_demand = data['additionalVehiclesPeakDemand']
    spatial_nodes = data['spatialNodes']
    tf = data['episodeLength']
    number_charge_levels = data['chargelevels']
    charge_levels_per_charge_step = data['chargeLevelsPerChargeStep']
    chargers = data['chargeLocations']
    cars_per_station_capacity = data['carsPerStationCapacity']
    p_energy = data["energy_prices"]
    time_granularity = data["timeGranularity"]
    operational_cost_per_timestep = data['operationalCostPerTimestep']
    try:
        peak_hours = data['peakHours']
    except:
        peak_hours = []

    scenario = Scenario(spatial_nodes=spatial_nodes, charging_stations=chargers, cars_per_station_capacity = cars_per_station_capacity, number_charge_levels=number_charge_levels, charge_levels_per_charge_step=charge_levels_per_charge_step, peak_hours=peak_hours, 
                        energy_distance=energy_dist, tf=tf, sd=seed, tripAttr = tripAttr, demand_ratio=1, reb_time=reb_time, total_acc = total_acc, p_energy=p_energy, time_granularity=time_granularity, operational_cost_per_timestep=operational_cost_per_timestep)
    return scenario

parser = argparse.ArgumentParser(description='A2C-GNN')

# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')

# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--equal_distr_baseline', type=bool, default=False,
                    help='activates the equal distribution baseline.')
parser.add_argument('--toy', type=bool, default=False,
                    help='activates toy mode for agent evaluation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=16000, metavar='N',
                    help='number of episodes to train agent (default: 16k)')
parser.add_argument('--T', type=int, default=10, metavar='N',
                    help='Time horizon for the A2C')
parser.add_argument('--lr_a', type=float, default=1e-3, metavar='N',
                    help='Learning rate for the actor')
parser.add_argument('--lr_c', type=float, default=1e-3, metavar='N',
                    help='Learning rate for the critic')
parser.add_argument('--grad_norm_clip_a', type=float, default=0.5, metavar='N',
                    help='Gradient norm clipping for the actor')
parser.add_argument('--grad_norm_clip_c', type=float, default=0.5, metavar='N',
                    help='Gradient norm clipping for the critic')
parser.add_argument('--charging_heuristic', type=str, default='empty_to_full',
                    help='Which charging heuristic to use')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
lr_a = args.lr_a
lr_c = args.lr_c
grad_norm_clip_a = args.grad_norm_clip_a
grad_norm_clip_c = args.grad_norm_clip_c
charging_heuristic = args.charging_heuristic
use_equal_distr_baseline = args.equal_distr_baseline
seed = args.seed
test = args.test
T = args.T

# toy 1x1
if args.toy:
    problem_folder = 'Toy'
    file_path = os.path.join('data', problem_folder, 'scenario_test_6_1x2_flip.json')
    experiment = 'training_' + problem_folder+ '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T) + file_path + '_heuristic_' + charging_heuristic + "_fast_charging"
    energy_dist_path = os.path.join('data', problem_folder,  'energy_distance_1x2.npy')
    scenario = create_scenario(file_path, energy_dist_path)
    env = AMoD(scenario)
    scale_factor = 0.01
    scale_price = 0.1
    model = A2C(env=env, T=T, lr_a=lr_a, lr_c=lr_c, grad_norm_clip_a=grad_norm_clip_a, grad_norm_clip_c=grad_norm_clip_c, seed=seed, scale_factor=scale_factor, scale_price=scale_price).to(device)
    model.load_checkpoint(path=f'saved_files/ckpt/{problem_folder}/a2c_gnn_final.pth')
    tf = env.tf
else:
    # problem_folder = 'NY/ClusterDataset1'
    # file_path = os.path.join('data', problem_folder,  'd1.json')
    problem_folder = 'NY_5'
    file_path = os.path.join('data', problem_folder,  'NY_5.json')
    # problem_folder = 'SF_5_clustered'
    # file_path = os.path.join('data', problem_folder,  'SF_5_short.json')
    experiment = 'training_' + file_path + '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T) + '_heuristic_' + charging_heuristic
    energy_dist_path = os.path.join('data', problem_folder, 'energy_distance.npy')
    scenario = create_scenario(file_path, energy_dist_path)
    env = AMoD(scenario)
    # Initialize A2C-GNN
    # NY
    # scale_factor = 0.01
    # scale_price = 0.1
    # SF
    # scale_factor = 0.0001 # potentially needs to be reschaled for NY5
    # scale_price = 0.1
    # NY 5 
    scale_factor = 0.0001
    scale_price = 0.1
    model = A2C(env=env, T=T, lr_a=lr_a, lr_c=lr_c, grad_norm_clip_a=grad_norm_clip_a, grad_norm_clip_c=grad_norm_clip_c, seed=seed, scale_factor=scale_factor, scale_price=scale_price).to(device)
    # model.load_checkpoint(path=f'saved_files/ckpt/{problem_folder}/a2c_gnn_final.pth')
    tf = env.tf
if use_equal_distr_baseline:
    experiment = 'uniform_distr_baseline_' + file_path + '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T) + '_heuristic_' + charging_heuristic
if test:
    experiment += "_test_evaluation"

# set Gurobi environment mine
# gurobi_env = gp.Env(empty=True)
# gurobi = "Dominik"
# gurobi_env.setParam('WLSACCESSID', '8cad5801-28d8-4e2e-909e-3a7144c12eb5')
# gurobi_env.setParam('WLSSECRET', 'a25b880b-8262-492f-a2e5-e36d6d78cc98')
# gurobi_env.setParam('LICENSEID', 799876)
# gurobi_env.setParam("OutputFlag",0)
# gurobi_env.start()

# set Gurobi environment Justin
# gurobi_env = gp.Env(empty=True)
# gurobi = "Justin"
# gurobi_env.setParam('WLSACCESSID', '82115472-a780-40e8-9297-b9c92969b6d4')
# gurobi_env.setParam('WLSSECRET', '0c069810-f45f-4920-a6cf-3f174425e641')
# gurobi_env.setParam('LICENSEID', 844698)
# gurobi_env.setParam("OutputFlag",0)
# gurobi_env.start()

# set Gurobi environment Karthik
gurobi_env = gp.Env(empty=True)
gurobi = "Karthik"
gurobi_env.setParam('WLSACCESSID', 'ad632625-ffd3-460a-92a0-6fef5415c40d')
gurobi_env.setParam('WLSSECRET', '60bd07d8-4295-4206-96e2-bb0a99b01c2f')
gurobi_env.setParam('LICENSEID', 849913)
gurobi_env.setParam("OutputFlag",0)
gurobi_env.start()

# set Gurobi environment Karthik2
# gurobi_env = gp.Env(empty=True)
# gurobi = "Karthik2"
# gurobi_env.setParam('WLSACCESSID', 'bc0f99a5-8537-45c3-89d9-53368d17e080')
# gurobi_env.setParam('WLSSECRET', '6dddd313-d8d4-4647-98ab-d6df872c6eaa')
# gurobi_env.setParam('LICENSEID', 799870)
# gurobi_env.setParam("OutputFlag",0)
# gurobi_env.start()

# set up wandb
wandb.init(
      # Set the project where this run will be logged
      project='eamod-optimization', 
      entity='karthikg',
      # pass a run name 
      name=experiment, 
      # Track hyperparameters and run metadata
      config={
        "number_chargelevels": env.scenario.number_charge_levels,
        "number_spatial_nodes": env.scenario.spatial_nodes,
        "dataset": file_path,
        "episodes": args.max_episodes,
        "number_vehicles_per_node_init": env.G.nodes[(0,env.scenario.number_charge_levels-1)]['accInit'],
        "charging_stations": list(env.scenario.charging_stations),
        "charging_station_capacities": list(env.scenario.cars_per_station_capacity),
        "learning_rate_actor": lr_a,
        "learning_rate_critic": lr_c,
        "gradient_norm_clip_actor": grad_norm_clip_a,
        "gradient_norm_clip_critic": grad_norm_clip_c,
        "scale_factor": scale_factor,
        "scale_price": scale_price,
        "time_horizon": T,
        "episode_length": env.tf,
        "seed": seed,
        "charging_heuristic": charging_heuristic,
        "charge_levels_per_timestep": env.scenario.charge_levels_per_charge_step, 
        "licence": gurobi,
      })


################################################
#############Training and Eval Loop#############
################################################

#Initialize lists for logging
n_episodes = args.max_episodes #set max number of training episodes
T = tf #set episode length
epochs = trange(n_episodes) #epoch iterator
best_reward = -10000

if test:
    rewards_np = np.zeros(n_episodes)
    served_demands_np = np.zeros(n_episodes)
    charging_costs_np = np.zeros(n_episodes)
    rebal_costs_np = np.zeros(n_episodes)
    epoch_times = np.zeros(n_episodes)
else:
    model.train() #set model in train mode
total_demand_per_spatial_node = np.zeros(env.number_nodes_spatial)
for region in env.nodes_spatial:
    for destination in env.nodes_spatial:
        for t in range(env.tf):
            total_demand_per_spatial_node[region] += env.demand[region,destination][t]

for i_episode in epochs:
    desired_accumulations_spatial_nodes = np.zeros(env.scenario.spatial_nodes)
    bool_random_random_demand = not test # only use random demand during training
    obs = env.reset(bool_random_random_demand) #initialize environment
    episode_reward = 0
    episode_served_demand = 0
    episode_rebalancing_cost = 0
    episode_charge_rebalancing_cost = 0
    episode_spatial_rebalancing_cost = 0
    time_start = time.time()
    for step in range(T):
        # take matching step (Step 1 in paper)
        if step == 0 and i_episode == 0:
            # initialize optimization problem in the first step
            pax_flows_solver = PaxFlowsSolver(env=env,gurobi_env=gurobi_env)
        else:
            pax_flows_solver.update_constraints()
            pax_flows_solver.update_objective()
        _, paxreward, done, info_pax = env.pax_step(pax_flows_solver=pax_flows_solver)
        episode_reward += paxreward
        charging_heuristic_reward = env.apply_charging_heurstic(charging_heuristic)
        episode_charge_rebalancing_cost -= charging_heuristic_reward
        episode_reward += charging_heuristic_reward
        episode_rebalancing_cost -= charging_heuristic_reward
        # use GNN-RL policy (Step 2 in paper)
        if use_equal_distr_baseline:
            action_rl = model.select_equal_action() # selects uniform distr.
            a_loss = 0
            v_loss = 0
            mean_value = 0
            mean_concentration = 0 
            mean_std = 0
            mean_log_prob = 0
            std_log_prob = 0
        else:
            action_rl = model.select_action()
        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
        # we only use idle vehicles that can go anywhere
        total_idle_acc = 0
        for n in env.nodes:
            if n[1]>=env.scenario.max_energy_distance:
                total_idle_acc += env.acc[n][env.time+1]
        desired_acc_spatial = {env.nodes_spatial[i]: int(action_rl[i] *total_idle_acc) for i in env.nodes_spatial} # over spatial nodes
        total_desiredAcc = sum(desired_acc_spatial[n] for n in env.nodes_spatial)
        missing_cars = total_idle_acc - total_desiredAcc
        most_likely_node = np.argmax(action_rl)
        if missing_cars != 0:
            desired_acc_spatial[env.nodes_spatial[most_likely_node]] += missing_cars   
            total_desiredAcc = sum(desired_acc_spatial[n] for n in env.nodes_spatial)
        assert abs(total_desiredAcc - total_idle_acc) < 0.0001
        for n in env.nodes_spatial:
            assert desired_acc_spatial[n] >= 0
        for n in env.nodes_spatial:
            desired_accumulations_spatial_nodes[n] += desired_acc_spatial[n]
        # solve minimum rebalancing distance problem (Step 3 in paper)
        if step == 0 and i_episode == 0:
            # initialize optimization problem in the first step
            rebal_flow_solver = RebalFlowSolver(env=env, desired_acc_spatial=desired_acc_spatial,gurobi_env=gurobi_env)
        else:
            rebal_flow_solver.update_constraints(desired_acc_spatial, env)
            rebal_flow_solver.update_objective(env)
        rebAction = rebal_flow_solver.optimize()

        # Take action in environment
        new_obs, rebreward, done, info_reb = env.reb_step(rebAction)
        episode_reward += rebreward
        episode_spatial_rebalancing_cost -= rebreward
        # Store the transition in memory
        model.rewards.append(paxreward + charging_heuristic_reward + rebreward)
        # track performance over episode
        episode_served_demand += info_pax['served_demand']
        episode_rebalancing_cost += info_reb['rebalancing_cost']
        # stop episode if terminating conditions are met
        if done:
            break
    # perform on-policy backprop
    if not use_equal_distr_baseline and not test:
        a_loss, v_loss, mean_value, mean_concentration, mean_std, mean_log_prob, std_log_prob = model.training_step()

    # Send current statistics to screen was episode_reward, episode_served_demand, episode_rebalancing_cost
    epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}")
    # Send current statistics to wandb
    for spatial_node in range(env.scenario.spatial_nodes):
        wandb.log({"Episode": i_episode+1, f"Desired Accumulation {spatial_node}": desired_accumulations_spatial_nodes[spatial_node]})
        wandb.log({"Episode": i_episode+1, f"Total Demand {spatial_node}": total_demand_per_spatial_node[spatial_node]})
        if total_demand_per_spatial_node[spatial_node] > 0:
            wandb.log({"Episode": i_episode+1, f"Desired Acc. to Total Demand ratio {spatial_node}": desired_accumulations_spatial_nodes[spatial_node]/total_demand_per_spatial_node[spatial_node]})
    
    
    if i_episode == 5:
        with open(f"./{args.directory}/ckpt/{problem_folder}/acc.p", "wb") as file:
            pickle.dump(env.acc, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/acc.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial.p", "wb") as file:
            pickle.dump(env.acc_spatial, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/new_charging_vehicles.p", "wb") as file:
            pickle.dump(env.new_charging_vehicles, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/new_charging_vehicles.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/new_rebalancing_vehicles.p", "wb") as file:
            pickle.dump(env.new_rebalancing_vehicles, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/new_rebalancing_vehicles.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/n_customer_vehicles_spatial.p", "wb") as file:
            pickle.dump(env.n_customer_vehicles_spatial, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/n_customer_vehicles_spatial.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/satisfied_demand.p", "wb") as file:
            pickle.dump(env.satisfied_demand, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/satisfied_demand.p")


    # Checkpoint best performing model
    if episode_reward > best_reward:
        print("Saving best model.")
        model.save_checkpoint(path=f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn.pth")
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn.pth")
        with open(f"./{args.directory}/ckpt/{problem_folder}/acc.p", "wb") as file:
            pickle.dump(env.acc, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/acc.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial.p", "wb") as file:
            pickle.dump(env.acc_spatial, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/new_charging_vehicles.p", "wb") as file:
            pickle.dump(env.new_charging_vehicles, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/new_charging_vehicles.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/new_rebalancing_vehicles.p", "wb") as file:
            pickle.dump(env.new_rebalancing_vehicles, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/new_rebalancing_vehicles.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/n_customer_vehicles_spatial.p", "wb") as file:
            pickle.dump(env.n_customer_vehicles_spatial, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/n_customer_vehicles_spatial.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/satisfied_demand.p", "wb") as file:
            pickle.dump(env.satisfied_demand, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/satisfied_demand.p")
        best_reward = episode_reward
        best_rebal_cost = episode_rebalancing_cost
        best_served_demand  = episode_served_demand
        best_spatial_rebal_cost = episode_spatial_rebalancing_cost
        best_charge_rebal_cost = episode_charge_rebalancing_cost
    if test:
        rewards_np[i_episode] = episode_reward
        served_demands_np[i_episode] = episode_served_demand
        charging_costs_np[i_episode] = episode_charge_rebalancing_cost
        rebal_costs_np[i_episode] = episode_rebalancing_cost
        epoch_times[i_episode] = time.time()-time_start
    else:
        wandb.log({"Episode": i_episode+1, "Reward": episode_reward, "Best Reward:": best_reward, "ServedDemand": episode_served_demand, "Best Served Demand": best_served_demand, 
        "Reb. Cost": episode_rebalancing_cost, "Best Reb. Cost": best_rebal_cost, "Charge Reb. Cost": episode_charge_rebalancing_cost, "Best Charge Reb. Cost": best_charge_rebal_cost, "Spatial Reb. Cost": -rebreward, "Best Spatial Reb. Cost": best_spatial_rebal_cost,
        "Actor Loss": a_loss, "Value Loss": v_loss, "Mean Value": mean_value, "Mean Concentration": mean_concentration, "Mean Std": mean_std, "Mean Log Prob": mean_log_prob, "Std Log Prob": std_log_prob})
        # regularly safe model
        if i_episode % 5000 == 0:
            model.save_checkpoint(path=f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn_{i_episode}.pth")
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn_{i_episode}.pth")
            with open(f"./{args.directory}/ckpt/{problem_folder}/acc_{i_episode}.p", "wb") as file:
                pickle.dump(env.acc, file)
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/acc_{i_episode}.p")
            with open(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial_{i_episode}.p", "wb") as file:
                pickle.dump(env.acc_spatial, file)
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial_{i_episode}.p")
            with open(f"./{args.directory}/ckpt/{problem_folder}/new_charging_vehicles{i_episode}.p", "wb") as file:
                pickle.dump(env.new_charging_vehicles, file)
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/new_charging_vehicles{i_episode}.p")
            with open(f"./{args.directory}/ckpt/{problem_folder}/new_rebalancing_vehicles{i_episode}.p", "wb") as file:
                pickle.dump(env.new_rebalancing_vehicles, file)
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/new_rebalancing_vehicles{i_episode}.p")
            with open(f"./{args.directory}/ckpt/{problem_folder}/satisfied_demand.p", "wb") as file:
                pickle.dump(env.satisfied_demand, file)
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/satisfied_demand.p")
if test:
    wandb.log({"AVG Reward ": rewards_np.mean(), "Std Reward ": rewards_np.std(), "AVG Satisfied Demand ": served_demands_np.mean(), "AVG Charging Cost ": episode_charge_rebalancing_cost.mean(), "AVG Spatial Rebalancing Cost": episode_rebalancing_cost.mean(), "AVG Epoch Time": epoch_times.mean()})
model.save_checkpoint(path=f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn_final.pth")
wandb.save(f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn_final.pth")
wandb.finish()
print("done")
    

