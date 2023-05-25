"""
Autonomous Mobility-on-Demand Environment
-----------------------------------------
This file contains the specifications for the AMoD system simulator. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""
from array import array
from cmath import inf
from collections import defaultdict
from email.charset import add_charset
from itertools import count
import gurobipy as gp
import numpy as np
from src.algos.pax_flows_solver import PaxFlowsSolver
import torch
import os
import math
import networkx as nx
from src.misc.utils import mat2str
from copy import deepcopy
import json


class AMoD:
    # initialization
    # updated to take scenario
    def __init__(self, scenario):
        if scenario.EV == True:
            # I changed it to deep copy so that the scenario input is not modified by env
            self.scenario = deepcopy(scenario)
            # Road Graph: node - node, edge - connection of node, node attr: 'accInit', edge attr: 'time'
            self.G = scenario.G
            self.G_spatial = scenario.G_spatial
            self.rebTime = self.scenario.rebTime
            self.time = 0  # current time
            self.tf = scenario.tf  # final time
            self.demand = defaultdict(dict)  # demand
            self.depDemand = dict()
            self.arrDemand = dict()
            self.nodes = list(self.G.nodes)
            self.nodes_spatial = list(self.G_spatial.nodes)
            self.gcn_edge_idx = None
            self.gcn_edge_idx_spatial = None
            self.number_nodes = len(self.nodes)  # number of nodes
            self.number_nodes_spatial = len(self.nodes_spatial)
            self.region = range(scenario.spatial_nodes)  # set of regions
            for i in self.region:
                self.depDemand[i] = defaultdict(float)
                self.arrDemand[i] = defaultdict(float)

            self.price = defaultdict(dict)  # price
            self.demand = self.scenario.demand_input
            self.price = self.scenario.p
            # number of vehicles within each node, key: i - node, t - time
            self.acc = defaultdict(dict)
            # number of vehicles arriving at each node, key: i - node, t - time
            self.dacc = defaultdict(dict)
            # number of vehicles within each spatial node, key: i - node, t - time
            self.acc_spatial = defaultdict(dict)
            self.n_charging_vehicles_spatial = defaultdict(dict)
            self.new_charging_vehicles = defaultdict(dict)
            self.n_rebal_vehicles_spatial = defaultdict(dict)
            self.n_customer_vehicles_spatial = defaultdict(dict)
            self.satisfied_demand = defaultdict(dict)
            self.new_rebalancing_vehicles = defaultdict(dict)
            # number of vehicles arriving at each spatial node, key: i - node, t - time
            self.dacc_spatial = defaultdict(dict)
            # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
            self.rebFlow = defaultdict(dict)
            # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
            self.paxFlow = defaultdict(dict)
            self.edges = list(self.G.edges)
            self.edges_spatial = list(self.G_spatial.edges)
            # map from od regions to road edges connecting them
            self.map_o_d_regions_to_pax_edges = None
            self.charging_edges = None  # edges only used for rebal
            self.map_node_to_outgoing_edges = None  # maps node to outgoing edges
            self.map_node_to_incoming_edges = None  # maps node to incoming edges
            # maps spatial node to outgoing edges in graph with charge
            self.map_node_spatial_to_outgoing_road_edges_charge_graph = None
            # maps spatial node to incoming edges in graph with charge
            self.map_node_spatial_to_incoming_road_edges_charge_graph = None
            self.map_spatial_node_to_charge_nodes = None  # maps spatial node to charge nodes
            self.create_edge_maps()
            self.create_node_maps()
            self.create_edge_idx_and_weights()

            for i, j in self.G.edges:
                self.rebFlow[i, j] = defaultdict(float)
                self.paxFlow[i, j] = defaultdict(float)

            for n in self.nodes:
                self.acc[n][0] = self.G.nodes[n]['accInit']
                self.dacc[n] = defaultdict(float)
            for n in self.nodes_spatial:
                self.acc_spatial[n][0] = self.G_spatial.nodes[n]['accInit']
                self.n_charging_vehicles_spatial[n][0] = 0
                self.new_charging_vehicles[n][0] = 0
                self.n_rebal_vehicles_spatial[n][0] = 0
                self.new_rebalancing_vehicles[n][0] = 0
                self.n_customer_vehicles_spatial[n][0] = 0
                self.satisfied_demand[n][0] = 0
                self.dacc_spatial[n] = defaultdict(float)
            self.servedDemand = defaultdict(dict)
            for i, j in self.demand:
                self.servedDemand[i, j] = defaultdict(float)

            self.N = len(self.nodes)  # total number of cells

            # add the initialization of info here
            self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost',
                                      'operating_cost', 'charge_rebalancing_cost', 'spatial_rebalancing_cost'], 0)
            self.reward = 0
            # observation: current vehicle distribution, time, future arrivals, demand
            self.obs = (self.acc, self.time, self.dacc, self.demand)
            self.obs_spatial = (self.acc_spatial, self.time,
                                self.dacc_spatial, self.demand)

    def create_edge_maps(self):
        self.map_o_d_regions_to_pax_edges = dict([])
        self.charging_edges = []
        self.map_node_to_outgoing_edges = dict([])
        self.map_node_to_incoming_edges = dict([])
        self.map_node_spatial_to_outgoing_road_edges_charge_graph = dict([])
        self.map_node_spatial_to_incoming_road_edges_charge_graph = dict([])

        for node in self.nodes:
            self.map_node_to_incoming_edges[node] = []
            self.map_node_to_outgoing_edges[node] = []

        for node_spatial in self.nodes_spatial:
            self.map_node_spatial_to_incoming_road_edges_charge_graph[node_spatial] = [
            ]
            self.map_node_spatial_to_outgoing_road_edges_charge_graph[node_spatial] = [
            ]

        for o_region in self.region:
            for d_region in self.region:
                self.map_o_d_regions_to_pax_edges[(o_region, d_region)] = []

        for e in range(len(self.edges)):
            o, d = self.edges[e]
            self.map_node_to_outgoing_edges[o].append(e)
            self.map_node_to_incoming_edges[d].append(e)
            o_region, o_charge = o
            d_region, d_charge = d
            energy_distance = self.scenario.energy_distance[o_region, d_region]
            if (o_charge - d_charge) == energy_distance:
                self.map_node_spatial_to_outgoing_road_edges_charge_graph[o_region].append(
                    e)
                self.map_node_spatial_to_incoming_road_edges_charge_graph[d_region].append(
                    e)
                self.map_o_d_regions_to_pax_edges[(
                    o_region, d_region)].append(e)
            else:
                self.charging_edges.append(e)

    def create_node_maps(self):
        self.map_spatial_node_to_charge_nodes = dict([])
        for node_spatial in self.nodes_spatial:
            self.map_spatial_node_to_charge_nodes[node_spatial] = []
        for node_ind in range(self.number_nodes):
            node = self.nodes[node_ind]
            node_spatial = node[0]
            self.map_spatial_node_to_charge_nodes[node_spatial].append(
                node_ind)

    def create_edge_idx_and_weights(self):
        edge_idx = torch.tensor([[], []], dtype=torch.long)
        edge_idx_spatial = torch.tensor([[], []], dtype=torch.long)
        for e_spatial in self.edges_spatial:
            origin_node_idx = self.nodes_spatial.index(e_spatial[0])
            destination_node_idx = self.nodes_spatial.index(e_spatial[1])
            new_edge = torch.tensor(
                [[origin_node_idx], [destination_node_idx]], dtype=torch.long)
            edge_idx_spatial = torch.cat((edge_idx, new_edge), 1)
        self.gcn_edge_idx_spatial = edge_idx_spatial
        for e in self.edges:
            origin_node_idx = self.nodes.index(e[0])
            destination_node_idx = self.nodes.index(e[1])
            new_edge = torch.tensor(
                [[origin_node_idx], [destination_node_idx]], dtype=torch.long)
            edge_idx = torch.cat((edge_idx, new_edge), 1)
        self.gcn_edge_idx = edge_idx

    def apply_charging_heurstic(self, heuristic: str):
        if heuristic == 'empty_to_full':
            return self.empty_to_full_charge()
        elif heuristic == 'off_peak_charging':
            return self.increased_charging_off_peaks()
        elif heuristic == 'off_peak_one_step':
            return self.increased_charging_off_peaks_one_step()
        elif heuristic == "relative_charging":
            return self.increased_charging_off_peaks_one_step_relative()
        else:
            print("Enter valid heuristic")
            assert False

    # only cars at locations with chargers will be charged
    def empty_to_full_charge(self):
        charging_heuristic_reward = 0
        for region in self.nodes_spatial:
            for c in range(self.scenario.avg_energy_dist):
                n_available_chargers = self.scenario.cars_per_station_capacity[region] - self.scenario.cars_charging_per_station[region]
                if self.acc[(region, c)][self.time+1] > 0 and n_available_chargers > 0:
                    target_charge = self.scenario.number_charge_levels-1
                    charge_time = math.ceil((target_charge-c)/self.scenario.charge_levels_per_charge_step) - self.scenario.time_normalizer
                    n_new_charging_vehicles = min(self.acc[(region, c)][self.time+1], n_available_chargers)
                    avg_energy_price = np.mean(
                        self.scenario.p_energy[self.time:self.time+charge_time + self.scenario.time_normalizer])
                    charging_heuristic_reward -= n_new_charging_vehicles * (target_charge-c) * avg_energy_price
                    if (self.time+charge_time) not in self.rebFlow[(region, c), (region, target_charge)].keys():
                        self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] = 0
                    self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] += n_new_charging_vehicles
                    self.scenario.cars_charging_per_station[region] += n_new_charging_vehicles
                    self.acc[(region, c)][self.time+1] -= n_new_charging_vehicles
                    self.acc_spatial[region][self.time+1] -= n_new_charging_vehicles
                    self.n_charging_vehicles_spatial[region][self.time+1] += n_new_charging_vehicles
                    self.new_charging_vehicles[region][self.time+1] += n_new_charging_vehicles
                
        return charging_heuristic_reward

    # only cars at locations with chargers will be charged
    def empty_to_one_step(self):
        charging_heuristic_reward = 0
        for region in self.nodes_spatial:
            for c in range(self.scenario.avg_energy_dist):
                n_available_chargers = self.scenario.cars_per_station_capacity[region] - self.scenario.cars_charging_per_station[region]
                if self.acc[(region, c)][self.time+1] > 0 and n_available_chargers > 0:
                    target_charge = min(self.scenario.number_charge_levels-1, c+self.scenario.charge_levels_per_charge_step)
                    charge_time = math.ceil((target_charge-c)/self.scenario.charge_levels_per_charge_step) - self.scenario.time_normalizer
                    n_new_charging_vehicles = min(self.acc[(region, c)][self.time+1], n_available_chargers)
                    avg_energy_price = np.mean(
                        self.scenario.p_energy[self.time:self.time+charge_time + self.scenario.time_normalizer])
                    charging_heuristic_reward -= n_new_charging_vehicles * (target_charge-c) * avg_energy_price
                    if (self.time+charge_time) not in self.rebFlow[(region, c), (region, target_charge)].keys():
                        self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] = 0
                    self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] += n_new_charging_vehicles
                    self.scenario.cars_charging_per_station[region] += n_new_charging_vehicles
                    self.acc[(region, c)][self.time+1] -= n_new_charging_vehicles
                    self.acc_spatial[region][self.time+1] -= n_new_charging_vehicles
                    self.n_charging_vehicles_spatial[region][self.time+1] += n_new_charging_vehicles
                    self.new_charging_vehicles[region][self.time+1] += n_new_charging_vehicles
                
        return charging_heuristic_reward

    def increased_charging_off_peaks(self):
        # increase charging during Mid day low (charge bottom 30%)
        charging_heuristic_reward = 0
        hour = int(self.time * self.scenario.time_granularity)
        if hour not in self.scenario.peak_hours:
            # we want to charge as many vehicles as possible in the bottom third of charge, starting with the lowest charges
            charge_limit = math.ceil(self.scenario.number_charge_levels*0.3)
            for c in range(charge_limit):
                for region in self.nodes_spatial:
                    n_available_chargers = self.scenario.cars_per_station_capacity[region] - self.scenario.cars_charging_per_station[region]
                    if self.acc[(region, c)][self.time+1] > 0 and n_available_chargers > 0:
                        target_charge = self.scenario.number_charge_levels - 1
                        charge_diff = target_charge - c
                        charge_time = math.ceil(charge_diff/self.scenario.charge_levels_per_charge_step) - self.scenario.time_normalizer
                        n_new_charging_vehicles = min(self.acc[(region, c)][self.time+1], n_available_chargers)
                        avg_energy_price = np.mean(self.scenario.p_energy[self.time:self.time+charge_time+self.scenario.time_normalizer])
                        charging_heuristic_reward -= n_new_charging_vehicles * charge_diff * avg_energy_price
                        if (self.time+charge_time) not in self.rebFlow[(region, c), (region, target_charge)].keys():
                            self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] = 0
                        self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] += n_new_charging_vehicles
                        self.scenario.cars_charging_per_station[region] += n_new_charging_vehicles
                        self.acc[(region, c)][self.time+1] -= n_new_charging_vehicles
                        self.acc_spatial[region][self.time+1] -= n_new_charging_vehicles
                        self.n_charging_vehicles_spatial[region][self.time+1] += n_new_charging_vehicles
                        self.new_charging_vehicles[region][self.time+1] += n_new_charging_vehicles
            return charging_heuristic_reward
        else:
            return self.empty_to_full_charge()

    def increased_charging_off_peaks_one_step(self):
        # increase charging during Mid day low (charge bottom 30%)
        charging_heuristic_reward = 0
        hour = int(self.time * self.scenario.time_granularity)
        if hour not in self.scenario.peak_hours:
            # we want to charge as many vehicles as possible in the bottom third of charge, starting with the lowest charges
            charge_limit = math.ceil(self.scenario.number_charge_levels*0.3)
            for c in range(charge_limit):
                for region in self.nodes_spatial:
                    n_available_chargers = self.scenario.cars_per_station_capacity[region] - self.scenario.cars_charging_per_station[region]
                    if self.acc[(region, c)][self.time+1] > 0 and n_available_chargers > 0:
                        target_charge = min(self.scenario.number_charge_levels - 1, c+self.scenario.charge_levels_per_charge_step)
                        charge_diff = target_charge - c
                        charge_time = math.ceil(charge_diff/self.scenario.charge_levels_per_charge_step) - self.scenario.time_normalizer
                        n_new_charging_vehicles = min(self.acc[(region, c)][self.time+1], n_available_chargers)
                        avg_energy_price = np.mean(self.scenario.p_energy[self.time:self.time+charge_time+self.scenario.time_normalizer])
                        charging_heuristic_reward -= n_new_charging_vehicles * charge_diff * avg_energy_price
                        if (self.time+charge_time) not in self.rebFlow[(region, c), (region, target_charge)].keys():
                            self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] = 0
                        self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] += n_new_charging_vehicles
                        self.scenario.cars_charging_per_station[region] += n_new_charging_vehicles
                        self.acc[(region, c)][self.time+1] -= n_new_charging_vehicles
                        self.acc_spatial[region][self.time+1] -= n_new_charging_vehicles
                        self.n_charging_vehicles_spatial[region][self.time+1] += n_new_charging_vehicles
                        self.new_charging_vehicles[region][self.time+1] += n_new_charging_vehicles
            return charging_heuristic_reward
        else:
            return self.empty_to_one_step()

    def increased_charging_off_peaks_one_step_relative(self):
        # increase charging during Mid day low (charge bottom 30%) of vehicles
        charging_heuristic_reward = 0
        hour = int(self.time * self.scenario.time_granularity)
        remaining_available_vehicles_for_charging = np.zeros(self.number_nodes_spatial)
        for region in self.nodes_spatial:
            remaining_available_vehicles_for_charging[region] = 0.3*self.acc_spatial[region][self.time+1]
        if hour not in self.scenario.peak_hours:
            # we want to charge as many vehicles as possible in the bottom third of vehicles, starting with the lowest charges
            for region in self.nodes_spatial:
                for c in range(self.scenario.number_charge_levels):
                    n_available_chargers = self.scenario.cars_per_station_capacity[region] - self.scenario.cars_charging_per_station[region]
                    if self.acc[(region, c)][self.time+1] > 0 and n_available_chargers > 0 and remaining_available_vehicles_for_charging[region] > 0:
                        target_charge = min(self.scenario.number_charge_levels - 1, c+self.scenario.charge_levels_per_charge_step)
                        charge_diff = target_charge - c
                        charge_time = math.ceil(charge_diff/self.scenario.charge_levels_per_charge_step) - self.scenario.time_normalizer
                        n_new_charging_vehicles = min(self.acc[(region, c)][self.time+1], n_available_chargers, remaining_available_vehicles_for_charging[region])
                        avg_energy_price = np.mean(self.scenario.p_energy[self.time:self.time+charge_time+self.scenario.time_normalizer])
                        charging_heuristic_reward -= n_new_charging_vehicles * charge_diff * avg_energy_price
                        if (self.time+charge_time) not in self.rebFlow[(region, c), (region, target_charge)].keys():
                            self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] = 0
                        self.rebFlow[(region, c), (region, target_charge)][self.time+charge_time] += n_new_charging_vehicles
                        self.scenario.cars_charging_per_station[region] += n_new_charging_vehicles
                        self.acc[(region, c)][self.time+1] -= n_new_charging_vehicles
                        self.acc_spatial[region][self.time+1] -= n_new_charging_vehicles
                        self.n_charging_vehicles_spatial[region][self.time+1] += n_new_charging_vehicles
                        self.new_charging_vehicles[region][self.time+1] += n_new_charging_vehicles
                        remaining_available_vehicles_for_charging[region] -= n_new_charging_vehicles
            return charging_heuristic_reward
        else:
            return self.empty_to_one_step()


    # pax step
    def pax_step(self, paxAction=None, pax_flows_solver=None):
        t = self.time
        self.reward = 0
        new_customer_vehicles = 0
        for n in self.nodes:
            self.acc[n][t+1] = self.acc[n][t]
        for n_spatial in self.nodes_spatial:
            self.acc_spatial[n_spatial][t+1] = self.acc_spatial[n_spatial][t]
            assert self.acc_spatial[n_spatial][t] >= -1e-8
            self.n_charging_vehicles_spatial[n_spatial][t+1] = self.n_charging_vehicles_spatial[n_spatial][t]
            self.new_charging_vehicles[n_spatial][t+1] = 0
            assert self.n_charging_vehicles_spatial[n_spatial][t] >= -1e-8
            self.n_rebal_vehicles_spatial[n_spatial][t+1] = self.n_rebal_vehicles_spatial[n_spatial][t]
            self.new_rebalancing_vehicles[n_spatial][t+1] = 0
            assert self.n_rebal_vehicles_spatial[n_spatial][t] >= -1e-8
            self.n_customer_vehicles_spatial[n_spatial][t+1] = self.n_customer_vehicles_spatial[n_spatial][t]
            assert self.n_customer_vehicles_spatial[n_spatial][t] >= -1e-8
            self.satisfied_demand[n_spatial][t+1] = 0
        
        self.info['served_demand'] = 0  # initialize served demand
        self.info["operating_cost"] = 0  # initialize operating cost
        self.info['revenue'] = 0
        self.info['rebalancing_cost'] = 0
        self.info['charge_rebalancing_cost'] = 0
        self.info['spatial_rebalancing_cost'] = 0
        # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
        if paxAction is None:
            paxAction = pax_flows_solver.optimize()
        self.paxAction = paxAction
        # serving passengers
        satisfied_demand_local = np.zeros(self.number_nodes_spatial)
        total_demand = np.zeros(self.number_nodes_spatial)
        for origin in range(self.number_nodes_spatial):
            for destination in range(self.number_nodes_spatial):
                total_demand[origin] += self.demand[origin, destination][t]
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            i_region = i[0]
            j_region = j[0]
            if (i_region, j_region) not in self.demand or t not in self.demand[i_region, j_region] or self.paxAction[k] < 1e-3 or i[1] < j[1]:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            assert paxAction[k] < self.acc[i][t+1] + 1e-3
            assert paxAction[k] >= 0
            self.paxAction[k] = min(self.acc[i][t+1], paxAction[k])
            self.servedDemand[i_region, j_region][t] += self.paxAction[k]
            satisfied_demand_local[i_region] += self.paxAction[k]
            self.paxFlow[i, j][t+self.G.edges[i, j]['time'][self.time]] = self.paxAction[k]
            self.info["operating_cost"] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.paxAction[k]
            self.acc[i][t+1] -= self.paxAction[k]
            self.acc_spatial[i[0]][t+1] -= self.paxAction[k]
            self.n_customer_vehicles_spatial[i[0]][t+1] += self.paxAction[k]
            self.satisfied_demand[i[0]][t+1] += self.paxAction[k]
            new_customer_vehicles += self.paxAction[k]
            self.info['served_demand'] += self.paxAction[k]
            self.dacc[j][t+self.G.edges[i, j]['time'][self.time]+self.scenario.time_normalizer] += self.paxFlow[i, j][t+self.G.edges[i, j]['time'][self.time]]
            self.dacc_spatial[j_region][t+self.G.edges[i, j]['time'][self.time]+self.scenario.time_normalizer] += self.paxFlow[i, j][t+self.G.edges[i, j]['time'][self.time]]
            self.reward += self.paxAction[k]*(self.price[i_region, j_region][t] - (self.G.edges[i, j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep)
            self.info['revenue'] += self.paxAction[k] * (self.price[i_region, j_region][t])

        test_spatial_acc_count = np.zeros(self.number_nodes_spatial)
        for n in self.nodes:
            test_spatial_acc_count[n[0]] += self.acc[n][t+1]
        for region in self.nodes_spatial:
            assert abs(test_spatial_acc_count[region] - self.acc_spatial[region][t+1]) < 1e-5
            assert satisfied_demand_local[region] - total_demand[region] < 1e-5

        # for acc, the time index would be t+1, but for demand, the time index would be t
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.obs_spatial = (self.acc_spatial, self.time, self.dacc_spatial, self.demand)
        done = False  # if passenger matching is executed first

        return self.obs_spatial, max(0, self.reward), done, self.info

    # reb step
    def reb_step(self, rebAction):
        t = self.time
        self.reward = 0  # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing

        self.rebAction = rebAction
        # rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.G.edges:
                assert False
            # update the number of vehicles
            if not rebAction[k] < self.acc[i][t+1] + 1e-3:
                print("Nodes", i,j)
                print("Values", rebAction[k], self.acc[i][t+1])
                print("k", k)
            assert rebAction[k] < self.acc[i][t+1] + 1e-3
            if rebAction[k] < 1e-3:
                continue
            self.rebAction[k] = min(self.acc[i][t+1], rebAction[k])
            if self.rebFlow[i, j][t+self.G.edges[i, j]['time'][self.time]] == None:
                self.rebFlow[i, j][t+self.G.edges[i, j]['time'][self.time]] = self.rebAction[k]
            else:
                self.rebFlow[i, j][t+self.G.edges[i, j]['time'][self.time]] += self.rebAction[k]
            self.dacc[j][t+self.G.edges[i, j]['time'][self.time]+self.scenario.time_normalizer] += self.rebFlow[i, j][t+self.G.edges[i, j]['time'][self.time]]
            self.dacc_spatial[j[0]][t+self.G.edges[i, j]['time'][self.time]+self.scenario.time_normalizer] += self.rebFlow[i, j][t+self.G.edges[i, j]['time'][self.time]]
            self.acc[i][t+1] -= self.rebAction[k]
            self.acc_spatial[i[0]][t+1] -= self.rebAction[k]
            # charging edge
            if i[1] < j[1] and self.rebAction[k] > 0:
                assert i[0] == j[0]
                charge_difference = j[1] - i[1]
                charge_time = math.ceil(charge_difference/self.scenario.charge_levels_per_charge_step)
                avg_energy_price = np.mean(self.scenario.p_energy[self.time:self.time+charge_time])
                self.info['rebalancing_cost'] += avg_energy_price * self.rebAction[k]*charge_difference
                self.info['charge_rebalancing_cost'] += avg_energy_price * self.rebAction[k]*charge_difference
                # charge cost negatively influences the reward
                self.reward -= avg_energy_price * self.rebAction[k]*charge_difference
                self.scenario.cars_charging_per_station[i[0]] += self.rebAction[k]
                self.n_charging_vehicles_spatial[i[0]][t+1] += self.rebAction[k]
                self.new_charging_vehicles[i[0]][t+1] += rebAction[k]
                # added 1e-5 as tolerance because optimisation result is not exact
                assert self.scenario.cars_charging_per_station[i[0]] <= self.scenario.cars_per_station_capacity[i[0]] + 1e-5
            # road edge
            elif self.rebAction[k] > 0:
                assert i[0] != j[0] or i == j
                self.n_rebal_vehicles_spatial[i[0]][t+1] += self.rebAction[k]
                self.new_rebalancing_vehicles[i[0]][t+1] += self.rebAction[k]
                self.info['rebalancing_cost'] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
                self.info['spatial_rebalancing_cost'] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
                self.info["operating_cost"] += (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
                self.reward -= (self.G.edges[i,j]['time'][self.time] + self.scenario.time_normalizer)*self.scenario.operational_cost_per_timestep*self.rebAction[k]
        # arrival for the next time step, executed in the last state of a time step
        for k in range(len(self.edges)):
            o, d = self.edges[k]
            if (o, d) in self.rebFlow and t in self.rebFlow[o, d]:
                self.acc[d][t+1] += self.rebFlow[o, d][t]
                self.acc_spatial[d[0]][t+1] += self.rebFlow[o, d][t]
                # check if charging capacity has freed up
                if d[1] > o[1]:
                    # charging should only happen at one location
                    assert o[0] == d[0]
                    self.scenario.cars_charging_per_station[o[0]] -= self.rebFlow[o, d][t]
                    self.n_charging_vehicles_spatial[o[0]][t+1] -= self.rebFlow[o, d][t]
                else:
                    self.n_rebal_vehicles_spatial[o[0]][t+1] -= self.rebFlow[o, d][t]

            if (o, d) in self.paxFlow and t in self.paxFlow[o, d]:
                self.acc[d][t+1] += self.paxFlow[o, d][t]
                self.acc_spatial[d[0]][t+1] += self.paxFlow[o, d][t]
                self.n_customer_vehicles_spatial[o[0]][t+1] -= self.paxFlow[o, d][t]
        
        # improve implementation, should not be hardcoded from data
        # hour = (t+1) * self.scenario.time_granularity + 8
        # # peak demand starts at 16h
        # if hour == 15:
        #     new_vehicles_per_node = int(self.scenario.additional_vehicles_peak_demand/self.number_nodes_spatial)
        #     for node_spatial in self.nodes_spatial:
        #         self.acc[(node_spatial, self.scenario.number_charge_levels-1)][t+1] += new_vehicles_per_node
        #         self.acc_spatial[node_spatial][t+1] += new_vehicles_per_node


        self.time += 1
        # use self.time to index the next time step
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        done = (self.tf == t+1)  # if the episode is completed
        return self.obs, self.reward, done, self.info

    def reset(self, bool_sample_demand=True):
        # reset the episode
        self.acc = defaultdict(dict)
        self.acc_spatial = defaultdict(dict)
        self.n_charging_vehicles_spatial = defaultdict(dict)
        self.new_charging_vehicles = defaultdict(dict)
        self.n_rebal_vehicles_spatial = defaultdict(dict)
        self.new_rebalancing_vehicles = defaultdict(dict)
        self.n_customer_vehicles_spatial = defaultdict(dict)
        self.satisfied_demand = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.demand = defaultdict(dict)  # demand
        self.price = defaultdict(dict)  # price
        self.scenario.cars_charging_per_station = np.zeros_like(self.scenario.cars_per_station_capacity)
        tripAttr = self.scenario.get_random_demand(bool_sample_demand)
        # trip attribute (origin, destination, time of request, demand, price)
        for i, j, t, d, p in tripAttr:
            self.demand[i, j][t] = d
            self.price[i, j][t] = p

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        for n_spatial in self.nodes_spatial:
            self.acc_spatial[n_spatial][0] = self.G_spatial.nodes[n_spatial]['accInit']
            self.dacc_spatial[n_spatial] = defaultdict(float)
            self.n_charging_vehicles_spatial[n_spatial][0] = 0
            self.new_charging_vehicles[n_spatial][0] = 0
            self.n_rebal_vehicles_spatial[n_spatial][0] = 0
            self.new_rebalancing_vehicles[n_spatial][0] = 0
            self.n_customer_vehicles_spatial[n_spatial][0] = 0
            self.satisfied_demand[n_spatial][0] = 0
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        return self.obs


class Scenario:
    def __init__(self, EV=True, spatial_nodes=4, charging_stations=None, cars_per_station_capacity=None, number_charge_levels=10, charge_levels_per_charge_step=1, energy_distance=None, tf=60, sd=None, tripAttr=None, peak_hours=[],
                 demand_ratio=None, trip_length_preference=0.25, grid_travel_time=1, alpha=0.2, reb_time=None, total_acc=None, additional_vehicles_peak_demand=0, p_energy=None, time_granularity=0.5, operational_cost_per_timestep=0.5):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each nodes,
        #          float/int - total demand out of each nodes satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        self.EV = EV
        if sd != None:
            np.random.seed(self.sd)

        if EV == True:
            # self.additional_vehicles_peak_demand = additional_vehicles_peak_demand
            self.peak_hours = peak_hours
            self.time_normalizer = 1
            self.time_granularity = time_granularity
            self.operational_cost_per_timestep = operational_cost_per_timestep
            self.spatial_nodes = spatial_nodes
            self.charging_stations = charging_stations
            self.cars_per_station_capacity = cars_per_station_capacity
            self.cars_charging_per_station = np.zeros_like(
                self.cars_per_station_capacity)
            self.number_charge_levels = number_charge_levels
            self.charge_levels_per_charge_step = charge_levels_per_charge_step
            self.energy_distance = energy_distance
            self.avg_energy_dist = math.ceil(np.mean(energy_distance))
            self.max_energy_distance = int(np.max(energy_distance))
            self.p_energy = np.array(p_energy)  # price of energy in $/kWh
            self.intermediate_charging_station = defaultdict(dict)
            self.time = 0  # current time
            self.is_json = False
            self.alpha = alpha
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.tf = tf
            self.G = nx.empty_graph()
            self.G = self.G.to_directed()
            self.G_spatial = nx.empty_graph()
            self.G_spatial = self.G_spatial.to_directed()

            self.demand_input, self.p, self.rebTime = defaultdict(
                dict), defaultdict(dict), defaultdict(dict)
            for item in tripAttr:
                t, o, d, v, p = item['time_stamp'], item['origin'], item['destination'], item['demand'], item['price']
                if (o, d) not in self.demand_input:
                    self.demand_input[o, d], self.p[o, d] = defaultdict(
                        float), defaultdict(float)
                self.demand_input[o, d][t] += v*demand_ratio
                self.p[o, d][t] += p*demand_ratio

            for item in reb_time:
                hr, o, d, rt = item["time_stamp"], item["origin"], item["destination"], item["reb_time"]
                for t in range(0, tf+1):
                    self.rebTime[o, d][t] = int(round(rt))

            # add charge edges
            self.add_charge_edges()

            # add road edges
            self.add_road_edges()

            self.edges = list(self.G.edges)
            self.spatial_edges = list(self.G_spatial.edges)
            self.tf = tf

            for o, d in self.edges:
                for t in range(0, tf*2):
                    if t in self.demand_input[o[0], d[0]] and self.demand_input[o[0], d[0]][t] > 0:
                        continue
                    else:
                        self.demand_input[o[0], d[0]][t] = 0
                        self.p[o[0], d[0]][t] = 0

            for item in total_acc:
                hr, acc = item['hour'], item['acc']
                number_vehicles_distr = 0
                for region in self.G_spatial.nodes:
                    self.G_spatial.nodes[region]['accInit'] = int(0)
                    for c in range(self.number_charge_levels):
                        # cut_off_charge = int(0.5*self.number_charge_levels)
                        # number_of_used_charges = (self.number_charge_levels-cut_off_charge)
                        # number_cars_per_node = int(acc/(len(list(self.G_spatial.nodes))*number_of_used_charges))
                        # if c >= cut_off_charge:
                        #     self.G.nodes[(region,c)]['accInit'] = number_cars_per_node
                        #     number_vehicles_distr += number_cars_per_node
                        #     self.G_spatial.nodes[region]['accInit'] += number_cars_per_node
                        # else:
                        #     self.G.nodes[(region,c)]['accInit'] = 0

                        # only bottom 60%
                        cut_off_charge = int(1.*self.number_charge_levels)
                        number_of_used_charges = cut_off_charge
                        number_cars_per_node = int(acc/(len(list(self.G_spatial.nodes))*number_of_used_charges))
                        if c <= cut_off_charge:
                            self.G.nodes[(region,c)]['accInit'] = number_cars_per_node
                            number_vehicles_distr += number_cars_per_node
                            self.G_spatial.nodes[region]['accInit'] += number_cars_per_node
                        else:
                            self.G.nodes[(region,c)]['accInit'] = 0
                print(acc, number_vehicles_distr)   
                break  # only need the first time step, if I want variable acc, I need to change this

            self.tripAttr = self.get_random_demand()  # randomly generated demand

    def add_charge_edges(self):
        for l in range(self.spatial_nodes):
            if not self.charging_stations[l]:
                continue
            for c1 in range(self.number_charge_levels - 1):
                fully_charged = c1 == (self.number_charge_levels-1)
                c2 = c1
                while not fully_charged:
                    c2 += self.charge_levels_per_charge_step
                    if c2 >= self.number_charge_levels:
                        c2 = (self.number_charge_levels-1)
                        fully_charged = True
                    assert c1 >= 0 and c2 > c1 and c2 < self.number_charge_levels
                    self.G.add_edge((l, c1), (l, c2))
                    self.G.edges[(l, c1), (l, c2)]['time'] = dict()
                    for t in range(0, self.tf+1):
                        self.G.edges[(l, c1), (l, c2)]['time'][t] = math.ceil((c2-c1)/self.charge_levels_per_charge_step) - self.time_normalizer

    def add_road_edges(self):
        for o in range(self.spatial_nodes):
            for d in range(self.spatial_nodes):
                self.G_spatial.add_edge(o, d)
                self.G_spatial.edges[o, d]['time'] = dict()
                for t in range(0, self.tf+1):
                    self.G_spatial.edges[o, d]['time'][t] = math.ceil(
                        self.rebTime[o, d][t]) - self.time_normalizer
                for c in reversed(range(self.number_charge_levels)):
                    # removes top and bottom node for nodes without charge stations -> removes infeasible edges
                    target_charge = int(c - self.energy_distance[o, d])
                    if (not self.charging_stations[o]) and (c == self.number_charge_levels-1):
                        continue
                    elif (target_charge < 0):
                        break
                    elif (not self.charging_stations[d]) and (target_charge == 0):
                        break
                    assert target_charge < c  # we have to loose energy to move
                    self.G.add_edge((o, c), (d, target_charge))
                    self.G.edges[(o, c), (d, target_charge)]['time'] = dict()
                    for t in range(0, self.tf+1):
                        self.G.edges[(o, c), (d, target_charge)]['time'][t] = math.ceil(
                            self.rebTime[o, d][t]) - self.time_normalizer

    def get_random_demand(self, bool_random = True):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand

        demand = defaultdict(dict)
        price = defaultdict(dict)
        tripAttr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        if self.EV:
            for t in range(0, self.tf*2):
                for i, j in self.edges:
                    if (i[0], j[0]) in self.demand_input and t in self.demand_input[i[0], j[0]]:
                        if bool_random:
                            demand[i[0],j[0]][t] = np.random.poisson(self.demand_input[i[0],j[0]][t])
                        else:
                            demand[i[0], j[0]][t] = self.demand_input[i[0], j[0]][t]
                        price[i[0], j[0]][t] = self.p[i[0],j[0]][t]  
                    else:
                        demand[i[0], j[0]][t] = 0
                        price[i[0], j[0]][t] = 0
                    tripAttr.append(
                        (i[0], j[0], t, demand[i[0], j[0]][t], price[i[0], j[0]][t]))

        return tripAttr
