from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import optparse
import random
import math
import numpy as np
import xml.etree.ElementTree as ET
from itertools import permutations
from ast import literal_eval
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    print(os.environ['SUMO_HOME'])
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import traci
import traci.constants as tc
import sumolib
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEPART = [
    "-gneE56", "-gneE5", "gneE4", "gneE17"
]
ARRIVAL_56 = [
     "gneE5", "-gneE4", "-gneE17"
]
ARRIVAL_5 = [ 
    "gneE56", "-gneE4", "-gneE17"
]
ARRIVAL_4 = [
    "gneE56", "gneE5", "-gneE17"
]
ARRIVAL_17 = [
    "gneE56", "gneE5", "-gneE4"
]

SPEED = 5
DISTANCE = SPEED * 10
net = sumolib.net.readNet('data/crossroads.net.xml')
JUNCTION_NODE = "gneJ4"
PRIORITY = {JUNCTION_NODE:"gneE5"}


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        #print(data)
        #print("\n")
        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.long))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, 64)
        self.l2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.state_size = 4
        self.action_size = 4

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.state_size, self.action_size)
        self.qnet_target = QNet(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :])
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def save(self, path):
        torch.save(self.qnet.state_dict(), path)

    def load(self, path):
        self.qnet.load_state_dict(torch.load(path))
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        

def init(is_gui):
    if is_gui:
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
    else:
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo')
    sumoCmd = [sumoBinary, "-c", "data/crossroads.sumocfg",]
    traci.start(sumoCmd)

def make_vehicle(vehID, routeID, depart_time):
    traci.vehicle.addLegacy(vehID, routeID, depart=depart_time)
    traci.vehicle.setSpeed(vehID, SPEED)
    traci.vehicle.setMaxSpeed(vehID, SPEED)

def make_random_route(num):
    ok = True
    while ok:
        depart = random.choice(DEPART)
        if depart=="-gneE56":
            arrive = random.choice(ARRIVAL_56)
        elif depart=="-gneE5":
            arrive = random.choice(ARRIVAL_5)
        elif depart=="gneE4":
            arrive = random.choice(ARRIVAL_4)
        elif depart=="gneE17":
            arrive = random.choice(ARRIVAL_17)
        try:
            traci.route.add(f"random_route_{num}", [depart, arrive])
            ok = False
        except:
            pass
    return f"random_route_{num}"

def get_state(nodeID, t_start):
    """
    状態空間を定義する関数

    状態は各車線の停止車両数 v1,v2,v3,v4
    車両待機列 t_wait
    今の優先車線が選ばれてからどれくらい経過したか t_priority

    state(input) = [v1,v2,v3,v4]

    """
    state = []
    for edge in net.getNode(nodeID).getIncoming():
        vehicles = traci.edge.getLastStepVehicleNumber(edge.getID())
        state.append(vehicles)
    state_normalize = normalize_minmax(state)
    return torch.tensor(state_normalize, dtype=torch.float32)

def normalize_minmax(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def get_distacne(vehID, net):
    try:

        current_edge = traci.vehicle.getRoadID(vehID)
        nextNodeID = net.getEdge(current_edge).getToNode().getID()
        vehicle_pos = traci.vehicle.getPosition(vehID)
        junction_pos = traci.junction.getPosition(nextNodeID)
        junction_vehicle_distance = traci.simulation.getDistance2D(
            vehicle_pos[0], vehicle_pos[1], junction_pos[0], junction_pos[1])

        return junction_vehicle_distance

    except:
        pass

def traffic_control(nodeID, action, prev_t_start): #nodeID=交差点
    control_obj = {}
    t_start = prev_t_start
    for edge in junction_edges:
        try:
            v = traci.edge.getLastStepVehicleIDs(edge)[-1]
            lane = traci.vehicle.getLaneID(v)
            distance = get_distacne(v, net)
            if distance < DISTANCE:
                control_obj[edge] = v
        except:
            pass

    #print("何もしない")  # next_action=4 の場合は変更なし
    PRIORITY[nodeID] = junction_edges[action]
    t_start = traci.simulation.getTime()
    # 優先車線を動的に設定

    for edge in control_obj:
        vehicle = control_obj[edge]
        if PRIORITY[nodeID] == edge:
            traci.vehicle.setColor(vehicle, (255, 0, 0))
            traci.vehicle.setSpeed(vehicle, SPEED)
        else:
            traci.vehicle.setSpeed(vehicle, 0)
    return t_start
    

def get_reward(junction_edges, prev_deadlock):
    reward = 0
    current_count = 0
    previous_vehicles = []
    """
    for edge in junction_edges:
        current_vehicles = set(traci.edge.getLastStepVehicleIDs(edge))
        new_vehicles = current_vehicles - prev_vehicles[edge]
        current_count += len(new_vehicles)
        # 現在の車両リストを記録
        previous_vehicles[edge] = current_vehicles
    """
    deadlock_detected = len(traci.simulation.getCollisions()) > 0 
    teleport_occurs = traci.simulation.getEndingTeleportNumber()
    #print("current_waiting=",current_waiting)
    #print("prev_waiting=", prev_waiting)
    #passed_vehicle = prev_waiting - current_waiting
    #total_pass_veh += passed_vehicle
    #print(total_pass_veh)
    
    # 車両が交差点を通過した分だけ報酬
    #print("current_count=", current_count)
    #reward += current_count

        # 各車線の待ち時間を取得
    #total_waiting_time = sum([traci.edge.getWaitingTime(edge) for edge in junction_edges])
    
    # 待ち時間をペナルティとして適用
    #reward -= total_waiting_time * 0.01

    #時間経過ペナルティ
    #reward -= time

    # デッドロック報酬
    if deadlock_detected:
        reward -= 5
        #print("collision!")
    elif prev_deadlock and not deadlock_detected:
        reward += 5
    #テレポート報酬
    if teleport_occurs > 0:
        reward -= teleport_occurs * 10
    """
    #車線変更コスト報酬
    if action != 4:
        reward -= 1
    else:
        reward += 1
    """
    #print("reward=", reward)
    return reward, deadlock_detected

def set_simulation_time_limit(limit):
    global SIMULATION_TIME_LIMIT
    SIMULATION_TIME_LIMIT = limit

def reward_save(episode, episode_reward):
    with open(file, "a", encoding="utf-8") as f:
        f.write(f"{episode}\t{episode_reward}\n")

    

sync_interval = 20
agent = DQNAgent()
reward_history = []

def simulation(num, episode_num):
    t_start = 0
    node = JUNCTION_NODE
    global junction_edges
    junction_edges = []
    for edge in net.getNode(node).getIncoming():
        junction_edges.append(edge.getID())
    for episode in range(1,episode_num+1):
        init(False)
        teleportNum = 0
        time= 0
        state = get_state(JUNCTION_NODE, t_start)
        done = False
        total_reward = 0
        prev_wait = 0
        prev_deadlock = 0
        epsilon = 0.1 + 0.9 * math.exp(-1. * episode / 100)
        teleported_vehicles = []

        for i in range(num):
            make_vehicle(f"vehicle_{i}", make_random_route(i), 0)

        while not done and traci.simulation.getMinExpectedNumber() > 0 and SIMULATION_TIME_LIMIT > time:
            traci.simulationStep()
            #print(f"teleported_vehicles={teleported_vehicles}\n")
            for vehID in teleported_vehicles:
                traci.vehicle.setSpeed(vehID, SPEED)
            if traci.simulation.getMinExpectedNumber() == 0 or time >= SIMULATION_TIME_LIMIT:
                done = True
            action = agent.get_action(state, epsilon)
            t_start = traffic_control(node, action, t_start)
            reward, prev_deadlock = get_reward(junction_edges, prev_deadlock) 
            next_state = get_state(node, t_start)
            #print(next_state,"\n")
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            teleportNum += traci.simulation.getEndingTeleportNumber()
            teleported_vehicles = traci.simulation.getEndingTeleportIDList()
            time = traci.simulation.getTime()
        if episode % sync_interval == 0:
            agent.sync_qnet()
        reward_history.append(total_reward)
        if episode % 10 == 0:
            print("episode :{}, total reward : {}".format(episode, total_reward))
        reward_save(episode, total_reward)
        traci.close()

mode = input("Mode (train/test) :").strip().lower()
num_of_vehicles = int(input("Num of vehicles :"))
num_of_episode = int(input("Num of episode :"))
name = input("what is rewardfilename :")
filename = "reward_" + name + ".txt"
file = os.path.join("output", filename)
limit = 3600

set_simulation_time_limit(limit)

if mode == "train":
    if os.path.exists("output/dqn_model.pth"):
        os.remove("output/dqn_model.pth")
    simulation(num_of_vehicles, num_of_episode)
    agent.save("output/dqn_model.pth")
    print("Model saved to dqn_model.pth")

elif mode == "test":
    agent.load("output/dqn_model.pth")
    print("Model loaded from dqn_model.pth")
    def test_simulation(num, episode_num):
        t_start = 0
        node = JUNCTION_NODE
        global junction_edges
        junction_edges = []
        for edge in net.getNode(node).getIncoming():
            junction_edges.append(edge.getID())
        for episode in range(1, episode_num + 1):
            init(True)
            time = 0
            state = get_state(JUNCTION_NODE, t_start)
            done = False
            total_reward = 0
            prev_reward = 0
            prev_teleport = 0
            teleport_num =0
            prev_deadlock = 0
            current_time = 0
            action = 0
            teleported_vehicles = []
            for i in range(num):
                make_vehicle(f"vehicle_{i}", make_random_route(i), 0)

            while not done and traci.simulation.getMinExpectedNumber() > 0 and SIMULATION_TIME_LIMIT > time:
                traci.simulationStep()
                if traci.simulation.getMinExpectedNumber() == 0 or time >= SIMULATION_TIME_LIMIT:
                    done = True
                for vehID in teleported_vehicles:
                    traci.vehicle.setSpeed(vehID, SPEED)
                prev_time = current_time
                print("action=",action)
                action = agent.get_action(state, epsilon=0.0)  # 探索なし
                t_start = traffic_control(node, action, t_start)
                reward, prev_deadlock = get_reward(junction_edges, prev_deadlock)
                next_state = get_state(node, t_start)
                state = next_state
                total_reward += reward
                teleport_num += traci.simulation.getEndingTeleportNumber()
                teleported_vehicles = traci.simulation.getEndingTeleportIDList()
            print(f"[Test] Episode {episode}, total reward: {total_reward}")
            traci.close()
    
    test_simulation(num_of_vehicles, num_of_episode)

