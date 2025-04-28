from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import optparse
import random
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
QTable_pass = "output/Q_table.txt"
Result_pass = "output/Simulation_results.txt"
class MyQTable():
    #QTable作成
    def __init__(self, action_num, state_num):
        if use_Qtable:
            with open(QTable_pass, 'r') as f:
                pre_qtable = f.read()
            self._QTable = np.array(literal_eval(pre_qtable))
        else:
            self._QTable = np.random.uniform(low=-1, high=1, size=(state_num, action_num))
            with open(QTable_pass, mode='w') as f:
                f.writelines(np.array2string(self._QTable, precision=8, separator=', ', suppress_small=True))
        with open(QTable_pass) as f:
            print(f.read())
        """
        行動空間
        0＝Aを優先
        1＝Bを優先
        2＝Cを優先
        3＝Dを優先
        4=何もしない

             |A|
           D  ×  B
             |C|
        """
    #行動選択
    def get_action(self, next_state, epsilon):
        if epsilon > np.random.uniform(0,1):
            next_action = np.random.choice(range(5))
        else:
            a = np.where(self._QTable[next_state]==self._QTable[next_state].max())[0]
            next_action = np.random.choice(a)
        return next_action
    #Q値更新
    def update_QTable(self, state, action, reward, next_state, episode):
        gamma = 0.99
        alpha = 0.1 
        next_maxQ = max(self._QTable[next_state])
        self._QTable[state, action] = (1 - alpha) * self._QTable[state, action] + alpha * (reward + gamma * next_maxQ)
        with open(QTable_pass, mode='w') as f:
            f.writelines(np.array2string(self._QTable, precision=8, separator=', ', suppress_small=True))
        return self._QTable


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


#Init
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


def is_priority(lane):
    if traci.lane.getEdgeID(lane) not in PRIORITY.values():
        return False
    return True

def get_state(nodeID):
    """
    状態空間を定義する関数
    (A,B,C,D)
    車両が存在すれば1,いなければ0
    (0,0,0,0)~(1,1,1,1) 16通り
    """
    # 各車線の状態を固定順序でバイナリ表現
    node_edges = []
    state = []
    for edge in net.getNode(nodeID).getIncoming():
        node_edges.append(edge.getID())
    for edge in node_edges:
        vehicles = traci.edge.getLastStepVehicleNumber(edge)
        state.append(1 if vehicles > 0 else 0)
    #print("state=", state)
    return int("".join(map(str, state)), 2)

def solve_deadlock(episode, nodeID, locked_vehicle, action): #nodeID=交差点
    control_obj = {}
    junction_edges = []
    for edge in net.getNode(nodeID).getIncoming():
        junction_edges.append(edge.getID())
    for edge in junction_edges:
        try:
            v = traci.edge.getLastStepVehicleIDs(edge)[-1]
            lane = traci.vehicle.getLaneID(v)
            distance = get_distacne(v, net)
            if distance < DISTANCE:
                control_obj[edge] = v
        except:
            pass
    if action != 4:
        #print("何もしない")  # next_action=0 の場合は変更なし
        PRIORITY[nodeID] = junction_edges[action]   
    # 優先車線を動的に設定
    
    for i, edge in enumerate(control_obj):
        vehicle = control_obj[edge]
        if PRIORITY[nodeID] == edge:
            traci.vehicle.setColor(vehicle, (255, 0, 0))
            traci.vehicle.setSpeed(vehicle, SPEED)
        else:
            traci.vehicle.setSpeed(vehicle, 0)
    return junction_edges
    
    


def init(is_gui):
    if is_gui:
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
    else:
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo')
    sumoCmd = [sumoBinary, "-c", "data/crossroads.sumocfg",]
    traci.start(sumoCmd)


#Start

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
PRIORITY = {}
PRIORITY_ARR = []

def set_simulation_time_limit(limit):
    global SIMULATION_TIME_LIMIT
    SIMULATION_TIME_LIMIT = limit
def calculate_reward(junction_edges, prev_deadlock, action, time):
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
    total_waiting_time = sum([traci.edge.getWaitingTime(edge) for edge in junction_edges])
    
    # 待ち時間をペナルティとして適用
    reward -= total_waiting_time * 0.01

    #時間経過ペナルティ
    #reward -= time

    # デッドロック報酬
    if deadlock_detected:
        reward -= 20
        #print("collision!")
    elif prev_deadlock and not deadlock_detected:
        reward += 50
    #テレポート報酬
    if teleport_occurs > 0:
        reward -= teleport_occurs * 50
    """
    #車線変更コスト報酬
    if action != 4:
        reward -= 1
    else:
        reward += 1
    """
    print("reward=", reward)
    return reward, deadlock_detected

def save_results(episode, reward, teleport_num, time):
    if not os.path.exists(Result_pass):
        with open(Result_pass, 'w') as f:
            f.write("Episode_num\tEpisode_reward\tTeleport_num\ttime\n")
    with open(Result_pass, 'a') as f:
        f.write(f"{episode}\t{reward}\t{teleport_num}\t{time}\n")

def simulation(num, with_program,episode_num):
    action_num =5
    state_num = 16
    waitingTime = {}
    if with_program:
        q_table = MyQTable(action_num,state_num) #引数は行動数
    #node = JUNCTION_NODE
    for episode in range(1,episode_num+1):
        if episode == 1 or episode == 300:
            init(True)
        else:
            init(False)
        if with_program:
            state = get_state(JUNCTION_NODE)
            prev_deadlock = 0
        teleportNum = 0
        time= 0
        episode_reward = 0
        for i in range(num):
            make_vehicle(f"vehicle_{i}", make_random_route(i), 0)
        while traci.simulation.getMinExpectedNumber() > 0 and SIMULATION_TIME_LIMIT > time:
            traci.simulationStep()
            teleported_vehicles = traci.simulation.getEndingTeleportIDList()
            for vehID in teleported_vehicles:
                traci.vehicle.setSpeed(vehID, SPEED)
            if with_program:
                
                locked_vehicle = {}
                collisions_arr = traci.simulation.getCollisions()
                if collisions_arr:
                    print(traci.simulation.getCollisions())
                #for index, node in enumerate(NODES):
                node = JUNCTION_NODE
                action = q_table.get_action(state, epsilon = 0.5 * (1 / (episode + 1)))
                junction_edges = solve_deadlock(episode, node, locked_vehicle, action)
                for edge in junction_edges:
                    waitingTime[edge] = waitingTime.get(edge, 0) + traci.edge.getWaitingTime(edge)
                reward, prev_deadlock = calculate_reward(junction_edges, prev_deadlock, action, time)
                #deadlockNum += 1
                next_state = get_state(node)
                q_table.update_QTable(state, action, reward, next_state, episode)
                state = next_state
                episode_reward += reward

            teleportNum += traci.simulation.getEndingTeleportNumber()
            time = traci.simulation.getTime()
        save_results(episode, episode_reward, teleportNum, time)
        print(f'Episode:{episode:4.0f},R:{episode_reward:4.0f},Teleport_num={teleportNum:4.0f},Endtime={time:4.0f}')
        for i in waitingTime:
            print(f'{i}_waitingtime={waitingTime[i]}')
        traci.close()
    if with_program:
        with open(QTable_pass) as f:
                print(f.read())
num_of_vehicles = int(input("Num of vehicles :"))
num_of_episode = int(input("Num of episode :"))
with_program = int(input("With Q-learning program (0=false, 1=true):"))
if with_program:
    use_Qtable = int(input("Use Q_table (0=false, 1=true)"))
limit = 3600
set_simulation_time_limit(limit)
simulation(num_of_vehicles, with_program,num_of_episode)
