import random
import numpy as np
import time
from datetime import datetime
from reinforcement_learning import DDPG


MAX_LASER_CURRENT = 5.5
MAX_LASER_FREQUENCY = 5000
MIN_LASER_CURRENT = 3
MIN_LASER_FREQUENCY = 600
STOP_LASER_CURRENT = 2.5
ACTION_DIM = 2  # rnn input size
STATE_DIM = 2

GOAL_STATE_ERROR = 0.01

class SerpenBot(object):
    def __init__(self):
        self.agent = DDPG(action_dim=ACTION_DIM, state_dim=STATE_DIM)

    def reward(self, goal_state, next_state):
        x_goal = goal_state[0]
        y_goal = goal_state[1]
        x_current = next_state[0]
        y_current = next_state[1]
        reward = -np.sqrt(np.square(x_current-x_goal)+np.square(y_current-y_goal))
        return reward

    def select_action(self, current_state):
        action = self.agent.select_action(current_state, add_noise=False)
        return action

    def learn(self):
        if self.agent.is_memory_full():
            print("learing")
            self.agent.learn()

    def store_experience(self, current_state, action, reward, next_state):
        self.agent.store_experience(current_state, action, reward, next_state)

    def is_done(self, goal_state, next_state):
        if np.sqrt(np.square(next_state[0]-goal_state[0])+np.square(next_state[1]-goal_state[1])) < GOAL_STATE_ERROR:
            return True
        return False

    def stop_robot(self):
        action = np.hstack((STOP_LASER_CURRENT, MIN_LASER_FREQUENCY))
        return action


    def save_agent(self):
        self.agent.save_actor()
        self.agent.save_critic()



serpenbot = SerpenBot()
current_state = 0
def init_control(init_state):
    global current_state
    current_state = np.array(init_state)

def init_control_labview(init_x, init_y):
    init_state = [init_x, init_y]
    init_control(init_state)


def training(goal_state, next_state):
    # Todo: return action (laser_current, laser_frequency)
    # Todo: if is_done(next_state), then turn off laser
    print(next_state)
    goal_state = np.array(goal_state)
    next_state = np.array(next_state)
    global current_state
    action = serpenbot.select_action(next_state)
    reward = serpenbot.reward(goal_state, next_state)
    print("reward:", reward)

    serpenbot.store_experience(current_state, action, reward, next_state)
    current_state = next_state
    serpenbot.learn()

    laser_current = np.round((MAX_LASER_CURRENT-MIN_LASER_CURRENT)*action[0]+MIN_LASER_CURRENT, 2)
    laser_frequency = np.round((MAX_LASER_FREQUENCY-MIN_LASER_FREQUENCY)*action[1]+MIN_LASER_FREQUENCY, 0)
    action = np.hstack((laser_current, laser_frequency))
    print("action:", action)

    return action

def training_labview(goal_x, goal_y, next_x, next_y):
    goal_state = [goal_x, goal_y]
    next_state = [next_x, next_y]
    return training(goal_state, next_state).tolist()

def finish_training():
    serpenbot.save_agent()

def control(current_state):
    # Todo: return action (laser_current, laser_frequency)
    action = serpenbot.select_action(np.array(current_state))
    laser_current = np.round((MAX_LASER_CURRENT - MIN_LASER_CURRENT) * action[0] + MIN_LASER_CURRENT, 2)
    laser_frequency = np.round((MAX_LASER_FREQUENCY - MIN_LASER_FREQUENCY) * action[1] + MIN_LASER_FREQUENCY, 0)
    action = np.hstack((laser_current, laser_frequency))
    return action

def control_labview(current_x, current_y):
    current_state = [current_x, current_y]
    return control(current_state).tolist()

def is_done(goal_state, next_state):
    return serpenbot.is_done(goal_state, next_state)

def is_done_labview(goal_x, goal_y, next_x, next_y):
    goal_state = [goal_x, goal_y]
    next_state = [next_x, next_y]
    return is_done(goal_state, next_state)

def stop_robot():
    return serpenbot.stop_robot()

if __name__ == '__main__':
    random.seed(datetime.now())
    x = random.uniform(-1, 1) * 16
    y = random.uniform(-1, 1) * 16

    robot_state = [x, y]
    init_control(robot_state)
    goal_x = 10
    goal_y = 10
    goal_state = [goal_x, goal_y]

    for i in range(10000):
        x = random.random() * 16
        y = random.random() * 16
        robot_state = [x, y]
        start = time.time()
        action = training(goal_state, robot_state)
        end = time.time()
        print("execution time:", end - start)

    finish_training()

