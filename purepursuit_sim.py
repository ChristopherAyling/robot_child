#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import sin, cos, degrees, radians, sqrt, atan2, pi, floor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from tqdm import tqdm
import random

from dataclasses import dataclass
# plt.style.use('seaborn')


# In[2]:


def draw_arena(ax):
    ax.plot(0, 0, 'go', markersize=(20))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))


# In[3]:


@dataclass
class Point:
    x: float
    y: float

    
@dataclass
class Robot:
    x: float = 0
    y: float = 0
    theta: float = 0 # radians
    wheel_raidus: float = 0.05
    wheel_distance: float = 0.1
    left_velocity: float = 0
    right_velocity: float = 0
    
    def step(self, dt):
        self.x += self.velocity * cos(self.theta) * dt
        self.y += self.velocity * sin(self.theta) * dt
        self.theta += self.angular_velocity * dt
        return self
        
    def draw(self, ax):
        ax.plot(self.x, self.y, marker=(3, 0, degrees(self.theta) + 90 + 180), color='r', markersize=10)
    
    
def get_distance(robot: Robot, target: Point) -> float: # bot, target
    error_x = target.x - robot.x
    error_y = target.y - robot.y
    return sqrt(error_x**2 + error_y**2)

def get_heading_error(robot: Robot, target: Point) -> float: # bot, target
    error_x = target.x - robot.x
    error_y = target.y - robot.y
    return atan2(error_y, error_x) - robot.theta

def wrap_to_pi(theta: float) -> float:
    if theta > pi:
        return theta - 2*pi
    if theta < -pi:
        return theta + 2*pi
    return theta
    
def clip(x, mn, mx):
    if x > mx:
        return mx
    if x < mn:
        return mn
    return x

@dataclass
class PurePursuit:
    """
    determine the request velocity and turn rate of the robot
    
    P controller for now
    """
    KVp: float = 0.1
    KHp: float = 5
    
    KVi: float = 1
    KHi: float = 1
    
    v_integral: float = 0
    a_integral: float = 0
    
    tolerance: float = 0.01 # 1cm
    
    def set_target(self, target):
        self.v_integral = 0
        self.a_integral = 0
        self.target = target
            
    def control(self, robot: Robot, dt: float) -> (float, float):
        distance_error = get_distance(robot, self.target)
        heading_error = wrap_to_pi(get_heading_error(robot, self.target))
        
        self.v_integral += dt * distance_error
        self.a_integral += dt * heading_error
        
        request_velocity = self.KVp * distance_error + self.KVi * self.v_integral
        request_angular_velocity = self.KHp * heading_error + self.KHi * self.a_integral
        
        # TODO clip
        request_velocity = clip(request_velocity, -0.5, 0.5)
        request_angular_velocity = clip(request_angular_velocity, -4, 4)
        
        return request_velocity, request_angular_velocity, distance_error, heading_error
    
    def is_complete(self, robot: Robot) -> bool:
        distance_error = get_distance(robot, self.target)
        return distance_error <= self.tolerance


# In[4]:


def simulate(robot: Robot, targets: [Point], controller: PurePursuit) -> int:
    dt = 1/60
    total_steps = 0
    positions = []
    for target in targets:
        controller.set_target(target)
        while True:
            robot.velocity, robot.angular_velocity, *_ = controller.control(robot, dt)
            robot.step(dt)
            positions.append((robot.x, robot.y, robot.theta))
            total_steps += 1
            
            if controller.is_complete(robot):
                break
                
            if total_steps >= 100_000:
                raise TimeoutError
    return total_steps, np.array(positions)


# In[5]:


def take_sample(n):
    # return tuple(random.choice(np.linspace(0.1, 10, 20)) for _ in range(n))
     return tuple(random.uniform(0, 2) for _ in range(n))

targets = [
    Point(0.4, 0.4),
    Point(-0.4, 0.4),
    Point(-0.2, -0.1),
    Point(-0.4, -0.4),
    Point(0.4, -0.4),
    Point(0, 0),
]

results = []
for _ in tqdm(range(200)):
    KVp, KHp, KVi, KHi = take_sample(4)
    robot = Robot()
    controller = PurePursuit(KVp=KVp, KHp=KHp, KVi=KVi, KHi=KHi)
    try:
        steps, positions = simulate(robot, targets, controller)
    except TimeoutError:
        continue
    result = {"KVp": KVp, "KHp": KHp, "KVi": KVi, "KHi": KHi, "steps": steps, "positions": positions}
    results.append(result)


# In[6]:


df = pd.DataFrame(results).sort_values("steps").reset_index()
df.head()


# In[7]:


sns.scatterplot(data=df, x="KVp", y="KHp", size="steps", hue="steps")


# In[8]:


sq = floor(sqrt(len(df)-1))
nrows = sq
ncols = sq

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(nrows*ncols):
    ax = axes[i]
    row = df.iloc[i]
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    for target in targets:
        ax.plot(target.x, target.y, 'g*', markersize=20)
    
    positions = row.positions
    ax.plot(positions[:, 0], positions[:, 1], '--', label=f"KVp={row.KVp:.2f} KHp={row.KHp:.2f} KVi={row.KVi:.2f} KHi={row.KHi:.2f}")
    ax.legend()
    ax.set_title(f"#{i} (steps={row.steps})")

    
plt.tight_layout()
plt.show()
plt.savefig("best100.png")


# In[9]:


fig, ax = plt.subplots(figsize=(20, 12))

for target in targets:
    ax.plot(target.x, target.y, 'b*', markersize=20)

for i, row in df[:5].iterrows():
    positions = row.positions
    ax.plot(positions[:, 0], positions[:, 1], '--', label=f"KVp={row.KVp:.2f} KHp={row.KHp:.2f} KVi={row.KVi:.2f} KHi={row.KHi:.2f}")
ax.legend()


# In[10]:


for i, row in df[-10:].iterrows():
    positions = row.positions
    plt.plot(positions[:, 0], positions[:, 1], '--', label=f"KVp={row.KVp:.2f} KHp={row.KHp:.2f}")
plt.legend()


# In[ ]:




