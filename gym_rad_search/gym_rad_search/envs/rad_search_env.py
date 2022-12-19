from dataclasses import dataclass, field, asdict
from email.policy import default
import os
import sys
import math
from matplotlib.markers import MarkerStyle
import matplotlib.collections as mcoll

import numpy as np
import numpy.typing as npt
import numpy.random as npr

import gym  # type: ignore
from gym import spaces  # type: ignore

import visilibity as vis  # type: ignore

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon as PolygonPatches

from typing import Any, List, Union, Literal, NewType, Optional, TypedDict, cast, get_args, Dict
from typing_extensions import TypeAlias


Point: TypeAlias = NewType("Point", tuple[float, float])
Polygon: TypeAlias = NewType("Polygon", list[Point])
Interval: TypeAlias = NewType("Interval", tuple[float, float])
BBox: TypeAlias = NewType("BBox", tuple[Point, Point, Point, Point])
Colorcode: TypeAlias = NewType("Colorcode", list[int])
Color: TypeAlias = NewType("Color", npt.NDArray[np.float64])

Metadata: TypeAlias = TypedDict(
    "Metadata", {"render.modes": list[str], "video.frames_per_second": int}
)

# BT
# These actions correspond to:
# -1: stay idle
# 0: left
# 1: up and left
# 2: up
# 3: up and right
# 4: right
# 5: down and right
# 6: down
# 7: down and left
Action: TypeAlias = Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7]
Directions: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7]

A_SIZE = len(get_args(Action))
DETECTABLE_DIRECTIONS = len(get_args(Directions)) # Ignores -1 idle state
FPS = 50
DET_STEP = 100.0  # detector step size at each timestep in cm/s
DET_STEP_FRAC = 71.0  # diagonal detector step size in cm/s
DIST_TH = 110.0  # Detector-obstruction range measurement threshold in cm
DIST_TH_FRAC = 78.0  # Diagonal detector-obstruction range measurement threshold in cm
EPSILON = 0.0000001
COLORS = [
    #Colorcode([148, 0, 211]), # Violet (Removed due to being too similar to indigo)
    Colorcode([255, 105, 180]), # Pink
    Colorcode([75, 0, 130]), # Indigo
    Colorcode([0, 0, 255]), # Blue
    Colorcode([0, 255, 0]), # Green
    Colorcode([255, 127, 0]) # Orange
    ]

def sum_p(p1: Point, p2: Point) -> Point:
    """
    Return the sum of the two points.
    """
    return Point((p1[0] + p2[0], p1[1] + p2[1]))


def sub_p(p1: Point, p2: Point) -> Point:
    """
    Return the difference of the two points.
    """
    return Point((p1[0] - p2[0], p1[1] - p2[1]))


def scale_p(p: Point, s: float) -> Point:
    """
    Return the scaled point p by the scalar s.
    """
    return Point((p[0] * s, p[1] * s))


def dist_p(p1: Point, p2: Point) -> float:
    """
    Return the distance between the two points.
    """
    def dist_sq_p(p1: Point, p2: Point) -> float:
        """
        Return the squared distance between the two points.
        """
        return float((p1[0] - p2[0]) ** 2) + float((p1[1] - p2[1]) ** 2)

    return math.sqrt(dist_sq_p(p1, p2))


def from_vis_p(p: vis.Point) -> Point:
    """
    Return a Point from a visilibity Point.
    """
    return Point((p.x(), p.y()))  # type: ignore


def to_vis_p(p: Point) -> vis.Point:
    """
    Return a visilibity Point from a Point.
    """
    return vis.Point(p[0], p[1])


def from_vis_poly(poly: vis.Polygon) -> Polygon:
    """
    Return a Polygon from a visilibity Polygon.
    """
    return Polygon(list(map(from_vis_p, poly)))


def to_vis_poly(poly: Polygon) -> vis.Polygon:
    """
    Return a visilibity Polygon from a Polygon.
    """
    return vis.Polygon(list(map(to_vis_p, poly)))


def count_matching_p(p1: Point, point_list: list[Point]) -> int:
    """
    Count number of times a Point appears in a list
    """
    count = 0
    for p2 in point_list:
        if p1[0] == p2[0] and p1[1] == p2[1]:
            count += 1
    return count    


# 0: (-1)*DET_STEP     *x, ( 0)*DET_STEP     *y
# 1: (-1)*DET_STEP_FRAC*x, (+1)*DET_STEP_FRAC*y
# 2: ( 0)*DET_STEP     *x, (+1)*DET_STEP     *y
# 3: (+1)*DET_STEP_FRAC*x, (+1)*DET_STEP_FRAC*y
# 4: (+1)*DET_STEP     *x, ( 0)*DET_STEP     *y
# 5: (+1)*DET_STEP_FRAC*x, (-1)*DET_STEP_FRAC*y
# 6: ( 0)*DET_STEP     *x, (-1)*DET_STEP     *y
# 7: (-1)*DET_STEP_FRAC*x, (-1)*DET_STEP_FRAC*y

# If action is odd, then we are moving on the diagonal and so our step size is smaller.
# Otherwise, we're moving solely in a cardinal direction.
def get_step_size(action: Action) -> float:
    """
    Return the step size for the given action.
    """
    return DET_STEP if action % 2 == 0 else DET_STEP_FRAC


# The signs of the y-coeffecients follow the signs of sin(pi * (1 - action/4))
def get_y_step_coeff(action: Action) -> int:
    return round(math.sin(math.pi * (1.0 - action / 4.0)))


# The signs of the x-coefficients follow the signs of cos(pi * (1 - action/4)) = sin(pi * (1 - (action + 6)/4))
def get_x_step_coeff(action: Action) -> int:
    return get_y_step_coeff((action + 6) % 8)


def get_step(action: Action) -> Point:
    """
    Return the step offset for the given action, scaled
        -1: stay idle 
        0: left
        1: up left
        2: up
        3: up right             
        4: right
        5: down right
        6: down
        7: down left
    """
    if action == -1:
        return Point((0.0, 0.0))
    else:
        return scale_p(
            Point((get_x_step_coeff(action), get_y_step_coeff(action))),
            get_step_size(action),
        )


def create_color(id: int) -> Color:
    ''' Pick initial Colorcode based on id number, then offset it '''
    specific_color: Colorcode = COLORS[id % (len(COLORS))]  # 
    if id > (len(COLORS)-1):
        offset: int = (id * 22) % 255  # Create large offset for that base color, bounded by 255
        specific_color[id % 3] = (255 + specific_color[id % 3] - offset) % 255  # Perform the offset
    return Color(np.array(specific_color) / 255)


@dataclass()
class StepResult():
    id: int = field(init=False)
    state: npt.NDArray[np.float32] = field(init=False)
    reward: float = field(init=False)
    done: bool = field(default=False)
    error: dict[Any, Any] = field(default_factory=dict)


@dataclass
class Agent():
    sp_dist: float = field(init=False) # Shortest path distance between agent and source
    euc_dist: float =  field(init=False) # Crow-Flies distance between agent and source
    det_coords: Point = field(init=False) # Detector Coordinates
    out_of_bounds: bool = field(init=False) # Artifact from rad_ppo; TODO remove from rad_ppo and have as a part of state return instead?
    out_of_bounds_count: int = field(init=False)  # Artifact - TODO decouple from rad_ppo agent?
    intersect: bool = field(default=False)  # Check if line of sight is blocked by obstacle 
    detector: vis.Point = field(init=False) # Visilibity graph detector coordinates 
    prev_det_dist: float = field(init=False)
    id: int = field(default=0)
    
    # Rendering
    marker_color: Color = field(init=False)
    det_sto: list[Point] = field(init=False)  # Coordinate history for episdoe
    meas_sto: list[float] = field(init=False) # Measurement history for episode
    reward_sto: list[float] = field(init=False) # Reward history for epsisode
    cum_reward_sto: list = field(init=False)  # Cumulative rewards tracker for episode

    def __post_init__(self):
        self.marker_color: Color = create_color(self.id)
        self.reset()
    
    def reset(self):
        self.out_of_bounds = False  
        self.out_of_bounds_count = 0
        self.det_sto: list[Point] = []  # Coordinate history for episdoe
        self.meas_sto: list[float] = [] # Measurement history for episode
        self.reward_sto: list[float] = [] # Reward history for epsisode
        self.cum_reward_sto: list = []  # Cumulative rewards tracker for episode
                

@dataclass
class RadSearch(gym.Env):
    """
        # bbox is the "bounding box"
        # Dimensions of radiation source search area in cm, decreased by observation_area param. to ensure visilibity graph setup is valid.
        #
        # observation_area
        # Interval for each obstruction area in cm aka observation area
        #
        # seed
        # A random number generator
        #
        # obstruct
        # Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7 obstructions
    """ 
    # Backwards compatiblility with single-agent step returns
    backwards_compatible: Union[Literal[1], None] = field(default=None)
    # Environment
    bbox: BBox = field(default_factory=lambda: BBox(
            tuple((Point((0.0, 0.0)), Point((2700.0, 0.0)), Point((2700.0, 2700.0)), Point((0.0, 2700.0))))
            ))
    observation_area: Interval = field(default_factory=lambda: Interval((200.0, 500.0)))
    np_random: npr.Generator = field(default_factory=lambda: npr.default_rng(0))
    obstruct: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7] = field(default=0)

    env_ls: list[Polygon] = field(init=False)
    max_dist: float = field(init=False)
    line_segs: list[list[vis.Line_Segment]] = field(init=False)
    poly: list[Polygon] = field(init=False)
    search_area: BBox = field(init=False)
    walls: Polygon = field(init=False)
    world: vis.Environment = field(init=False)
    vis_graph: vis.Visibility_Graph = field(init=False)
    intensity: int = field(init=False)
    bkg_intensity: int = field(init=False)
    obs_coord: list[list[Point]] = field(init=False)

    # Detector
    agents: dict[int, Agent] = field(init=False)
    
    # Source
    # TODO move into own class to easily handle multi-source
    src_coords: Point = field(init=False)
    source: vis.Point = field(init=False)

    # Values with default values which are not set in the constructor
    number_agents: int = 1
    action_space: spaces.Discrete = spaces.Discrete(A_SIZE)
    _max_episode_steps: int = 120
    background_radiation_bounds: Point = Point((10, 51))
    continuous: bool = False
    done: bool = False
    epoch_cnt: int = 0
    radiation_intensity_bounds: Point = Point((1e6, 10e6))
    metadata: Metadata = field(default_factory=lambda: {"render.modes": ["human"], "video.frames_per_second": FPS})  # type: ignore
    observation_space: spaces.Box = spaces.Box(0, np.inf, shape=(11,), dtype=np.float32)
    coord_noise: bool = False
    seed: Union[int, None] = field(default=None)  # TODO make env generation work with this
    
    # Rendering
    iter_count: int = field(default=0)   # For render function, believe it counts timesteps

    def __post_init__(self):
        self.search_area: BBox = BBox(
            (
                Point(
                    (
                        self.bbox[0][0] + self.observation_area[0],
                        self.bbox[0][1] + self.observation_area[0],
                    )
                ),
                Point(
                    (
                        self.bbox[1][0] - self.observation_area[1],
                        self.bbox[1][1] + self.observation_area[0],
                    )
                ),
                Point(
                    (
                        self.bbox[2][0] - self.observation_area[1],
                        self.bbox[2][1] - self.observation_area[1],
                    )
                ),
                Point(
                    (
                        self.bbox[3][0] + self.observation_area[0],
                        self.bbox[3][1] - self.observation_area[1],
                    )
                ),
            )
        )
        self.epoch_end = True
        self.agents = {i: Agent(id=i) for i in range(self.number_agents)}
        self.max_dist: float = dist_p(self.search_area[2], self.search_area[1])
        if self.seed != None:
            np.random.seed(self.seed) # TODO Fix to work with rng arg?
        self.reset()

    def step(
        self, action: Optional[Action] = None, action_list: Optional[dict] = None 
    ) -> dict[int, StepResult]:
        """
        Wrapper that captures gymAI env.step() and expands to include multiple agents for one "timestep". 
        Accepts literal action for single agent, or a dict of agent-IDs and actions.
        
        Returns dictionary of agent IDs and StepReturns. Agent coodinates are scaled for graph.
        
        Action:
        Literal single-action. Empty Action indicates agent is stalling for a timestep.
        
        action_list:
        A dict of agent_IDs and their corresponding Actions. If none passed in, this will return just agents current states (often used during a environment reset).
        
        """ 
        
        def agent_step(
            action: Optional[Action], agent: Agent, proposed_coordinates: list[Point] = []
        ) -> tuple[npt.NDArray[np.float32], float, bool, dict[Any, Any]]:
            """
            Method that takes an action and updates the detector position accordingly.
            Returns an observation, reward, and whether the termination criteria is met.
            
            Action:
            Single proposed action represented by a scalar value
            
            Agent:
            Agent to take the action
            
            Proposed Coordinates:
            A list of all resulting coordinates if all agents successfully take their actions. Used for collision prevention.
            """
            
            if self.take_action(agent, action, proposed_coordinates):
                # Check if out of bounds
                if (
                    agent.det_coords < self.search_area[0]
                    or self.search_area[2] < agent.det_coords
                ):
                    agent.out_of_bounds = True  
                    agent.out_of_bounds_count += 1 

                # Returns the length of a Polyline, which is a double
                # https://github.com/tsaoyu/PyVisiLibity/blob/80ce1356fa31c003e29467e6f08ffdfbd74db80f/visilibity.cpp#L1398
                agent.sp_dist: float = self.world.shortest_path(  # type: ignore
                    self.source, agent.detector, self.vis_graph, EPSILON
                ).length()
                agent.euc_dist = dist_p(agent.det_coords, self.src_coords)
                agent.intersect = self.is_intersect(agent)
                meas: float = self.np_random.poisson(
                    self.bkg_intensity
                    if agent.intersect
                    else self.intensity / agent.euc_dist + self.bkg_intensity
                )

                # Reward logic
                if agent.sp_dist < 110:
                    reward = 0.1
                    self.done = True
                elif agent.sp_dist < agent.prev_det_dist:
                    reward = 0.1
                    agent.prev_det_dist = agent.sp_dist
                else:
                    reward = -0.5 * agent.sp_dist / self.max_dist
            # If take_action is false, usually due to agent being in obstacle or empty action on env reset.
            else:
                # If detector starts on obs. edge, it won't have the shortest path distance calculated
                if self.iter_count > 0:
                    agent.euc_dist = dist_p(agent.det_coords, self.src_coords)
                    agent.sp_dist: float = self.world.shortest_path(  # type: ignore
                        self.source, agent.detector, self.vis_graph, EPSILON
                    ).length()
                    agent.intersect = self.is_intersect(agent)
                    meas: float = self.np_random.poisson(
                        self.bkg_intensity
                        if agent.intersect
                        else self.intensity / agent.euc_dist + self.bkg_intensity
                    )
                else:
                    agent.sp_dist = agent.prev_det_dist  # Set in reset function with current coordinates
                    agent.euc_dist = dist_p(agent.det_coords, self.src_coords)
                    agent.intersect = self.is_intersect(agent)
                    meas: float = self.np_random.poisson(
                        self.bkg_intensity
                        if agent.intersect
                        else self.intensity / agent.euc_dist + self.bkg_intensity
                    )

                reward = -0.5 * agent.sp_dist / self.max_dist

            # If detector coordinate noise is desired
            # TODO why is noise coordinate being added here? Why is noise a coordinate at all?
            noise: Point = Point(
                tuple(self.np_random.normal(scale=5, size=2))
                if self.coord_noise
                else (0.0, 0.0)
            )

            # Scale detector coordinates by search area of the DRL algorithm
            det_coord_scaled: Point = scale_p(
                sum_p(agent.det_coords, noise), 1 / self.search_area[2][1]
            )

            # Observation with the radiation meas., detector coords and detector-obstruction range meas.
            # TODO: State should really be better organized. If there are distinct components to it, why not make it
            # a named tuple?

            # Sensor measurement for in obstacles?
            sensor_meas: npt.NDArray[np.float64] = self.dist_sensors(agent=agent) if self.num_obs > 0 else np.zeros(DETECTABLE_DIRECTIONS)  # type: ignore
            # State is an 11-tuple ndarray
            state: npt.NDArray[np.float32] = np.array([meas, *det_coord_scaled, *sensor_meas])  # type: ignore
            agent.out_of_bounds = False  # Artifact - TODO decouple from rad_ppo agent?
            agent.det_sto.append(agent.det_coords)
            agent.meas_sto.append(meas)
            agent.reward_sto.append(reward)
            agent.cum_reward_sto.append(reward + agent.cum_reward_sto[-1] if len(agent.cum_reward_sto) > 0 else reward)
            return state, round(reward, 2), self.done, {'out_of_bounds': agent.out_of_bounds, 'out_of_bounds_count': agent.out_of_bounds_count}
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        aggregate_step_result: dict[int, StepResult] = {_: StepResult() for _ in self.agents}
        
        if action_list:
            tentative_coords = [sum_p(self.agents[agent_id].det_coords, get_step(action)) for agent_id, action in action_list.items()]
            for agent_id, action in action_list.items():
                aggregate_step_result[agent_id].id = agent_id
                (
                    aggregate_step_result[agent_id].state, 
                    aggregate_step_result[agent_id].reward, 
                    aggregate_step_result[agent_id].done,
                    aggregate_step_result[agent_id].error,
                ) = agent_step(agent=self.agents[agent_id], action=action, proposed_coordinates=tentative_coords)   
            self.iter_count += 1
            #return {k: asdict(v) for k, v in aggregate_step_result.items()}       
        else:
            # Provides backwards compatability for single actions instead of action lists for single agents.
            if action and len(self.agents) > 1:
                print("WARNING: Passing single action to mutliple agents during step.", file=sys.stderr)
            # Used during reset to get initial state or during single-agent move
            for agent_id, agent in self.agents.items():
                aggregate_step_result[agent_id].id = agent_id
                
                (
                    aggregate_step_result[agent_id].state, 
                    aggregate_step_result[agent_id].reward, 
                    aggregate_step_result[agent_id].done,
                    aggregate_step_result[agent_id].error,
                ) = agent_step(action=action, agent=agent)
            self.iter_count += 1
        return aggregate_step_result

    def reset(self) -> dict[int, StepResult]:
        """
        Method to reset the environment.
        """
        for agent in self.agents.values():
            agent.reset() 
            
        self.done = False
        self.iter_count = 0
        self.dwell_time = 1

        if self.epoch_end:
            if self.obstruct == -1:
                self.num_obs = self.np_random.integers(1, 6)  # type: ignore
            elif self.obstruct == 0:
                self.num_obs = 0
            else:
                self.num_obs = self.obstruct

            self.create_obs()
            self.walls = Polygon(list(self.bbox))
            self.env_ls: list[Polygon] = [self.walls, *self.poly]

            # Create Visilibity environment
            self.world = vis.Environment(list(map(to_vis_poly, self.env_ls)))

            # Create Visilibity graph to speed up shortest path computation
            self.vis_graph = vis.Visibility_Graph(self.world, EPSILON)
            self.epoch_cnt += 1
            self.epoch_end = False

        (
            self.source,
            detector_vis_start_location,
            detector_start_location,
            self.src_coords,
        ) = self.sample_source_loc_pos()
        
        for agent in self.agents.values():
            agent.detector = detector_vis_start_location 
            agent.det_coords = detector_start_location
            agent.prev_det_dist: float = self.world.shortest_path(  # type: ignore
                self.source, agent.detector, self.vis_graph, EPSILON
            ).length()
        
        self.intensity = self.np_random.integers(self.radiation_intensity_bounds[0], self.radiation_intensity_bounds[1])  # type: ignore
        self.bkg_intensity = self.np_random.integers(self.background_radiation_bounds[0], self.background_radiation_bounds[1])  # type: ignore

        # Check if the environment is valid
        if not (self.world.is_valid(EPSILON)):
            print("Environment is not valid, retrying!")
            self.epoch_end = True
            self.reset()

        # Get current states
        step = self.step(action=None, action_list=None)
        # Reclear iteration count 
        self.iter_count = 0
        return step

    def take_action(self, agent: Agent, action: Optional[Action], proposed_coordinates: list, agent_id: int = 0) -> bool:
        """
        Method that checks which direction to move the detector based on the action.
        If the action moves the detector into an obstruction, the detector position
        will be reset to the prior position.
        0: #left
        1: up left
        2: up
        3: up right             
        4: right
        5: down right
        6: down
        7: down left
        
        Return success of action: True if moved, false if stalled.
        """
        # Take no action
        if action is None:
            return False

        roll_back_action: bool = False
        step = get_step(action)
        tentative_coordinates = sum_p(agent.det_coords, step)
        
        # If proposed move will collide with another agents proposed move, 
        if count_matching_p(tentative_coordinates, proposed_coordinates) > 1:
            return False
        
        agent.detector = to_vis_p(tentative_coordinates)

        if self.in_obstruction(agent=agent):
            roll_back_action = True
                        
        if roll_back_action:
            # If we're in an obsticle, roll back
            agent.detector = to_vis_p(agent.det_coords)
        else:
            # If we're not in an obsticle, update the detector coordinates
            agent.det_coords = from_vis_p(agent.detector)

        return False if roll_back_action else True

    def create_obs(self) -> None:
        """
        Method that randomly samples obstruction coordinates from 90% of search area dimensions.
        Obstructions are not allowed to intersect.
        """
        ii = 0
        intersect = False
        self.obs_coord: list[list[Point]] = [[] for _ in range(self.num_obs)]
        self.poly: list[Polygon] = []
        self.line_segs: list[list[vis.Line_Segment]] = []
        obs_coord: list[Point] = []
        
        while ii < self.num_obs:
            seed_x: float = self.np_random.integers(  # type: ignore
                self.search_area[0][0], self.search_area[2][0] * 0.9  # type: ignore
            ).astype(np.float64)
            seed_y: float = self.np_random.integers(  # type: ignore
                self.search_area[0][1], self.search_area[2][1] * 0.9  # type: ignore
            ).astype(np.float64)
            ext_x: float = self.np_random.integers(  # type: ignore
                self.observation_area[0], self.observation_area[1]  # type: ignore
            ).astype(np.float64)
            ext_y: float = self.np_random.integers(  # type: ignore
                self.observation_area[0], self.observation_area[1]  # type: ignore
            ).astype(np.float64)
            obs_coord = [
                Point(t)
                for t in (
                    (seed_x, seed_y),
                    (seed_x, seed_y + ext_y),
                    (seed_x + ext_x, seed_y + ext_y),
                    (seed_x + ext_x, seed_y),
                )
            ]

            if ii > 0:
                kk = 0
                while not intersect and kk < ii:
                    intersect = math.isclose(vis.boundary_distance(to_vis_poly(Polygon(self.obs_coord[kk])), to_vis_poly(Polygon(obs_coord))), 0.0, abs_tol=EPSILON)  # type: ignore
                    if intersect:
                        obs_coord = []
                    kk += 1

            if not intersect:
                self.obs_coord[ii].extend(obs_coord)
                poly = Polygon(obs_coord)
                self.poly.append(poly)
                self.line_segs.append(
                    [
                        vis.Line_Segment(to_vis_p(p1), to_vis_p(p2))
                        for p1, p2 in (
                            (poly[0], poly[1]),
                            (poly[0], poly[3]),
                            (poly[2], poly[1]),
                            (poly[2], poly[3]),
                        )
                    ]
                )
                ii += 1
                intersect = False
                obs_coord = []
            intersect = False

    def sample_source_loc_pos(self,) -> tuple[vis.Point, vis.Point, Point, Point]:
        """
        Method that randomly generate the detector and source starting locations.
        Locations can not be inside obstructions and must be at least 1000 cm apart.
        Detectors all begin at same location.
        """
        det_clear = False
        src_clear = False
        resamp = False
        obstacle_index = 0

        def rand_point() -> Point:
            """
            Generate a random point within the search area.
            """
            return Point(
                tuple(
                    self.np_random.integers(  # type: ignore
                        int(self.search_area[0][0]), int(self.search_area[1][0]), size=2
                    ).astype(np.float64)
                )
            )

        # Generate initial point values
        source = rand_point()
        src_point = to_vis_p(source)
        
        detector = rand_point()
        
        det_point = to_vis_p(detector)

        # Check if detectors starting location is in an object
        while not det_clear:
            while not resamp and obstacle_index < self.num_obs:
                if det_point._in(to_vis_poly(self.poly[obstacle_index]), EPSILON):  # type: ignore
                    resamp = True
                obstacle_index += 1
            if resamp:
                detector = rand_point()
                det_point = to_vis_p(detector)
                obstacle_index = 0
                resamp = False
            else:
                det_clear = True
        
        # Check if source starting location is in object and is far enough away from detector
        # TODO change to multi-source
        resamp = False
        inter = False
        obstacle_index = 0
        num_retry = 0
        while not src_clear:
            while dist_p(detector, source) < 1000:
                source = rand_point()
            src_point = to_vis_p(source)
            L: vis.Line_Segment = vis.Line_Segment(det_point, src_point)
            while not resamp and obstacle_index < self.num_obs:
                poly_p: vis.Polygon = to_vis_poly(self.poly[obstacle_index])
                if src_point._in(poly_p, EPSILON):  # type: ignore
                    resamp = True
                if not resamp and vis.boundary_distance(L, poly_p) < 0.001:  # type: ignore
                    inter = True
                obstacle_index += 1
            if self.num_obs == 0 or (num_retry > 20 and not resamp):
                src_clear = True
            elif resamp or not inter:
                source = rand_point()
                src_point = to_vis_p(source)
                obstacle_index = 0
                resamp = False
                inter = False
                num_retry += 1
            elif inter:
                src_clear = True

        return src_point, det_point, detector, source

    def is_intersect(self, agent: Agent, threshold: float = 0.001) -> bool:
        """
        Method that checks if the line of sight is blocked by any obstructions in the environment.
        """
        inter = False
        kk = 0
        L = vis.Line_Segment(agent.detector, self.source)
        while not inter and kk < self.num_obs:
            if vis.boundary_distance(L, to_vis_poly(self.poly[kk])) < threshold and not math.isclose(  # type: ignore
                math.sqrt(agent.euc_dist), agent.sp_dist, abs_tol=0.1
            ):
                inter = True
            kk += 1
        return inter

    def in_obstruction(self, agent: Agent):
        """
        Method that checks if the detector position intersects or is inside an obstruction.
        """
        jj = 0
        obs_boundary = False
        while not obs_boundary and jj < self.num_obs:
            if agent.detector._in(to_vis_poly(self.poly[jj]), EPSILON):  # type: ignore 
                obs_boundary = True
            jj += 1

        if obs_boundary:
            bbox: vis.Bounding_Box = to_vis_poly(self.poly[jj - 1]).bbox()
            return all(
                [  # type: ignore
                    agent.detector.y() > bbox.y_min,
                    agent.detector.y() < bbox.y_max,
                    agent.detector.x() > bbox.x_min,
                    agent.detector.x() < bbox.x_max,
                ]
            )
        else:
            return False

    def collision_check(self, agent: Agent):
        """
        Method that checks if the new detector position will conflict with another detectors new position
        """        

    def dist_sensors(self, agent: Agent) -> list[float]:
        """
        Method that generates detector-obstruction range measurements with values between 0-1. 
        This detects obstructions within 1.1m of itself. 0 means no obstructions were detected.
        Currently supports 8 directions
        """
        detector_p: Point = from_vis_p(agent.detector)
        
        segs: list[vis.Line_Segment] = [
            vis.Line_Segment(
                agent.detector, to_vis_p(sum_p(detector_p, get_step(action)))
            )
            for action in cast(tuple[Directions], get_args(Directions))
        ]
        # TODO: Currently there are only eight actions -- what happens if we change that?
        # This annotation would need to change as well.
        dists: list[float] = [0.0] * len(segs)  # Directions where an obstacle is detected
        obs_idx_ls: list[int] = [0] * len(self.poly)  # Keeps track of how many steps will interect with which obstacle
        inter = 0  # Intersect flag
        seg_dist: list[float] = [0.0] * 4  # TODO what is the purpose of this? Saves into dists, appears to be the max "distance", but only tracks intersects?
        if self.num_obs > 0:
            for idx, seg in enumerate(segs): # TODO change seg to direction_segment
                for obs_idx, poly in enumerate(self.line_segs): # TODO change poly to obstacle
                    for seg_idx, obs_seg in enumerate(poly):  # TODO change obs_seg to obstacle_line_segment
                        if inter < 2 and vis.intersect(obs_seg, seg, EPSILON):  # type: ignore 
                            # check if step dir intersects poly seg
                            obstacle_distance = vis.distance(seg.first(), obs_seg)
                            line_distance = (DIST_TH - obstacle_distance) / DIST_TH # type: ignore 
                            seg_dist[seg_idx] = line_distance
                            inter += 1
                            obs_idx_ls[obs_idx] += 1 
                    if inter > 0:
                        max_distance = max(seg_dist)
                        if max_distance > dists[idx]:
                            dists[idx] = max_distance 
                        seg_dist = [0.0] * 4
                inter = 0
            # If there are more than three dists equal to one, we need to correct the coordinates. 
            #   Inter only triggers if inter is less than two, however there can be up to 5 intersections ... TODO
            if sum(filter(lambda x: x == 1.0, dists)) > 3:
                # Take the polygon which corresponds to the index with the maximum number of intersections.
                argmax = max(zip(obs_idx_ls, self.poly))[1]  # Gets the line coordinates for the obstacle that is intersecting TODO rename!
                dists = self.correct_coords(poly=argmax, agent=agent)
        
        assert len(dists) == DETECTABLE_DIRECTIONS  # Sanity check - if this is wrong it will mess up the step return shape of "state" and make training fail

        return dists

    def correct_coords(self, poly: Polygon, agent: Agent) -> list[float]:
        """
        Method that corrects the detector-obstruction range measurement if more than the correct
        number of directions are being activated due to the Visilibity implementation.
        This often happens when an agent is on the edge of an obstruction.
        """
        x_check: list[bool] = [False] * DETECTABLE_DIRECTIONS
        dist = 0.1  # TODO Scaled?
        length = 1
        poly_p: vis.Polygon = to_vis_poly(poly)

        qs: list[Point] = [from_vis_p(agent.detector)] * DETECTABLE_DIRECTIONS  # Offsets agent position by 0.1 to see if actually inside obstacle
        dists: list[float] = [0.0] * DETECTABLE_DIRECTIONS
        while not any(x_check):
            for action in cast(tuple[Directions], get_args(Directions)):
                # Gets slight offset to remove effects of being "on" an obstruction
                step = scale_p(
                    Point((get_x_step_coeff(action), get_y_step_coeff(action))),
                    dist * length,
                )
                qs[action] = sum_p(qs[action], step)
                if to_vis_p(qs[action])._in(poly_p, EPSILON):  # type: ignore
                    x_check[action] = True

        # i.e. if one outside the poly then
        if sum(x_check) >= 4:
            for ii in [0, 2, 4, 6]:
                if x_check[ii - 1] == True and x_check[ii + 1] == True:
                    dists[ii] = 1.0
                    dists[ii-1] = 1.0
                    dists[ii+1] = 1.0
                    #dists[ii - 1 : ii + 2] = [1.0, 1.0, 1.0]  # This causes there to be 11 elements when there should only be 8
                    
        assert len(dists) == DETECTABLE_DIRECTIONS  # Sanity check - if this is wrong it will mess up the step return shape of "state" and make training fail
        return dists

    def render(
        self,
        save_gif: bool = False,
        path: Optional[str] = None,
        epoch_count: Optional[int] = None,
        just_env: Optional[bool] = False,
        obstacles=[],
        episode_rewards={},
        data=[],
        measurements: Optional[list[float]] = None,
        params=[],
        location_estimate=None,
    ):
        """
        Method that produces a gif of the agent interacting in the environment. Only renders one episode at a time.
        """       
        reward_length = field(init=False) 
        # global location_estimate 
        # location_estimate = None # TODO Trying to get out of global scope; this is for source prediction

        def update(
            frame_number: int, 
            #data: list, 
            ax1: plt.Axes, 
            ax2: plt.Axes, 
            ax3: plt.Axes, 
            src: Point, 
            area_dim: BBox, 
            measurements: list,
            flattened_rewards: list
            ) -> None:
            """
            Renders each frame
            
            data:
            From detector storage - agent location
            TODO get rid of
            
            ax1:
            Actual grid
            
            ax2:
            Radiation counts
            
            ax3:
            Rewards
            
            src:
            Source coordinates
            
            area_dim:
            BBox - size of grid
            
            measurements:
            From detector storage - intensity readings
            TODO get rid of
            
            location_estimate
            TODO fix
            PathCollection variable from matplotlib for holding plot points

            """
            print(f"Current Frame: {frame_number}", end='\r') # Acts as a progress bar
            
            current_index = frame_number % (self.iter_count)
            # global location_estimate # TODO Trying to get out of global scope; this is for source prediction
            
            # Set up graphs for first frame
            if current_index == 0:
                intensity_sci = "{:.2e}".format(self.intensity)
                ax1.cla()
                ax1.set_title(
                    "Activity: "
                    + intensity_sci
                    + " [gps] Bkg: "
                    + str(self.bkg_intensity)
                    + " [cps]"
                )

                # Plot source
                ax1.scatter(
                    src[0] / 100,
                    src[1] / 100,
                    marker_size,
                    c="red",
                    marker=MarkerStyle("*"),
                    label="Source",
                )
                    
                for agent_id, agent in self.agents.items():
                    data = np.array(agent.det_sto[current_index]) / 100 
                    data_sub = (np.array(agent.det_sto[current_index + 1]) / 100)- (np.array(agent.det_sto[current_index]) / 100)
                    orient = math.degrees(math.atan2(data_sub[1], data_sub[0]))
                    ax1.scatter(
                        data[0],
                        data[1],
                        marker_size,
                        c=[agent.marker_color],
                        marker=MarkerStyle((3, 0, orient - 90)),
                    )
                    ax1.scatter(
                        -1000, -1000, marker_size, c=[agent.marker_color], marker=MarkerStyle("^"), label=f"{agent_id}_Detector"
                    )
                
                # Plot Obstacles
                ax1.grid()
                if not (obstacles == []) and obstacles != None:
                    for coord in obstacles:
                        #p_disp = PolygonPatches(coord[0] / 100, color="gray")
                        p_disp = PolygonPatches(np.array(coord) / 100, color="gray")
                        ax1.add_patch(p_disp)
                        
                # Plot location prediction
                # TODO make multi-agent and fix
                # if not (location_estimate is None):
                #     location_estimate = ax1.scatter(
                #         location_estimate[0][current_index][1] / 100,
                #         location_estimate[0][current_index][2] / 100,
                #         marker_size,
                #         c="magenta",
                #         label="Loc. Pred.",
                #     )
                    
                # Finish setting up grids
                ax1.set_xlim(0, area_dim[1][0] / 100)
                ax1.set_ylim(0, area_dim[2][1] / 100)
                ax1.set_xticks(np.linspace(0, area_dim[1][0] / 100 - 2, 5))
                ax1.set_yticks(np.linspace(0, area_dim[1][0] / 100 - 2, 5))
                ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                ax1.yaxis.set_major_formatter(FormatStrFormatter("%d"))
                ax1.set_xlabel("X[m]")
                ax1.set_ylabel("Y[m]")
                ax1.legend(loc="lower left", fontsize=8)

                # Set up radiation graph
                # TODO make this less terrible
                count_max: float = 0.0
                for agent in self.agents.values():
                    for measurement in agent.meas_sto:
                        if count_max < measurement:
                            count_max = measurement
                ax2.cla()
                ax2.set_xlim(0, self.iter_count)
                ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                ax2.set_ylim(0, count_max)
                ax2.set_xlabel("n")
                ax2.set_ylabel("Counts")                
                for agent_id, agent in self.agents.items():
                    markerline, _, _ = ax2.stem([0], [agent.meas_sto[0]], use_line_collection=True, label=f"{agent_id}_Detector")
                    current_color = tuple(agent.marker_color)
                    markerline.set_markerfacecolor(current_color)
                    markerline.set_markeredgecolor(current_color)
                
                # Set up rewards graph
                #flattened_rewards = [x for v in episode_rewards.values() for x in v]        
                ax3.cla()
                ax3.set_xlim(0, self.iter_count)
                ax3.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                ax3.set_ylim(min(flattened_rewards) - 0.5, max(flattened_rewards) + 0.5)
                ax3.set_xlabel("n")
                ax3.set_ylabel("Cumulative Reward")
                ax3.plot()
                    
            else: # If not first frame
                # if location_estimate:
                #     location_estimate.remove()
                    
                for agent_id, agent in self.agents.items():
                    data = np.array(agent.det_sto[current_index]) / 100
                    # If not last step, adjust orientation
                    if current_index != len(agent.det_sto)-1:
                        data_sub = (np.array(agent.det_sto[current_index + 1]) / 100)- (np.array(agent.det_sto[current_index]) / 100)
                        orient = math.degrees(math.atan2(data_sub[1], data_sub[0]))
                        # Plot detector
                        ax1.scatter(
                            data[0],
                            data[1],
                            marker_size,
                            c=[agent.marker_color],
                            marker=MarkerStyle((3, 0, orient - 90)),
                        )
                    else:
                        ax1.scatter(
                            data[0],
                            data[1],
                            marker_size,
                            c=[agent.marker_color],
                            marker=MarkerStyle((3, 0)),
                        )                        
                    # TODO What is this doing?
                    ax1.scatter(
                        -1000, -1000, marker_size, [agent.marker_color], marker=MarkerStyle("^"), label=f"{agent_id}_Detector"
                    )
                    # Plot detector path if not last step
                    if current_index != len(agent.det_sto)-1:
                        data_prev: npt.NDArray[np.float64] = np.array(agent.det_sto[current_index-1]) / 100
                        data_current: npt.NDArray[np.float64] = np.array(agent.det_sto[current_index]) / 100
                        data_next: npt.NDArray[np.float64] = np.array(agent.det_sto[current_index+1]) / 100
                        line_data: npt.NDArray[np.float64] = np.array([data_prev, data_current, data_next])
                        ax1.plot(
                            line_data[0 : 2, 0],
                            line_data[0 : 2, 1],
                            3,
                            c=agent.marker_color,
                            alpha=0.3,
                            ls="--",
                        )
                    # Plot radiation counts - stem graph
                    current_color = tuple(agent.marker_color)
                    markerline, _, _ = ax2.stem(
                        [current_index], [agent.meas_sto[current_index]], use_line_collection=True, label=f"{agent_id}_Detector"
                        )
                    markerline.set_markerfacecolor(current_color)
                    markerline.set_markeredgecolor(current_color)
                    
                    # Plot rewards graph - line graph, previous reading connects to current reading   
                    ax3.scatter(current_index, agent.reward_sto[current_index], marker=',', c=[agent.marker_color], s=2, label=f"{agent_id}_Detector") # Current state reward              
                    ax3.plot([current_index-1, current_index], agent.cum_reward_sto[current_index-1:current_index+1], c=agent.marker_color, label=f"{agent_id}_Detector")  # Cumulative line graph
                        
                
                # TODO make multi-agent and fix
                # if not (location_estimate is None):
                #     location_estimate = ax1.scatter(
                #         location_estimate[0][current_index][1] / 100,
                #         location_estimate[0][current_index][2] / 100,
                #         marker_size * 0.8,
                #         c="magenta",
                #         label="Loc. Pred.",
                #     )

        # Initialize render environment 
        if data or measurements:
            print(f"Error: Not implemented for upgraded multi-agent version. TBD")
            return
            # TODO Dont overwrite! Make local var instead
            # TODO update to work with new mulit-agent framework
            # self.intensity = params[0]
            # self.bkg_intensity = params[1]
            # self.src_coords = params[2]
            # self.iter_count = len(measurements)
            # data = np.array(data) / 100
        #else:
            # TODO change to multi-agent
            #data = np.array(agent.det_sto) / 100  # Detector stored locations in an array?
            #measurements = agent.meas_sto # Unneeded?

        # Check only rendering one episode aka data readings available match number of rewards (+1 as rewards dont include the first position). 
        #if data.shape[0] != len(episode_rewards)+1:
        
        # TODO make multiagent
        if episode_rewards:
            print(f"Error: Episode rewards are deprecated. Rendering plots from existing agent storage.")
        
        cum_episode_rewards = [a.cum_reward_sto for a in self.agents.values()]
        flattened_rewards = [x for v in cum_episode_rewards for x in v]   
        data_length = len(self.agents[0].det_sto)
        reward_length = len(cum_episode_rewards[0]) if len(cum_episode_rewards) > 0 else 0
        if data_length != reward_length:
            print(f"Error: episode reward array length: {reward_length} does not match existing detector locations array length {data_length}. \
            Check: Are you trying to render more than one episode?")
            return

        if obstacles == []:
            obstacles = self.obs_coord

        # Check only rendering one episode aka data readings available match number of rewards 
        # (+1 as rewards dont include the first position). 
        reward_length = len(ep_rew)
        if data.shape[0] != len(ep_rew)+1:
            print(f"Error: episode reward array length: {reward_length} does not match existing detector locations array length, \
            minus initial start position: {data.shape[0]}. \
            Check: Are you trying to render more than one episode?")
            return 1

        if just_env:
            # Setup Graph
            plt.rc("font", size=12)
            fig, ax1 = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
            
            # Plot source
            ax1.scatter(
                self.src_coords[0] / 100,
                self.src_coords[1] / 100,
                60,
                c="red",
                marker=MarkerStyle("*"),
                label="Source",
            )
            # Plot Agents
            for agent_id, agent in self.agents.items():
                data = np.array(agent.det_sto[0]) / 100 # TODO make just a single op instead of whole array
                ax1.scatter(
                    data[0],
                    data[1],
                    42,
                    c=[agent.marker_color],
                    #c="black",
                    marker=MarkerStyle("^"),
                    label=f"{agent_id}_Detector",
                )
            # Plot Obstacles
            ax1.grid()
            if not (obstacles == []):
                for coord in obstacles:
                    p_disp = PolygonPatches((np.array(coord) / 100), color="gray")
                    ax1.add_patch(p_disp)

            # TODO Make multi-agent and fix
            # if not (location_estimate is None):
            #     ax1.scatter(
            #         # location_estimate[0][current_index][1] / 100,
            #         # location_estimate[0][current_index][2] / 100,
            #         location_estimate[0][0][1] / 100,
            #         location_estimate[0][0][2] / 100,
            #         42,
            #         c="magenta",
            #         label="Loc. Pred.",
            #     )
            # Finish Graph
            ax1.set_xlim(0, self.search_area[1][0] / 100)
            ax1.set_ylim(0, self.search_area[2][1] / 100)
            ax1.set_xlabel("X[m]")
            ax1.set_ylabel("Y[m]")
            ax1.legend(loc="lower left", fontsize=8)
        
            # Save
            if save_gif:
                if os.path.isdir(str(path) + "/gifs/"):
                    fig.savefig(str(path) + f"/gifs/environment.png")
                else:
                    os.mkdir(str(path) + "/gifs/")
                    fig.savefig(str(path) + f"/gifs/environment.png")
            else:
                plt.show()
            # Figure is not reused, ok to close 
            plt.close(fig)
            return
            
        else:
            # Setup Graph for gif
            plt.rc("font", size=12)
            fig, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=(15, 5), tight_layout=True
            )
            marker_size = 25

            # Setup animation
            print("Frames to render ", reward_length)

            ani = animation.FuncAnimation(
                fig,
                update,
                #frames=reward_length,
                frames=data_length,
                fargs=(ax1, ax2, ax3, self.src_coords, self.search_area, measurements, flattened_rewards),
            )
            if save_gif:
                writer = PillowWriter(fps=5)
                if os.path.isdir(str(path) + "/gifs/"):
                    ani.save(str(path) + f"/gifs/test_{epoch_count}.gif", writer=writer)
                    print("")
                else:
                    os.mkdir(str(path) + "/gifs/")
                    ani.save(str(path) + f"/gifs/test_{epoch_count}.gif", writer=writer)
            else:
                plt.show()
            return

# TODO make multi-agent
    def FIM_step(self, agent: Agent, action: Action, coords: Optional[Point] = None) -> Point:

        """
        Method for the information-driven controller to update detector coordinates in the environment
        without changing the actual detector positon.

        Args:
        action : action to move the detector
        coords : coordinates to move the detector from that are different from the current detector coordinates
        """

        # Make a copy of the current detector coordinates
        detector_coordinates = agent.det_coords # TODO make multi-agent
        det_coords = detector_coordinates
        if coords:
            coords_p: vis.Point = to_vis_p(coords)
            agent.detector = coords_p
            agent.det_coords = coords # TODO make multi-agent

        in_obs = False if self.take_action(agent, action, proposed_coordinates=[]) else True
        detector_coordinates = agent.det_coords # TODO make multi-agent
        det_ret = detector_coordinates
        if coords is None and not in_obs or coords:
            # If successful movement, return new coords. Set detector back.
            det_coords_p: vis.Point = to_vis_p(det_coords)
            agent.detector = det_coords_p
            agent.det_coords = det_coords # TODO make multi-agent

        return det_ret