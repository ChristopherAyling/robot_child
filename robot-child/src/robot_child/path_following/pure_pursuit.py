from dataclasses import dataclass

from math import sin, cos, sqrt, atan2, pi


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Robot:
    x: float = 0
    y: float = 0
    theta: float = 0  # radians
    # wheel_raidus: float = 0.05
    # wheel_distance: float = 0.1
    # left_velocity: float = 0
    # right_velocity: float = 0
    velocity: float = 0
    angular_velocity: float = 0

    def step(self, dt):
        self.x += self.velocity * cos(self.theta) * dt
        self.y += self.velocity * sin(self.theta) * dt
        self.theta += self.angular_velocity * dt
        return self


def get_distance(robot: Robot, target: Point) -> float:  # bot, target
    error_x = target.x - robot.x
    error_y = target.y - robot.y
    return sqrt(error_x**2 + error_y**2)


def get_heading_error(robot: Robot, target: Point) -> float:  # bot, target
    error_x = target.x - robot.x
    error_y = target.y - robot.y
    return atan2(error_y, error_x) - robot.theta


def wrap_to_pi(theta: float) -> float:
    if theta > pi:
        return theta - 2 * pi
    if theta < -pi:
        return theta + 2 * pi
    return theta


def clip(x, mn, mx):
    if x > mx:
        return mx
    if x < mn:
        return mn
    return x


MAX_SPEED_CM_S = 20


@dataclass
class PointController:
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

    tolerance: float = 0.01  # 1cm
    target: Point | None = None

    def set_target(self, target: Point):
        self.v_integral = 0
        self.a_integral = 0
        self.target = target

    def control(self, robot: Robot, dt: float) -> tuple[float, float]:
        if self.is_complete(robot):
            return 0, 0

        distance_error = get_distance(robot, self.target)
        heading_error = wrap_to_pi(get_heading_error(robot, self.target))

        self.v_integral += dt * distance_error
        self.a_integral += dt * heading_error

        request_velocity = self.KVp * distance_error + self.KVi * self.v_integral
        request_angular_velocity = self.KHp * heading_error + self.KHi * self.a_integral

        request_velocity = clip(request_velocity, -MAX_SPEED_CM_S, MAX_SPEED_CM_S)
        request_angular_velocity = clip(request_angular_velocity, -4, 4)

        return request_velocity, request_angular_velocity

    def is_complete(self, robot: Robot) -> bool:
        distance_error = get_distance(robot, self.target)
        return distance_error <= self.tolerance


@dataclass
class PurePursuitController:
    point_controller: PointController
    points: list[Point]
    i: int = 0

    def control(self, robot: Robot, dt: float) -> tuple[float, float]:
        if self.point_controller.target is None:
            self.point_controller.set_target(self.points[self.i])
        if self.point_controller.is_complete(robot):
            self.i += 1
            if self.i == len(self.points):
                return 0, 0
            self.point_controller.set_target(self.points[self.i])
        return self.point_controller.control(robot, dt)

    def is_complete(self, robot: Robot) -> bool:
        return self.i == len(self.points)
