# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)

import math
import random
import sys
import os

import pygame
import numpy as np
import time

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 30
CAR_SIZE_Y = 30

BORDER_COLOR = (0, 0, 0, 255)  # Color To Crash on Hit
OBJ_COLOR = (45, 157, 10, 255)
current_generation = 0  # Generation counter
pygame.init()
bck_driver = pygame.display.get_driver()
pygame.quit()
class Car:

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load('utils/utils_9/car.png').convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # self.position = [690, 740] # Starting Position
        self.position = [200, 900]  # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.collision = False  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed
        self.times = 0.

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.collision = False
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if 0<= point[0] < WIDTH and 0<=point[1] <HEIGHT:
                if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                    self.collision = True

                    break
            else:
                self.collision = True
                break

    def check_radar(self, degree, game_map):

        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        #print(x)
        #print(y)
        #print(game_map.get_size())
        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further

        # s
        color = game_map.get_at((x, y))
        cosl = math.cos(0.0174533*(360 - (self.angle + degree)))
        senl = math.sin(0.0174533*(360 - (self.angle + degree)))
        while not (color == BORDER_COLOR or color == OBJ_COLOR) and length <100:
            length = length + 1
            ll = time.time()
            x = int(self.center[0] + cosl * length)
            self.times += time.time() - ll
            y = int(self.center[1] + senl * length)
            color = game_map.get_at((x, y))


        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])


    def _inner_update(self, game_map):
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)

        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 120)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 90)
        self.position[1] = min(self.position[1], WIDTH - 90)

        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]



        # Check Collisions And Clear Radars

        self.check_collision(game_map)
        self.radars.clear()
        #ll = time.time()
        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(0, 360, 90):
            self.check_radar(d, game_map)
        #self.times = time.time() - ll

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down

        if not self.collision:
            self._inner_update(game_map)

        else: #calculate new position if blocked not update else update
            self.times =0
            newPosition = [0,0]
            newPosition[0] += int(math.cos(math.radians(360 - self.angle)) * self.speed)
            newPosition[0] = max(self.position[0], 20)
            newPosition[0] = min(self.position[0], WIDTH - 120)
            # Same For Y-Position
            newPosition[1] += int(math.sin(math.radians(360 - self.angle)) * self.speed)
            newPosition[1] = max(self.position[1], 90)
            newPosition[1] = min(self.position[1], HEIGHT - 90)
            x = int(newPosition[0])
            y = int(newPosition[1])

            if not game_map.get_at((x,y)) == BORDER_COLOR:
                self._inner_update(game_map)

        #print(self.times[-1])

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = radar[1] / 100

        return return_values

    def is_collided(self):
        # Basic Alive Function
        return self.collision

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


def run_simulation(agents, map ="white", render = False):


    # Initialize PyGame And The Display
    init = time.time()
    if not render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    else:
        os.environ["SDL_VIDEODRIVER"] = bck_driver
        if os.environ["SDL_VIDEODRIVER"] == "dummy":
            os.environ.pop("SDL_VIDEODRIVER", None)
        
    pygame.init()
    screen = None
    if render:
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN|pygame.SCALED)
        
    else:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))


    # Clock Settings
    # Font Settings & Loading Map

    clock = pygame.time.Clock()

    clock.tick(10 if render else 100000)
    game_map = pygame.image.load(map).convert()  # Convert Speeds Up A Lot
    #print("Init "+str(time.time()-init))
    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0
    cars = [Car() for car in range(len(agents))]
    fitness = 0
    obs = [[] for i in range(len(agents))]
    ss = time.time()
    #print("ffff "+str(len(cars)))
    ups = []
    timess = [[], [], []]
    for j in range(250):
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        #acts = time.time()
        for i in range(len(agents)):
            #print("cars "+str(i)+"  "+str(cars[i]))
            outtime = time.time()
            ob  = cars[i].get_data()
            pos = [cars[i].center[0]/WIDTH, cars[i].center[1]/HEIGHT]
            distance_norm = math.sqrt(((1730/WIDTH)-pos[0])**2 + ((165/HEIGHT)-pos[1])**2)
            bearing = np.arctan2(165 - cars[i].center[1], 1730 - cars[i].center[0])/np.pi
            
            ob = [distance_norm, bearing, *ob]
            obs[i].append(ob)

            s = time.time()
            output = agents[i].activate(ob)

            choice = np.argmax(output)#random.randint(0,4)#
            if choice == 0:
                cars[i].angle += 5  # Left
            elif choice == 1:
                cars[i].angle -= 5  # Right
            elif choice == 2:
                cars[i].speed = -10  # Slow Down
            elif choice == 3:
                cars[i].speed = 10  # Speed Up
            else:
                cars[i].speed = 0 # stop
            timess[0].append(time.time()-outtime)
            # Check If Car Is Still Alive
            # Increase Fitness If Yes And Break Loop If Not
            updtime = time.time()
            cars[i].update(game_map)
            timess[1].append(time.time()-updtime)

            ups.append(cars[i].times)
            cars[i].times = 0
            #fitness[i] +=  cars[i].get_reward()


        #print("acts "+str(time.time()-acts))
        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        drawtime =time.time()
        for i in range(len(agents)):
            cars[i].draw(screen)
        timess[2].append(time.time() - drawtime)
        pygame.display.flip()
        clock.tick(24 if render else 100000)
    # print("actions time " + str(sum(timess[0])))
    # print("update time " + str(sum(timess[1])))
    # print("draws time " + str(sum(timess[2])))
    # print("ups "+str(sum(ups))+"   "+str(len(ups)))
    # print("  alla cosa "+str(time.time()-ss))

    pygame.display.quit()
    dis = [math.sqrt((1730-cars[i].center[0])**2 + (165-cars[i].center[1])**2) for i in range(len(agents))]
    #print(dis)
    return dis, obs
