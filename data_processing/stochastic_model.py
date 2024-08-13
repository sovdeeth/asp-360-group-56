# split world into grids
# for each point pair, find grid square a and b
# record transitions between squares to find probabilities

import csv
import random

class Grid:

    def __init__(self) -> None:
        self.grid = {}
        lat_start = 0.0
        lat_end = 90.0
        long_start = -110.0
        long_end = 15.0
        lat = lat_start
        while lat < lat_end:
            long = long_start
            while long < long_end:
                self.grid[(lat, long)] = {}
                self.grid[(lat, long)]["transitions"] = {}
                self.grid[(lat ,long)]["total"] = 0
                long += 0.25
            lat += 0.25

    def to_grid_coords(self, lat, long):
        return int(lat * 4) / 4, int(long * 4) / 4
    
    def add_point(self, lat_a, long_a, lat_b, long_b):
        grid_point_a = self.to_grid_coords(lat_a, long_a)
        grid_point_b = self.to_grid_coords(lat_b, long_b)

        # print("Adding {} -> {}".format(grid_point_a, grid_point_b))

        count = self.grid[grid_point_a]["transitions"].get(grid_point_b, 0)
        self.grid[grid_point_a]["transitions"][grid_point_b] = count + 1

        self.grid[grid_point_a]["total"] += 1

    def get_transitions(self, lat, long):
        grid_point = self.to_grid_coords(lat, long)
        chances = {}
        total = self.grid[grid_point]["total"]
        for target, count in self.grid[grid_point]["transitions"].items():
            chances[target] = count / total
        return chances
    
    # create a list of 6 transition points based on probs
    def create_model_data(self, lat, long):
        chances = self.get_transitions(lat, long)
        if len(chances) == 0:
            print("Error, no chances found for ({}, {})".format(lat, long))
            return [0] * 12
        for point, chance in chances.items():
            chances[point] = int(chance * 6 + 1)
        schances = sorted(chances)
        points = []
        failsafe = 0
        while len(points) < 12:
            failsafe += 1
            for point in schances:
                if chances[point] > 0:
                    chances[point] -= 1
                    points.extend([point[0], point[1]])
                if len(points) == 12:
                    break
            if (failsafe > 10):
                print("error: ({}, {}), {}, {}".format(lat, long, points, chances))
                return [0] * 12
        return points

def read_storms_in(grid):
    current_storm = ""
    with open("./data_processing/stripped_storms.csv") as input:
        csv_reader = csv.reader(input, delimiter=',')
        for row in csv_reader:
            if len(row) == 0:
                continue
            # new storm!
            if row[0] != current_storm:
                current_storm = row[0]
                prev_lat = float(row[2])
                prev_long = float(row[3])
                continue

            grid.add_point(prev_lat, prev_long, float(row[2]), float(row[3]))
            prev_lat = float(row[2])
            prev_long = float(row[3])

# grid = Grid()
# read_storms_in(grid)
# print(grid.get_transitions(27.5,-79))

def predict(grid, lat, long, depth = 0):
    chances = grid.get_transitions(lat, long)
    value = random.choices(list(chances.keys()), weights=chances.values())[0]

    new_lat = value[0]
    new_long = value[1]

    if depth > 0:
        print("Guessed ({}, {}), proceeding...".format(new_lat, new_long))
        return predict(grid, new_lat, new_long, depth - 1)
    return new_lat, new_long

# predict(grid, 27.5, -79)

# predict(grid, 27.5, -79, 5)



# create_model_data(grid, 28.5, -79)