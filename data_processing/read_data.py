import csv
import datetime
import numpy as np

def dateFromMDH(month, day, hour):
    return (datetime.datetime(2001, int(month), int(day), 0).timetuple().tm_yday - 1) * 24 + int(hour)

# read file into list of points, omit status, category, diameter, etc.
def read_storm_data():
    stripped_data = []
    with open("./data_processing/storms.csv") as input:
        skipped_first = False
        csv_reader = csv.reader(input, delimiter=',')
        for row in csv_reader:
            if not skipped_first:
                skipped_first = True
                continue
            data_point = [row[1], dateFromMDH(row[3], row[4], row[5]), row[6], row[7], row[10]]
            stripped_data.append(data_point)
    
    with open("./data_processing/stripped_storms.csv", "w", newline='') as output:
        storm_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for point in stripped_data:
            storm_writer.writerow(point)

# categorizes into 1 of 4 buckets
# see https://medium.com/@kap923/hurricane-path-prediction-using-deep-learning-2f9fbb390f18 for source of idea
def categorize_location(lat, long):
    one_hots = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]
    bucket = 0
    if float(lat) > 20: 
        bucket |= 2
    
    if float(long) > 70:
        bucket |= 1
    
    return one_hots[bucket]

def flatten(xss):
    return [x for xs in xss for x in xs]

# turn list of points into segments, keep name for id purposes
# calculate starting quadrant
def segment_storm_data(input_length = 2):
    points_by_storm = {}
    current_storm = ""
    id = 0
    with open("./data_processing/stripped_storms.csv") as input:
        csv_reader = csv.reader(input, delimiter=',')
        header = True
        for row in csv_reader:
            if header is True:
                header = False
                continue
            if len(row) == 0:
                continue
            # new storm!
            if row[0] != current_storm:
                id += 1
                current_storm = row[0]
                storm_category = categorize_location(row[2], row[3])
            point = points_by_storm.setdefault(row[0] + "_" + str(id), [])
            row = row[1:]
            row.extend(storm_category)
            point.append(row)
    
    # split into segments
    segments = []
    for storm, storm_points in points_by_storm.items():
        length = len(storm_points)
        if (length < input_length):
            continue
        for i in range(0, length - input_length):
            if i % 1000 == 0:
                print(flatten(storm_points[i:i+input_length+1]))
            data = [storm]
            data.extend(flatten(storm_points[i:i+input_length+1]))
            segments.append(data)

    columns = ["name"]
    for i in range(input_length):
        columns.extend([(x + "_" + str(i)) for x in ["hour_of_year","lat","long","wind","region0","region1","region2","region3"]])

    with open("./data_processing/segmented_data_" + str(input_length) +"s_cat.csv", "w", newline='') as output:
        storm_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        storm_writer.writerow(columns)
        for point in segments:
            storm_writer.writerow(point)
    return

read_storm_data()
for i in range(2, 11):
    segment_storm_data(i)

