import csv
import datetime

def dateFromMDH(month, day, hour):
    return (datetime.datetime(2001, int(month), int(day), 0).timetuple().tm_yday - 1) * 24 + int(hour)

# read file into list of points, omit status, category, diameter, etc.
def read_storm_data():
    stripped_data = []
    with open("./storms.csv") as input:
        skipped_first = False
        csv_reader = csv.reader(input, delimiter=',')
        for row in csv_reader:
            if not skipped_first:
                skipped_first = True
                continue
            data_point = [row[1], dateFromMDH(row[3], row[4], row[5]), row[6], row[7], row[10]]
            stripped_data.append(data_point)
    
    with open("./stripped_storms.csv", "w", newline='') as output:
        storm_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for point in stripped_data:
            storm_writer.writerow(point)


def flatten(xss):
    return [x for xs in xss for x in xs]

# turn list of points into segments, keep name for id purposes
def segment_storm_data(input_length = 2):
    points_by_storm = {}
    with open("./stripped_storms.csv") as input:
        csv_reader = csv.reader(input, delimiter=',')
        header = True
        for row in csv_reader:
            if header is True:
                header = False
                continue
            if len(row) == 0:
                continue
            point = points_by_storm.setdefault(row[0], [])
            row = row[1:]
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
        columns.extend([(x + "_" + str(i)) for x in ["day_of_year","lat","long","wind"]])

    with open("./segmented_data_" + str(input_length) +"s.csv", "w", newline='') as output:
        storm_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        storm_writer.writerow(columns)
        for point in segments:
            storm_writer.writerow(point)
    return

# read_storm_data()
# for i in range(3, 11):
#     segment_storm_data(i)


def get_data_by_storm(storm, input_length = 3):
    points = []
    with open("./segmented_data_" + str(input_length) + "s.csv") as input:
        csv_reader = csv.reader(input, delimiter=',')
        header = True
        for row in csv_reader:
            if header is True:
                header = False
                continue
            if len(row) == 0:
                continue
            if row[0] == storm:
                points.append(row[1:])
    return points

# print(get_data_by_storm("Caroline", 7))


# categorizes into 1 of 4 buckets
# see https://medium.com/@kap923/hurricane-path-prediction-using-deep-learning-2f9fbb390f18 for source of idea
def categorize_location(lat, long):
    bucket = 0
    if float(lat) > 20: 
        bucket |= 2
    
    if float(long) > 70:
        bucket |= 1
    
    return bucket

# for i in get_data_by_storm("Eloise", 7):
#     print(i[1] + ", " + i[2])
#     print(categorize_location(i[1], i[2]))
