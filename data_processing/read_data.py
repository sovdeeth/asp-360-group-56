import csv

# read file into list of points, omit status, category, diameter, etc.
def read_storm_data():
    stripped_data = []
    with open("./storms.csv") as input:
        csv_reader = csv.reader(input, delimiter=',')
        for row in csv_reader:
            data_point = [row[1], row[3], row[4], row[5], row[6], row[7], row[10]]
            stripped_data.append(data_point)
    
    with open("./stripped_storms.csv", "w", newline='') as output:
        storm_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for point in stripped_data:
            storm_writer.writerow(point)


def flatten(xss):
    return [x for xs in xss for x in xs]

# turn list of points into triplets, strip name
def segment_storm_data():
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
    
    # split into triplets
    triplets = []
    for storm_points in points_by_storm.values():
        length = len(storm_points)
        for i in range(0, length - 2):
            if i % 1000 == 0:
                print(flatten(storm_points[i:i+3]))
            triplets.append(flatten(storm_points[i:i+3]))

    with open("./triplets.csv", "w", newline='') as output:
        storm_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        storm_writer.writerow(["monthA","dayA","hourA","latA","longA","windA","monthB","dayB","hourB","latB","longB","windB","monthC","dayC","hourC","latC","longC","windC"])
        for point in triplets:
            storm_writer.writerow(point)
    
        



            
    return

read_storm_data()
segment_storm_data()