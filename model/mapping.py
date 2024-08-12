import csv
import keras
from format_data import *
import folium
from folium import plugins
from PIL import Image


def get_data_by_storm(storm, input_length = 2):
    points = []
    with open("./data_processing/segmented_data_" + str(input_length) + "s_cat.csv") as input:
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

def map_against_storm(storm, model, normalizer, segments, features):
    global name
    # get storm data, segmented
    truth = get_data_by_storm(storm, segments)
    truth = [[float(value) for value in row] for row in truth]
    norm_truth = normalizer.normalize(truth)
    print(truth[0])
    truth_inputs, truth_labels = split_data(truth, segments, features)
    norm_truth_inputs, norm_truth_labels = split_data(norm_truth, segments, features)
    print(truth_inputs[0])

    input = norm_truth_inputs[0]
    print(input)

    
    true_coordinates = {
        'lat': [],
        'long': [],
        'windspeed': []
    }


    pred_coordinates = {
        'lat': [],
        'long': [],
        'windspeed': []
    }

    for point in truth_inputs[0]:
        pred_coordinates['lat'].append(point[1])
        pred_coordinates['long'].append(point[2])
        pred_coordinates['windspeed'].append(point[3])
        true_coordinates['lat'].append(point[1])
        true_coordinates['long'].append(point[2])
        true_coordinates['windspeed'].append(point[3])

    for point in truth_labels:
        true_coordinates['lat'].append(point[0])
        true_coordinates['long'].append(point[1])
        true_coordinates['windspeed'].append(point[2])
    
    for i in range(len(truth)):
        output = model.predict(np.array([input]), steps = 1)
        lat, long, windspeed = normalizer.unnormalize_output(output[0])
        pred_coordinates['lat'].append(lat)
        pred_coordinates['long'].append(long)
        pred_coordinates['windspeed'].append(windspeed)

        # shift input
        input = np.concatenate((input[1:], [[norm_truth_labels[i][0], output[0][0], output[0][1], output[0][2], input[0][4],input[0][5],input[0][6],input[0][7]]]))
    
    df = pd.DataFrame(pred_coordinates)
    df2 = pd.DataFrame(true_coordinates)

    # Calculate mean latitude and longitude for the map center
    latm, longm = df['lat'].mean(), df['long'].mean()

    # Create a folium map centered at the mean location
    eq_map = folium.Map(location=[latm, longm], tiles='OpenStreetMap', zoom_start=6.0, min_zoom=2.0)

    # Add a heatmap to the map
    # eq_map.add_child(plugins.HeatMap(df[['lat', 'long']].values, radius=15))

    # Add lines connecting consecutive points
    for i in range(len(df) - 1):
        p1 = df.iloc[i][['lat', 'long']].tolist()
        p2 = df.iloc[i + 1][['lat', 'long']].tolist()

        # Set lines to different colors based on windspeed
        if df.iloc[i]['windspeed'] >= 40:
            folium.PolyLine([p1, p2], color='red', weight=2.5, opacity=1).add_to(eq_map)
        elif 40 > df.iloc[i]['windspeed'] >= 30:
            folium.PolyLine([p1, p2], color='orange', weight=2.5, opacity=1).add_to(eq_map)
        elif 30 > df.iloc[i]['windspeed'] >= 20:
            folium.PolyLine([p1, p2], color='yellow', weight=2.5, opacity=1).add_to(eq_map)
        elif 20 > df.iloc[i]['windspeed'] >= 10:
            folium.PolyLine([p1, p2], color='green', weight=2.5, opacity=1).add_to(eq_map)
        elif 10 > df.iloc[i]['windspeed'] >= 0:
            folium.PolyLine([p1, p2], color='blue', weight=2.5, opacity=1).add_to(eq_map)


    # Add a marker for the starting point
    starting_point = df.iloc[0][['lat', 'long']].tolist()
    folium.Marker(location=starting_point,
                popup='Starting Point',
                icon=folium.Icon(color='orange', icon=None)).add_to(eq_map)

    # Add lines connecting consecutive points for the second set of coordinates
    for i in range(len(df2) - 1):
        p1 = df2.iloc[i][['lat', 'long']].tolist()
        p2 = df2.iloc[i + 1][['lat', 'long']].tolist()

            # Set lines to different colors based on windspeed
        if df2.iloc[i]['windspeed'] >= 40:
            folium.PolyLine([p1, p2], color='red', weight=2.5, opacity=1, dash_array='10').add_to(eq_map)
        elif 40 > df2.iloc[i]['windspeed'] >= 30:
            folium.PolyLine([p1, p2], color='orange', weight=2.5, opacity=1, dash_array='10').add_to(eq_map)
        elif 30 > df2.iloc[i]['windspeed'] >= 20:
            folium.PolyLine([p1, p2], color='yellow', weight=2.5, opacity=1, dash_array='10').add_to(eq_map)
        elif 20 > df2.iloc[i]['windspeed'] >= 10:
            folium.PolyLine([p1, p2], color='green', weight=2.5, opacity=1, dash_array='10').add_to(eq_map)
        elif 10 > df2.iloc[i]['windspeed'] >= 0:
            folium.PolyLine([p1, p2], color='blue', weight=2.5, opacity=1, dash_array='10').add_to(eq_map)

    # Add a marker for the starting point of the second set of coordinates
    starting_point2 = df2.iloc[0][['lat', 'long']].tolist()
    folium.Marker(location=starting_point2,
                popup='Starting Point 2',
                icon=folium.Icon(color='green', icon=None)).add_to(eq_map)

    # Display the map
    eq_map
    eq_map.save("./output/Model Trials/Predictions_Map_"+name+"_Storm_"+storm+".html")


segments = 3
features = 8

data = read_data_np(segments)
normalizer = DataNormalizer(data, segments, features)
name = "Model_3_S3_(B32-E32)"
model = keras.saving.load_model("output\Model Trials\\"+name+".keras", custom_objects=None, compile=False, safe_mode=True)
map_against_storm("Caroline_3", model, normalizer, segments, features)

