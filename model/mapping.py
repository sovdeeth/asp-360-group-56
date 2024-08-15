import csv
import keras
from format_data import *
from model_keras import *
import global_var
import folium
from folium import plugins
from PIL import Image
import random
import webbrowser

def get_data_by_storm(storm, input_length = 2):
    points = []
    with open("./data_processing/test_segmented_data_" + str(input_length) + "s_cat_velo.csv") as input:
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

def map_against_storm(name, storm, model, normalizer, segments, features):
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
        new_point = [norm_truth_labels[i][0]]
        new_point.extend(output[0])
        new_point.extend(input[0][4:8])
        new_point.extend(normalizer.normalize_delta(lat - pred_coordinates['lat'][-2], long - pred_coordinates['long'][-2]))
        input = np.concatenate((input[1:], [new_point]))

        # print("Guess: ({}, {}) vs Truth ({}, {})".format(lat, long, truth_labels[i][0], truth_labels[i][1]))
    
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
    eq_map.save("./output/Model Trials/(Vel) Predictions_Map_"+name+"_Storm_"+storm+".html")
    eq_map.show_in_browser()


# function that takes in steps then generates the true coordinates and inputs, if steps < len(true), then crop true
def get_true_coord(steps, storm, segments):
    truth = get_data_by_storm(storm, segments)
    truth = [[float(value) for value in row] for row in truth]
    norm_truth = global_var.normalizer.normalize(truth)
    truth_inputs, truth_labels = split_data(truth, segments, features)
    norm_truth_inputs, norm_truth_labels = split_data(norm_truth, segments, features)
    input = norm_truth_inputs[0]
    true_coordinates = {
        'lat': [],
        'long': [],
        'windspeed': []
    }
    for point in truth_inputs:
        true_coordinates['lat'].append(point[0][1])
        true_coordinates['long'].append(point[0][2])
        true_coordinates['windspeed'].append(point[0][3])
    for i in range(len(truth_inputs[-1])-1):
        true_coordinates['lat'].append(truth_inputs[-1][i+1][1])
        true_coordinates['long'].append(truth_inputs[-1][i+1][2])
        true_coordinates['windspeed'].append(truth_inputs[-1][i+1][3])

    if steps >= len(true_coordinates['lat']):
        return truth_inputs, input, pd.DataFrame(true_coordinates)
    else:
        print(len(true_coordinates['lat']))
        true_coordinates['lat'] = true_coordinates['lat'][:steps]
        true_coordinates['long'] = true_coordinates['long'][:steps]
        true_coordinates['windspeed'] = true_coordinates['windspeed'][:steps]
        return truth_inputs, input, pd.DataFrame(true_coordinates)
    
def get_pred_coord(steps, model, input, truth_inputs):
    pred_coordinates = {
        'lat': [],
        'long': [],
        'windspeed': [],
    }

    for point in truth_inputs[0]:
        pred_coordinates['lat'].append(point[1])
        pred_coordinates['long'].append(point[2])
        pred_coordinates['windspeed'].append(point[3])

    for i in range(steps-len(truth_inputs[0])):
        output = model.predict(np.array([input]), steps = 1)
        lat, long, windspeed = global_var.normalizer.unnormalize_output(output[0])
        pred_coordinates['lat'].append(lat)
        pred_coordinates['long'].append(long)
        pred_coordinates['windspeed'].append(windspeed)
        
        # shift input
        new_point = [input[-1][0] + global_var.normalizer.normalize(6)[0]]
        new_point.extend(output[0])
        new_point.extend(input[0][4:8])
        new_point.extend(global_var.normalizer.normalize_delta(lat - pred_coordinates['lat'][-2], long - pred_coordinates['long'][-2]))
        input = np.concatenate((input[1:], [new_point]))

    return pd.DataFrame(pred_coordinates)


# function that gets true coordinates, predicted coordinates, map, and plot the data on the map ()
def plot_on_map(df, df2, eq_map):
    for i in range(len(df) - 1):
        print(i)
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
        print(i)
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

# haversine formula for calculating distance between two points on a sphere
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Radius of Earth in kilometers (mean value)
    r = 6371.0
    return c * r

def average_distance(predCoor, realCoor, row_limit=None):
    # If row_limit is provided, truncate the dataframes to the specified limit
    if row_limit is not None:
        predCoor = predCoor.iloc[:row_limit]
        realCoor = realCoor.iloc[:row_limit]
    
    if len(predCoor) > len(realCoor):
        predCoor = predCoor.iloc[:len(realCoor)]
    
    # Calculate distance for each row
    distances = haversine(predCoor['lat'], predCoor['long'], realCoor['lat'], realCoor['long'])

    # Calculate and return the average distance
    return distances.mean()

from sklearn.metrics import mean_squared_error
def mse_of_pred(predCoor, realCoor, row_limit=None):

    if row_limit is not None:
        predCoor = predCoor.iloc[:row_limit]
        realCoor = realCoor.iloc[:row_limit]

    if len(predCoor) > len(realCoor):
        predCoor = predCoor.iloc[:len(realCoor)]

    # Calculate MSE for latitude and longitude separately
    mse_lat = mean_squared_error(realCoor['lat'], predCoor['lat'])
    mse_long = mean_squared_error(realCoor['long'], predCoor['long'])

    # Average the MSE of lat and long
    mse = (mse_lat + mse_long) / 2

    return mse

def ensemble_pred(num_iteration, name, storm, segments, features, steps):
    # Initiate training data
    train_set, valid_set, test_set, global_var.normalizer = get_data(segments, features)

    # Get true coordinates and inputs
    true_inputs, input, true_coordinates = get_true_coord(steps, storm, segments)

    # Calculate mean latitude and longitude for the map center
    latm, longm = true_coordinates['lat'].mean(), true_coordinates['long'].mean()

    # Create a folium map centered at the mean location
    eq_map = folium.Map(location=[latm, longm], tiles='OpenStreetMap', zoom_start=6.0, min_zoom=2.0)

    # initialize a list to store all average distances and mse
    avg_dist_list = []
    mse_list = []

    # iterate multiple times
    for i in range(num_iteration):
        print("Iteration: ", i)
        # model = create_model(segments, features)
        # train(i, model, train_set, valid_set, batch_size, epochs)

        # name = "(Ensemble Pred) Model_{}_S{}_(B{}-E{})".format(i, segments, batch_size, epochs)
        # model = keras.saving.load_model("output\Ensemble Preds\\"+name+".keras", custom_objects=None, compile=False, safe_mode=True)

        model = keras.saving.load_model("output\checkpoints\\"+name, compile = False) # change this
        
        for i in range(len(input[-2])):
            input[-2][i] += (random.random() - 0.5) * 0.05

        # get predicted coordinates        
        pred_coordinates = get_pred_coord(steps, model, input, true_inputs)

        # plot on map
        plot_on_map(pred_coordinates, true_coordinates, eq_map)

        # calculate average distance
        avg_dist = average_distance(pred_coordinates, true_coordinates)
        avg_dist_list.append(avg_dist)

        # calculate mse
        mse = mse_of_pred(pred_coordinates, true_coordinates)
        mse_list.append(mse)

    # Save and display map
    eq_map.save(name+"_Storm_"+storm+".html")

    webbrowser.open(name+"_Storm_"+storm+".html")

    # Calculate average distance and mse
    avg_dist = np.mean(avg_dist_list)
    mse = np.mean(mse_list)

    f = open("ensemble_eval.txt", "a")
    f.write(name[22:37] + "Average Distance: " + str(round(avg_dist,2)) + "km\t" + "MSE: " + str(round(mse,2)) + "\n")
    f.close()

    f = open("ensemble_eval.txt", "r")
    print(f.read())

import os
def whole_batchsize_ensemble(num_iteration, model_iteration, storm, segments, features, batch_size, steps):
    for filename in os.listdir('output/checkpoints'):
        if filename.startswith("(Ensemble Pred) Model_{}_S{}_(B{}".format(model_iteration,segments,batch_size)):
            if filename.endswith(".keras"):
                ensemble_pred(num_iteration, filename, storm, segments, features, steps)


whole_ensemble = True

if whole_ensemble:
    model_iteration = 2
    iterations = 5
    storm = "Debby_1"
    global_var.segments = 4
    features = 10
    batch_size = 32
    steps = len(get_data_by_storm(storm, global_var.segments)) + global_var.segments

    whole_batchsize_ensemble(iterations, model_iteration, storm, global_var.segments, features, batch_size, steps)


is_ensemble_pred = True
if is_ensemble_pred and not whole_ensemble:
    iterations = 2
    filename = "(Ensemble Pred) Model_2_S4_(B64-E70).keras"
    storm = "Debby_1"
    global_var.segments = 4
    features = 10
    steps = len(get_data_by_storm(storm, global_var.segments)) + global_var.segments
    ensemble_pred(iterations, filename, storm, global_var.segments, features, steps)
elif not (is_ensemble_pred) and not(whole_ensemble):
    data = read_data_np(global_var.segments)
    normalizer = DataNormalizer(data, global_var.segments, features)
    name = "(Ensemble Pred) Model_2_S4_(B128-E71)"
    model = keras.saving.load_model("output\checkpoints\\"+name+".keras", custom_objects=None, compile=False, safe_mode=True)
    map_against_storm(name, "Debby_1", model, normalizer, global_var.segments, features)



# implement multi storm ensemble pred