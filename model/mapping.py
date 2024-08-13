import csv
import keras
from format_data import *
from model_keras import *
import global_var
import folium
from folium import plugins
from PIL import Image


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


# data = read_data_np(segments)
# normalizer = DataNormalizer(data, segments, features)
# name = "(Vel) Model_{}_S{}_(B{}-E{})".format(iteration, segments, batch_size, epochs)
# name = "(Vel) Model_1_S2_(B64-E61)"
# model = keras.saving.load_model("output\checkpoints\\"+name+".keras", custom_objects=None, compile=False, safe_mode=True)
# map_against_storm(name, "Debby_1", model, normalizer, segments, features)

#     eq_map
#     eq_map.save("./output/Model Trials/Predictions_Map_"+name+"_Storm_"+storm+".html")



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
    if steps > len(truth_inputs):
        return input, pd.DataFrame(true_coordinates)
    else:
        true_coordinates['lat'] = true_coordinates['lat'][:steps]
        true_coordinates['long'] = true_coordinates['long'][:steps]
        true_coordinates['windspeed'] = true_coordinates['windspeed'][:steps]
        return truth_inputs, input, pd.DataFrame(true_coordinates)
    
def get_pred_coord(steps, model, input, truth_inputs):
    print("pred is called")
    pred_coordinates = {
        'lat': [],
        'long': [],
        'windspeed': []
    }

    for point in truth_inputs[0]:
        pred_coordinates['lat'].append(point[1])
        pred_coordinates['long'].append(point[2])
        pred_coordinates['windspeed'].append(point[3])

    for i in range(steps):
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

# data = read_data_np(segments)
# global_var.normalizer = DataNormalizer(data, segments, features)
# name = "(Vel) Model_{}_S{}_(B{}-E{})".format(iteration, segments, batch_size, epochs)
# name = "(Vel) Model_1_S2_(B64-E61)"
# model = keras.saving.load_model("output\checkpoints\\"+name+".keras", custom_objects=None, compile=False, safe_mode=True)
# steps = 5
# true_inputs, input, true_coordinates = get_true_coord(steps, storm, segments)
# pred_coordinates = get_pred_coord(steps, model, input, true_inputs)
# print(pred_coordinates)

# function that gets true coordinates, predicted coordinates, map, and plot the data on the map ()
def plot_on_map(df, df2, eq_map):
    print("plot is called")
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


def ensemble_pred(num_iteration, storm, segments, features, batch_size, epochs, steps):
    # Initiate training data
    train_set, valid_set, test_set, global_var.normalizer = get_data(segments, features)

    # Get true coordinates and inputs
    true_inputs, input, true_coordinates = get_true_coord(steps, storm, segments)

    # Calculate mean latitude and longitude for the map center
    latm, longm = true_coordinates['lat'].mean(), true_coordinates['long'].mean()

    # Create a folium map centered at the mean location
    eq_map = folium.Map(location=[latm, longm], tiles='OpenStreetMap', zoom_start=6.0, min_zoom=2.0)

    # iterate multiple times
    for i in range(num_iteration):
        model = create_model(segments, features)
        train(i, model, train_set, valid_set, batch_size, epochs)

        name = "(Ensemble Pred) Model_{}_S{}_(B{}-E{})".format(i, segments, batch_size, epochs)
        model = keras.saving.load_model("output\Ensemble Preds\\"+name+".keras", custom_objects=None, compile=False, safe_mode=True)

        # get predicted coordinates        
        pred_coordinates = get_pred_coord(steps, model, input, true_inputs)

        # plot on map
        plot_on_map(pred_coordinates, true_coordinates, eq_map)

    # Save and display map
    eq_map.save("./output/Ensemble Preds/(Ensemble Pred) Predictions_Map_"+name+"_Storm_"+storm+".html")
    eq_map.show_in_browser()

        # initiate the truth values
        # generate predictions for each model
        # loop plot_on_map for each model
        # calculate deviation for each model then average them

iterations = 5
storm = "Debby_1"
segments = 2
features = 10
batch_size = 64
epochs = 52
steps = len(get_data_by_storm(storm, segments))

ensemble_pred(iterations, storm, segments, features, batch_size, epochs, steps)