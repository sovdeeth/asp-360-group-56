import folium

# Create your folium map (assuming eq_map is your map object)
eq_map = folium.Map(location=[45.5236, -122.6750], zoom_start=13)

# Save the map to an HTML file
eq_map.save('map.html')

# Modify the title of the HTML file
with open('map.html', 'r') as file:
    content = file.read()

# Replace the default title (usually "Map") with your desired title
new_title = "My Custom Map Title"
content = content.replace("<title>Map</title>", f"<title>{new_title}</title>")

# Save the modified content back to the HTML file
with open('map.html', 'w') as file:
    file.write(content)

# Open the map in the browser
import webbrowser
webbrowser.open('map.html')