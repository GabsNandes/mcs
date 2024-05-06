# Plot Health units on a state map

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import matplotlib.pyplot as plt

# Load the shapefile
state_shape = gpd.read_file('/run/media/metatron/Yggdrasil/python/arboseer/data/shapes/RJ_Municipios_2022/RJ_Municipios_2022.shp')

path = 'data/processed/sinan/sinan.parquet'

df_health_units = pd.read_parquet(path)

gdf_health_units = gpd.GeoDataFrame(df_health_units, geometry=[Point(xy) for xy in zip(df_health_units.lon, df_health_units.lat)])

# Create a plot
fig, ax = plt.subplots(figsize=(10, 10))
state_shape.plot(ax=ax, color='lightgray')  # Plot the state shape
gdf_health_units.plot(ax=ax, marker='o', color='red', markersize=1)  # Plot health units, adjust markersize as needed

# Add titles and labels (optional)
plt.title('Health Units on State Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()

