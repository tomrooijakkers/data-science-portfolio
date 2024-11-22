import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# 1. Fictieve regenmeetdata
# Meetlocaties (lon, lat) en neerslag in mm
data = {
    "lon": [4.90, 5.12, 5.35, 5.48, 4.65],
    "lat": [52.37, 52.16, 51.98, 52.30, 52.55],
    "rain": [10, 20, 15, 25, 5],  # neerslag in mm
}

# Converteer naar numpy arrays
lon = np.array(data["lon"])
lat = np.array(data["lat"])
rain = np.array(data["rain"])

# 2. Definieer een raster over Nederland
grid_lon = np.linspace(4.5, 6.0, 100)  # Lengtegraad
grid_lat = np.linspace(51.8, 52.6, 100)  # Breedtegraad
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 3. Kriging interpolatie
# Initialiseer Ordinary Kriging
OK = OrdinaryKriging(
    lon, lat, rain,
    variogram_model="exponential",
    verbose=False,
    enable_plotting=False
)

# Interpoleer naar het raster
rain_interp, ss = OK.execute("grid", grid_lon[0, :], grid_lat[:, 0])

# 4. Plot de resultaten
plt.figure(figsize=(10, 8))
plt.contourf(grid_lon, grid_lat, rain_interp, cmap="Blues", levels=20)
plt.colorbar(label="Neerslag (mm)")
plt.scatter(lon, lat, c=rain, cmap="Reds", edgecolor="k", label="Meetpunten")
plt.xlabel("Lengtegraad")
plt.ylabel("Breedtegraad")
plt.title("Interpolatie van neerslag in Nederland")
plt.legend()
plt.show()