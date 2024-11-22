import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from mpl_toolkits.basemap import Basemap

# 1. Voorbeelddata (fictieve neerslagmetingen op verschillende locaties)
# X, Y zijn de coördinaten van meetstations (in dit geval willekeurige punten in Nederland)
# Z zijn de gemeten neerslaghoeveelheden in mm

# Coördinaten van meetstations (bijvoorbeeld X, Y in Nederland)
stations_x = np.array([4.9, 5.4, 5.9, 6.5, 6.0])  # longitude (bijv. Oost-West coördinaten)
stations_y = np.array([52.5, 52.0, 52.2, 51.7, 51.9])  # latitude (bijv. Noord-Zuid coördinaten)
neerslag = np.array([12.9, 5.3, 14.7, 1.0, 13.2])  # gemeten neerslag in mm

# 2. Het grid maken waarop we de interpolatie willen uitvoeren
grid_x, grid_y = np.meshgrid(np.linspace(3., 8., 500, endpoint=True), 
                             np.linspace(50., 54., 500, endpoint=True))

# 3. Kriging-instelling: We gebruiken Ordinary Kriging voor interpolatie
OK = OrdinaryKriging(stations_x, stations_y, neerslag,
                     variogram_model="spherical",
                     verbose=False, enable_plotting=False)

# 4. Kriging uitvoeren om de neerslag op het grid te voorspellen
grid_z, ss = OK.execute('grid', grid_x[0, :], grid_y[:, 0])

# 5. Visualisatie van de resultaten (map van Nederland)
plt.figure(figsize=(10, 8))

# Maak een basemap van Nederland
m = Basemap(projection='merc', llcrnrlat=50.7, urcrnrlat=53.7, llcrnrlon=3.3, urcrnrlon=7.3, resolution='h')
m.drawcountries(linewidth=0.5, linestyle='-.', color='k')
m.drawrivers(linewidth=0.3, linestyle='solid', color='darkblue')
m.drawcoastlines(linewidth=0.5, linestyle='solid', color='k')

# Zet de gridcoördinaten om naar kaartprojectiecoördinaten
x, y = m(grid_x, grid_y)

# Mask values outside land borders using the Basemap land-sea mask
land_mask = np.vectorize(m.is_land)(x, y)
land_grid_z = np.ma.masked_where(np.logical_not(land_mask), grid_z)

# Plot de interpolatiewaarden (neerslag)
cs = m.contourf(x, y, land_grid_z, cmap='Blues')

# Voeg een colorbar toe voor de neerslagwaarden
cbar = m.colorbar(cs, location='right', pad="10%")
cbar.set_label('Neerslag (mm)')

# Plot de meetstations als rode punten
stn_x, stn_y = m(stations_x, stations_y)
m.scatter(stn_x, stn_y, color="red", edgecolor="k", marker="x", label="Meetstations")

plt.title("Kriging-interpolatie van neerslag in Nederland")
plt.show()

"""
# Convert grid coordinates to map projection coordinates
x, y = m(grid_lon, grid_lat)

# Mask values outside land borders using the Basemap land-sea mask
land_mask = m.is_land(grid_lon, grid_lat)
masked_rain_interp = np.ma.masked_where(~land_mask, rain_interp)

# Plot the interpolated rainfall only within land borders
cs = m.contourf(x, y, masked_rain_interp, cmap="Blues", levels=20)

# Add a colorbar for rainfall values
cbar = m.colorbar(cs, location='right', pad="10%")
cbar.set_label("Neerslag (mm)")

# Overlay the measurement points
mx, my = m(lon, lat)  # Project station coordinates onto the map
m.scatter(mx, my, c=rain, cmap="Reds", edgecolor="k", marker="o", label="Meetpunten")

# Add labels and title
plt.title("Kriging-interpolatie van neerslag in Nederland")
plt.legend()
plt.show()
"""