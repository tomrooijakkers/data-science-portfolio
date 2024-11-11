import json

class KNMIMeteoStation:
    def __init__(self, location_name: str, station_code: str | int):
        self.location_name = location_name
        self.station_code = str(station_code).zfill(3)

    def __repr__(self):
        """Return instance class name and key properties."""
        return f"{self.__class__.__name__}({self.station_code}, {self.location_name}"


class KNMIMeteoStationList:
    def __init__(self):
            self.stations_floc = "datafiles/knmi_meteo_stations.json"
            self.stations = self.get_saved_stations(self.stations_floc)
    

    def get_saved_stations(self, saved_loc: str) -> list[dict]:
        """Load saved stations from JSON file to Python."""
        with open(saved_loc) as f:
            saved_stations = json.load(f)

        return saved_stations
