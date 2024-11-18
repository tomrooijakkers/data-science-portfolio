

import os
import json

import pandas as pd



def load_tf_json(filename: str) -> list[dict] | dict:
    """Load 'transform' JSON to Python object."""
    with open(os.path.join("transform", filename)) as f:
        json_obj = json.load(f)
    
    return json_obj


def transform_param_values(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: Create text here."""
    

# Step 1: Load JSON mapping file
# TODO: Find parameter code in JSON file ('parameter_code')
# TODO: Remap colname to 'parameter_name'
# TODO: if 'is_summable', change all '-1' observations to 0.25 (indicates small observations >0 mm, but <0.5 x 0.1 mm)
# TODO: multiply all values in col by 'parameter_mult' to get unit value


# TODO: Update stations transform to be part of this script, using 'transform_stations.json' as colmapper
    #stations = (df_stations.rename(columns=col_map)
    #                       .to_dict(orient="records"))

""" {
      "parameter_code": "DDVEC",
      "parameter_desc": "Vectorgemiddelde windrichting in graden (360=noord, 90=oost, 180=zuid, 270=west, 0=windstil/variabel)",
      "parameter_name": "vect_avg_wind_dir_deg",
      "parameter_unit": "deg",
      "parameter_mult": 1,
      "is_summable": false
    }, 
"""