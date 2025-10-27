Source code for Geomatics MSc thesis:"A Graph Neural Network-based Approach to Predict the Effects if Urban Climate on Personal Mobility Choices in Seoul South Korea".  
Find the full text here: _[I WILL INSERT TU DELFT REPOSITORY LINK]_
  
Results shown in the thesis report can be reproduced by running the following scripts:  
  
- **mobility_model_with_climate.ipynb** to train and evaluate the STGNN variants.  
(both urban and limate features will be used by default, you can toggle either with the USE_CLIMATE_FEATURES or USE_URBAN_FEATURES booleans).
- **losses_plots.ipynb** for the plotting of the training and validation loss curves.
- **scenarios.ipynb** to run the time-of-day and weather scenarios shown in the results.
- **scenarios_weather.ipynb** to alter or create new scenarios based on the S-DoT weather stations values.

Other scripts can be used to for the processing of data and construction of graph structure. In particular:

- **roads_preprocessing.ipynb, road_intersection_points.ipynb, node_creation_intersections.ipynb** for the creation of road intersection nodes.
- **climate_features.py** for the calculation of temperature, humidity, and PM10 node features from the S-DoT dataset.
- **node_urban_features_intersections.ipynb** for the calculation of node urban metrics from urban morphology datasets.

