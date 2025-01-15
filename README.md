#  NYC Traffic Collisions - Visual Analytics
A comprehensive project for studying traffic collisions in New York City during the summer months of 2018.


## About:
This project investigates the root causes and contributing conditions of motor vehicle collisions in the summer 
of 2018. By integrating an interactive visual analytics tool with dimensionality reduction techniques, key factors 
such as location, weather  conditions, vehicle types, and contributing causes are highlighted. The insights gained 
can guide city authorities, insurance companies, and the public in developing effective strategies to enhance road safety.


## Project Structure

Below is an overview of the files and folders in this repository:

* `data.zip`: Zipped folder containing the raw and processed datasets used for analysis.

* `preprocessing.ipynb`: Jupyter notebook containing data cleaning and transformation steps.
  
* `NYC_map.geojson`: File containing NYC map for the visualizations.

* `visualizations.ipynb`: Jupyter notebook used for generating visualizations and exploratory data analysis (including dimensionality reduction analysis)
  
* `dashboard.py`: Python file with the main code for your data dashboard.

* `streamlit.py``: Streamlit application that calls `dashboard.py` to render the interactive visualizations.

* `NYC_collisions_report.pdf`: Paper with final report detailing the findings, methods, and analysis of this project.


## Prerequisites

For running the project you will need `Python3` and `pip3` so make sure to have them updated.

Packages used for this project include:
- `pandas` and `numpy` for basic data manipulation
- `geopandas` for preprocessing location information
- `scikit-learn` for dimensionality reduction techniques
- `altair` for data visualization
- `streamlit` for final dashboard

To install all the packages used execute the following command:
```
pip3 install -r requirements.txt
```
If any problems may arise, install them separately running the command `pip3 install` followed by the package name.


## Author

Project developed by student Alba García Ochoa for Visual Analytics course on Sapienza University. 

For more information please contact me at `garciaochoa.2181770@studenti.uniroma1.it`
