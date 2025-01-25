# KNMI Weather Data

This repository contains a series of Jupyter Notebooks for analyzing weather data retrieved from the KNMI (<em>Dutch Royal Meteorological Institute</em>) web script service. 

The analysis includes: 
- Data retrieval
- Data cleaning
- Data visualization
- Insight generation, among which a drought analysis

And all this only by using publicly available weather data!

## Project Structure

Each of the following Notebooks can be used and run on a stand-alone basis.

[01-knmi_data_retrieval.ipynb](./01-knmi_data_retrieval.ipynb): Demonstrates how to retrieve weather data from the KNMI web script service.

[02-knmi_data_visualization.ipynb](./02-knmi_data_visualization.ipynb): Shows how to visualize and explore the weather data to uncover trends and insights.

[03-knmi_drought_analysis.ipynb](./03-knmi_drought_analysis.ipynb): Clarifies how to analyze and quantify periods of drought and wetness using Standardized Indexes (SPI and SPEI), and contains a MICE-based data imputation.

## Getting Started

To get started with this project, you need to set up the required environment and dependencies. Follow the steps below to create and activate the virtual environment.

### 1. Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/tomrooijakkers/data-science-portfolio.git

cd data-science-portfolio/knmi-weather
```

### 2. Set Up the Virtual Environment
#### Using ```conda``` (strongly recommended for managing environments)

Create a new virtual environment from the ```knmi_weather_environment.yml``` file:

```bash
conda env create -f knmi_weather_environment.yml
```

Then, activate the environment:

```bash
conda activate knmi-weather-env
```

#### Using venv (if you prefer using ```pip```):

Create a virtual environment:
```bash
python3 -m venv knmi-weather-env
```

Activate the environment:

- On Linux / macOS:
```bash
source knmi-weather-env/bin/activate
```
- On Windows:
```bash
knmi-weather-env\Scripts\activate
```

Then, install the dependencies using ```pip```:
```bash
pip install -r knmi_weather_requirements.txt
```

### 3. Install Dependencies
Ensure that all required dependencies are installed. 

- If you are using ```conda```, the ```knmi_weather_environment.yml``` file should have all the necessary packages listed. 

- If you are using ```pip```, install the packages via: 
```bash
pip install -r knmi_weather_requirements.txt
```

### 4. Optionally: install GeoPandas
If you wish to run the geo-operation workflows in the Data Retrieval Notebook, you can install GeoPandas as follows:

#### Using ```conda```:
```bash
conda activate knmi-weather-env

conda install geopandas
```

#### Using ```pip```:
Assuming your virtual PIP environment is already activated:
```bash
pip install geopandas==0.14.2
```  

### 5. Run the Notebooks
After setting up your virtual environment, you can run the Jupyter Notebooks in the <code>knmi-weather</code> folder.

Then, launch Jupyter Notebook: 
```bash
jupyter notebook
```
After that, open the desired notebook in your browser: 
- [01-knmi_data_retrieval.ipynb](./01-knmi_data_retrieval.ipynb)
- [02-knmi_data_visualization.ipynb](./02-knmi_data_visualization.ipynb)
- [03-knmi_drought_analysis.ipynb](./03-knmi_drought_analysis.ipynb)

Finally, follow the steps in the notebook to perform the data analysis and visualizations.


At this point you should be ready to go - happy reading / coding!

## Folder Structure

- **knmi-weather/**: <em>project directory</em>.
    - ```01-knmi_data_retrieval.ipynb```: Data retrieval and preprocessing.
    - ```02-knmi_data_visualization.ipynb```: Data analysis and visualizations.
    - ```03-knmi_drought_analysis.ipynb```: Drought analysis using SPI and SPEI.
    - ```knmi_weather_environment.yml```: Conda environment configuration (for ```conda``` users).
    - ```knmi_weather_requirements.txt```: Python dependencies (for ```pip``` users).
    - ```scripts/``` <em>directory</em>: Python helper scripts for powering the workflows in the Jupyter Notebooks.
        - ```metadata/``` <em>subdirectory</em>: JSON files for station and parameter metadata (loaded dynamically).
        - ```transform/``` <em>subdirectory</em>: JSON files for (pre-)cleaning data transformation rules; can be changed to your preference.

## Disclaimer
I am not officially affiliated with KNMI, and the scripts provided throughout this Data Science Portfolio section are offered on an unofficial basis for educational and exploratory purposes.

## License

This project is licensed under the same terms as the parent repository. Also see [LICENSE](../LICENSE) and [LICENSE_CONTENT](../LICENSE_CONTENT).