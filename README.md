# Bus Ridership Prediction for Trondheim

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project contains a complete machine learning pipeline for predicting bus ridership in Trondheim, Norway. The model uses historical ridership data, weather data, and event data to generate accurate predictions.

## Features

*   **End-to-End ML Pipeline:** From data collection to model training and evaluation.
*   **Feature Engineering:** Creates a rich set of features, including time-based, weather-based, and event-based features.
*   **Hyperparameter Tuning:** Optimizes the model for the best performance.
*   **Business Impact Analysis:** Includes scripts to analyze the business value of the model.

## Project Structure

The project is organized as follows:

```
├── .gitignore
├── complete_pipeline.py
├── config
│   └── config.json
├── Data
│   ├── atb_data
│   ├── processed
│   └── weather
├── events_config.json
├── notebooks
├── README.md
├── reports
├── requirements.txt
├── scripts
├── src
│   ├── business_impact_analysis.py
│   ├── event_feature_builder.py
│   ├── feature_engineered_RF.py
│   ├── hyperparameter_tuning_RF.py
│   ├── master_dataset_builder.py
│   ├── models
│   │   └── final_bus_ridership_model.joblib
│   ├── retrieve_hourly_weather.py
│   └── train_final_model.py
└── venv
```

*   **`complete_pipeline.py`**: The main script to run the entire ML pipeline.
*   **`src/`**: Contains the core Python scripts for the pipeline.
*   **`Data/`**: Contains the raw, processed, and weather data.
*   **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA) and experimentation.
*   **`scripts/`**: Auxiliary scripts for analysis, visualization, and reporting.
*   **`reports/`**: Generated reports, metrics, and figures.
*   **`config/`**: Configuration files for the project.
*   **`requirements.txt`**: A list of all the Python packages required to run the project.

## Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

*   Python 3.9 or higher
*   `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the complete ML pipeline, execute the `complete_pipeline.py` script from the root of the project:

```bash
python complete_pipeline.py
```

You can also run specific steps of the pipeline by using the command-line arguments. For example, to run the pipeline starting from step 2, use:

```bash
python complete_pipeline.py --start-from 2
```

To see all the available options, use:

```bash
python complete_pipeline.py --help
```

## Data

The model uses the following data sources:

*   **ATB Open Data:** Historical bus ridership data from ATB.
*   **MET Norway Frost API:** Weather data for Trondheim.
*   **Event Data:** A custom-built dataset of major events in Trondheim.

## Modeling

The final model is a Random Forest Regressor with optimized hyperparameters. The model is trained on a rich set of features, including:

*   **Time-based features:** Hour of the day, day of the week, month, etc.
*   **Weather features:** Temperature, precipitation, wind speed, etc.
*   **Event features:** Public holidays, major events, etc.
*   **Lag features:** Ridership from previous hours.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
