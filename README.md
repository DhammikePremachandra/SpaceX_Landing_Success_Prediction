# SpaceX Landing Success Prediction

## Overview
This project focuses on predicting the success of the first stage landings of SpaceX's Falcon 9 rockets. The workflow includes data collection, data wrangling, exploratory data analysis (EDA), feature engineering, and training multiple machine learning models to classify landing outcomes.

## Table of Contents
- [Overview](#overview)
- [Data Collection](#data-collection)
- [Data Wrangling](#data-wrangling)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Setup Instructions](#setup-instructions)
- [Contributing](#contributing)

## Data Collection
The dataset is collected from the SpaceX API, focusing on past rocket launches. The data includes details about rockets, launch sites, payloads, and landing outcomes. Additional information is retrieved using endpoints for rockets, launchpads, payloads, and cores.

## Data Wrangling
Data wrangling includes:
- Filtering relevant columns and rows.
- Handling missing values.
- Extracting specific information from nested JSON structures.
- Converting date columns to datetime format.
- Preparing the dataset for analysis.

## Exploratory Data Analysis (EDA)
EDA helps in understanding the dataset better:
- Launch counts per launch site.
- Distribution of orbits.
- Mission outcomes and their occurrences.
- Relationships between various variables (e.g., Payload Mass vs. Flight Number, Flight Number vs. Launch Site).
- Yearly trend of landing success rates.

## Feature Engineering
Selected features include:
- FlightNumber, PayloadMass, Orbit, LaunchSite, Flights, GridFins, Reused, Legs, LandingPad, Block, ReusedCount, Serial.
- Applied one-hot encoding to categorical variables (Orbit, LaunchSite, LandingPad, Serial).
- Standardized the data for better model performance.

## Model Training and Evaluation
The following models were trained and tuned using GridSearchCV:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**

Each model was evaluated based on accuracy on the training and test datasets.

## Results
### Logistic Regression
- **Best Hyperparameters**: {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
- **Training Accuracy**: 83.04%
- **Test Accuracy**: 94.00%

### Support Vector Machine (SVM)
- **Best Hyperparameters**: {'C': 0.03162277660168379, 'gamma': 0.001, 'kernel': 'linear'}
- **Training Accuracy**: 83.21%
- **Test Accuracy**: 83.00%

### Decision Tree
- **Best Hyperparameters**: {'criterion': 'gini', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'splitter': 'random'}
- **Training Accuracy**: 86.25%
- **Test Accuracy**: 72.00%

### K-Nearest Neighbors (KNN)
- **Best Hyperparameters**: {'algorithm': 'auto', 'n_neighbors': 8, 'p': 1}
- **Training Accuracy**: 81.61%
- **Test Accuracy**: 89.00%

## Conclusion
The Logistic Regression model performed the best with a test accuracy of 94%, demonstrating good generalization to unseen data. The KNN model also performed well, while the Decision Tree showed signs of overfitting.

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SpaceX_Landing_Success_Prediction.git
   cd SpaceX_Landing_Success_Prediction
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the data collection script:**
   ```bash
   python Collecting_the_data.ipynb
   ```

4. **Run the data wrangling script:**
   ```bash
   python Data_Wrangling.ipynb
   ```

5. **Run the model training script:**
   ```bash
   python Exploring_Data_Preprocessing_and_MachineLearningPrediction.ipynb
   ```

6. **View the results and performance metrics in the output files.**

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
