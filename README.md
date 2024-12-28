# Analysis of Fight Data and Predicting Prices With Machine Learning
This repository explores flight data to uncover factors influencing airfare and build predictive models to estimate ticket prices. By leveraging various machine learning regression algorithms, we aim to provide insights and accurate predictions for ticket prices. The project focuses on analyzing flight data, performing Exploratory Data Analysis (EDA), and predicting ticket prices by testing multiple regression models. The dataset used for this project is sourced from Kaggle.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Algorithms Tested](#algorithms-tested)
- [Exploratory Data Analysis and Preprocessing](#exploratory-data-analysis-and-preprocessing)
- [Tools and Technologies](#tools-and-technologies)
- [Workflow](#workflow)
- [Key Insights](#key-insights)
- [Results and Evaluation](#results-and-evaluation)
- [Final Model](#final-model)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview

Predicting flight ticket prices can help airlines, travel agencies, and passengers make informed decisions. This project provides detailed insights through EDA, visualizations, and the implementation of machine learning models.

## Dataset

The dataset used in this project collected from Kaggle and available to use in the `data` folder includes:

- Airline details
- Date and time of travel
- Source and destination locations
- Duration of the flight
- Number of stops
- Ticket prices (target variable)

> **Note**: The dataset was preprocessed to ensure data integrity and prepare it for modeling.

## Algorithms Tested
The following regression algorithms were tested:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Ada Boost Regression
- XGBoost Regression

## Exploratory Data Analysis and Preprocessing

### Data Analysis and Visualization
- Exploratory Data Analysis (EDA) was performed to understand the data distribution, trends, and correlations.
- Visualizations highlighted the relationship between various features and ticket prices.

### Data Preprocessing
The preprocessing steps applied to the dataset include:
- Removal of duplicate entries.
- One-hot encoding of categorical variables.
- Normalizing numerical features.
- Preparing the data for model training and evaluation.

## Tools and Technologies

The project leverages the following technologies:

- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
- **Techniques**:
  - Regression modeling
  - Cross-validation
  - Hyperparameter tuning
- **Visualization**: Data visualizations for exploratory data analysis (EDA).
- **Jupyter Notebook**: Development and analysis environment.

## Workflow

1. **Data Collection**: Obtained flight data from Kaggle.
2. **Data Cleaning and Preprocessing:**
    - Removed duplicate entries.
    - Applied one-hot encoding for categorical variables.
    - Normalized and scaled numerical features.
3. **Exploratory Data Analysis (EDA)**:
   - Explored key patterns, distributions, and relationships.
   - Visualized trends using `matplotlib` and `seaborn`.
4. **Model Implementation**:
   - Tested nine regression models.
   - Evaluated models using RMSE, R² Score, and Cross-Validation.
5. **Hyperparameter Tuning**:
   - Fine-tuned the best-performing model (XGBoost Regressor).
6. **Final Prediction**:
   - Generated predictions using the optimized XGBoost model.

## Key Insights

- **Feature Importance**: Key features impacting ticket prices include airline, number of stops, and travel duration.
- **Data Trends**:
  - Flights with more stops tend to have higher ticket prices.
  - Peak travel times (e.g., weekends or holidays) are associated with increased prices.
- **Model Performance**:
  - Among the models tested, XGBoost Regressor provided the highest accuracy, making it suitable for price prediction tasks.
- **Challenges**:
  - Data preprocessing was critical to handle categorical features and normalize numerical variables effectively.
  - Balancing model complexity with interpretability was essential during model selection.

## Results and Evaluation
### Performance of Models
The performance metrics (RMSE, R² Score, Cross-Validation Score) for each algorithm are as follows:

| Model                   | RMSE     | R² Score | Cross-Validation Score | R² - CV Difference |
|-------------------------|----------|----------|-------------------------|--------------------|
| Linear Regression       | 3253.25 | 41.94%   | 37.03%                 | 4.92%             |
| Ridge Regression        | 3253.25 | 41.94%   | 37.03%                 | 4.92%             |
| Lasso Regression        | 3253.24 | 41.94%   | 37.03%                 | 4.92%             |
| Support Vector Regression (SVR) | 4217.93 | 2.41%    | 2.11%                  | 0.30%             |
| Decision Tree Regression| 1963.14 | 78.86%   | 83.04%                 | -4.18%            |
| Random Forest Regression| 3057.13 | 48.73%   | 44.51%                 | 4.23%             |
| Gradient Boosting Regression | 2483.34 | 66.17%   | 66.71%                 | -0.54%            |
| Ada Boost Regression    | 3066.19 | 48.43%   | 32.32%                 | 16.11%            |
| XGBoost Regression      | 1488.18 | 87.85%   | 89.42%                 | -1.57%            |


### Best Model
Among the models evaluated, the **XGBoost Regression** model delivered the best performance. It achieved the lowest RMSE of 1488.18 and the highest R² Score of 87.85%. Additionally, its Cross-Validation Score was an impressive 89.42%, and the small R² - CV Difference of -1.57% indicated consistent performance across different folds of the data. These metrics suggest that XGBoost is highly effective at capturing the underlying patterns in the data and making accurate predictions, making it the best model for this task.

## Final Model

After determining that XGBoost Regression provided the best accuracy, we proceeded to optimize the model further through hyperparameter tuning. This process involved fine-tuning various parameters to enhance the model's performance and predictive accuracy. 

The steps we took included:
- **Parameter Adjustment:** Tweaking key parameters such as learning rate, max depth, and n_estimators to find the optimal combination.
- **Cross-Validation:** Performing extensive cross-validation to ensure the model's robustness and to prevent overfitting.
- **Evaluation Metrics:** Closely monitoring evaluation metrics like RMSE and R² Score throughout the tuning process.

As a result of this rigorous optimization, we achieved an impressive R² score of 89.94%, significantly improving the model's predictive power. The tuned XGBoost model was then used for the final predictions, confirming its effectiveness in capturing the underlying patterns in the data and making precise predictions.

Key highlights of the final model:
- **Best Performance:** XGBoost Regression emerged as the top-performing model with the highest R² score.
- **Enhanced Accuracy:** Achieved an R² score of 89.94% after hyperparameter tuning.
- **Robust Predictions:** The model's predictions were consistent and reliable across different data subsets.

The final, optimized XGBoost model demonstrates its superior capability in handling complex datasets, making it an excellent choice for this regression task.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshaEadara/Analysis-of-Fight-Data-and-Predicting-Prices-With-Machine-Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Analysis-of-Fight-Data-and-Predicting-Prices-With-Machine-Learning
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Analysis_of_Fight_Data_and_Predicting_Prices_With_Machine_Learning.ipynb
   ```
5. Ensure the dataset `Flight_Data_Testing.xlsx` and  `Flight_Data_Training.xlsx` is available in the project directory.
6. Run the notebook cells sequentially to reproduce the analysis.

## Contributing

Contributions to this project are welcome. If you'd like to suggest improvements or report issues, please open an issue or submit a pull request on the repository. Let's collaborate to make this project even better!

