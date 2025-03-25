# Formula 1 Prediction for the Japanese Grand Prix 2025

This project predicts the winner and podium finishers for the 2025 Japanese Grand Prix at Suzuka using historical Formula 1 (F1) data from 1950 to 2024.The prediction model is an ensemble of XGBoost and LightGBM classifiers, trained on a dataset of past Suzuka races. The project also evaluates the model's performance using various error metrics, such as accuracy, confusion matrix, log loss, ROC-AUC score, and mean absolute error (MAE) for predicted probabilities. <br> <br>
The 2025 season includes significant driver and team changes, which are incorporated into the prediction. The final prediction considers factors like weather forecasts (dry, 21°C) and the latest team dynamics.


## Project Structure

The project is organized as follows:

- `predict_2025_japanese_gp.ipynb`: The main Jupyter Notebook containing the code for data loading, preprocessing, feature engineering, model training, and prediction.
- `data/`: This directory should contain the CSV files with the F1 data (races.csv, results.csv, drivers.csv, constructors.csv, qualifying.csv, lap_times.csv, pit_stops.csv).

## Data

The project uses historical Formula 1 data from the Ergast Developer API, which can be found on Kaggle: [Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). Make sure to download the data and place it in the `data/` directory before running the code.

## Methodology

1. **Data Loading and Preprocessing:** The code loads the data from the CSV files and merges relevant information into a single DataFrame.
2. **Feature Engineering:** Several features are engineered to improve model accuracy, including driver form, constructor form, Suzuka-specific performance, driver age, qualifying time difference, pole position indicator, Suzuka race experience, recent form, fastest lap potential, tire strategy, and weather impact.
3. **Model Training:** An ensemble model is trained using XGBoost and LightGBM classifiers. SMOTE is used to handle class imbalance in the dataset. Hyperparameter tuning is performed using GridSearchCV to optimize model performance.
4. **Prediction:** The trained model is used to predict the winner and podium for the 2025 Japanese Grand Prix based on input data.

## Usage

1. Make sure you have the necessary libraries installed (pandas, numpy, scikit-learn, xgboost, lightgbm, imblearn).
2. Place the F1 data files in the `data/` directory.
3. Open the `predict_2025_japanese_gp.ipynb` notebook in Google Colab or Jupyter Notebook.
4. Run the notebook cells sequentially.

## Results

The model's performance is evaluated using metrics such as accuracy, classification report, log loss, ROC-AUC score, and mean absolute error. The prediction results for the 2025 Japanese Grand Prix are displayed in the notebook, showing the predicted winner and podium.

## Future Improvements

- Incorporate more advanced weather data.
- Add more relevant features, such as tire compound and pit stop strategy.
- Explore other machine learning models.
- Improve hyperparameter tuning and model evaluation techniques.
