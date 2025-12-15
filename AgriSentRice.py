import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
#data = pd.read_csv('merged_data.csv')
data = pd.read_csv('merged_climate_sentiment_rice1.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date column is datetime

# Sort by Date to maintain temporal order
data = data.sort_values(by='Date').reset_index(drop=True)

print(len(data))

###############################################################################
# Feature Engineering
###############################################################################
# Create lagged features for all relevant features to avoid data leakage
features_to_lag = ['Maza_Rice_Price', 'TAVG', 'TMAX', 'TMIN',
                   'average_weighted_sentiment', 'sum_of_negative_sentiment',
                   'average_sentiment', 'sum_of_positive_sentiment']

# Generate lagged features and rolling averages
for feature in features_to_lag:
    data[f'{feature}_Lag_1'] = data[feature].shift(3)
    data[f'{feature}_Lag_2'] = data[feature].shift(4)
    data[f'{feature}_Rolling_3_Lag_1'] = data[feature].rolling(6).mean().shift(3)

# Extract month and year for seasonality and trends
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year

# Optional: Encode month as cyclic features
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
print(len(data))

# Drop rows with missing lagged values
data.dropna(inplace=True)

###############################################################################
# Define Feature Sets
###############################################################################
features_without_sentiment = [
    'Maza_Rice_Price_Lag_1', 'Maza_Rice_Price_Lag_2', 'Maza_Rice_Price_Rolling_3_Lag_1',
    'TAVG_Lag_1',  'TAVG_Rolling_3_Lag_1',
    'TMAX_Lag_1',  'TMAX_Rolling_3_Lag_1',
    'TMIN_Lag_1',  'TMIN_Rolling_3_Lag_1', 'month'
]

features_with_sentiment = [
    'Maza_Rice_Price_Lag_1', 'Maza_Rice_Price_Lag_2', 'Maza_Rice_Price_Rolling_3_Lag_1',
    'TAVG_Lag_1',  'TMAX_Rolling_3_Lag_1',
    'TMIN_Lag_1',  'TMIN_Rolling_3_Lag_1', 'month',
    'average_weighted_sentiment_Lag_1',  'average_weighted_sentiment_Rolling_3_Lag_1',

]

target = 'Maza_Rice_Price'

def get_sarimax_feature_importance(model, feature_names):
    """
    Extract and visualize feature importance from SARIMAX exogenous variables.

    Args:
        model: Fitted SARIMAX model.
        feature_names: List of feature names used as exogenous variables.

    Returns:
        DataFrame with feature names and their coefficients.
    """
    if model is not None and hasattr(model, 'params'):
        # Extract coefficients for the exogenous variables
        params = model.params
        exog_params = {name: params[name] for name in feature_names if name in params}
        
        # Convert to DataFrame
        importance_df = pd.DataFrame.from_dict(exog_params, orient='index', columns=['Coefficient']).reset_index()
        importance_df.columns = ['Feature', 'Coefficient']
        importance_df['Absolute Coefficient'] = importance_df['Coefficient'].abs()
        importance_df = importance_df.sort_values(by='Absolute Coefficient', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Coefficient'], color='skyblue')
        plt.xlabel('Coefficient')
        plt.ylabel('Features')
        plt.title('Maza Rice: SARIMAX Feature Importance (Exogenous Variables)')
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()

        return importance_df
    else:
        print("The SARIMAX model does not have exogenous coefficients to interpret.")
        return None

def get_feature_importance(model, feature_names, model_name):
    """
    Get and plot feature importance for a given model.
    
    Args:
        model: Trained model with a `feature_importances_` attribute.
        feature_names: List of feature names corresponding to the input data.
        model_name: Name of the model (for plot title).
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(f'Maza Rice: Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.show()
        
        return importance_df
    else:
        print(f"The model {model_name} does not support feature importances.")
        return None

def get_linear_regression_feature_importance(model, feature_names):
    """
    Extract and display feature importance from Linear Regression coefficients.
    
    Args:
        model: Fitted Linear Regression model.
        feature_names: List of feature names used as input.
    """
    if model is not None and hasattr(model, 'coef_'):
        coefficients = model.coef_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        importance_df['Absolute Coefficient'] = importance_df['Coefficient'].abs()
        importance_df = importance_df.sort_values(by='Absolute Coefficient', ascending=False)

        # Plot coefficients
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Coefficient'], color='skyblue')
        plt.xlabel('Coefficient')
        plt.ylabel('Features')
        plt.title('Maza Rice: Linear Regression Feature Importance')
        plt.gca().invert_yaxis()
        plt.show()

        return importance_df
    else:
        print("The Linear Regression model does not have coefficients to interpret.")
        return None


def get_ridge_regression_feature_importance(model, feature_names):
    """
    Extract and display feature importance from Ridge Regression coefficients.
    
    Args:
        model: Fitted Ridge Regression model.
        feature_names: List of feature names used as input.
    """
    if model is not None and hasattr(model, 'coef_'):
        coefficients = model.coef_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        importance_df['Absolute Coefficient'] = importance_df['Coefficient'].abs()
        importance_df = importance_df.sort_values(by='Absolute Coefficient', ascending=False)

        # Plot coefficients
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Coefficient'], color='skyblue')
        plt.xlabel('Coefficient')
        plt.ylabel('Features')
        plt.title('Maza Rice: Ridge Regression Feature Importance')
        plt.gca().invert_yaxis()
        plt.show()

        return importance_df
    else:
        print("The Ridge Regression model does not have coefficients to interpret.")
        return None

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def calculate_residuals(y_actual, y_pred):
    """
    Calculate residuals between actual and predicted values.

    Args:
        y_actual (array-like): Actual target values (Pandas Series or NumPy array).
        y_pred (array-like): Predicted target values (Pandas Series or NumPy array).

    Returns:
        residuals (NumPy array): Difference between actual and predicted values.
    """
    # Convert to NumPy arrays for consistent processing
    if isinstance(y_actual, pd.Series):
        y_actual = y_actual.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    # Ensure lengths match
    if len(y_actual) != len(y_pred):
        raise ValueError(f"Length mismatch: y_actual ({len(y_actual)}), y_pred ({len(y_pred)})")

    # Calculate residuals
    residuals = y_actual - y_pred

    print("len(y_actual):", len(y_actual))
    print("len(y_pred):", len(y_pred))
    print("len(residuals):", len(residuals))

    return residuals


def plot_residuals(y_actual, y_pred, model_name):
    """
    Plot residuals for a given model.
    
    Args:
        y_actual (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        model_name (str): Name of the model.
    """
    residuals = calculate_residuals(y_actual, y_pred)
    
    print("len(y_actual):",len(y_actual))
    print("len(residuals):",len(residuals))
    
    # Set the output directory
    import os
    output_dir = r"C:\Users\Musaad Alzahrani\Downloads\twitter-sentiment results\Maza"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist


    plt.figure(figsize=(14, 6))

    # Scatter plot of residuals
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6, color="blue")
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
    plt.title(f'Maza Rice: {model_name}: Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.grid(True)

    # Distribution of residuals
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, color="skyblue", bins=30)
    plt.title(f'Maza Rice: {model_name}: Residual Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.show()
    # Save the plot
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{model_name}_residuals_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the plot to free memory

import os

def plot_residuals2(y_actual, y_pred, model_name):
    """
    Plot residuals for a given model and save the plot.
    
    Args:
        y_actual (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        model_name (str): Name of the model.
    """
    # Calculate residuals
    residuals = calculate_residuals(y_actual, y_pred)
    
    print("len(y_actual):", len(y_actual))
    print("len(residuals):", len(residuals))

    # Set the output directory
    output_dir = r"C:\Users\Musaad Alzahrani\Downloads\twitter-sentiment results\maza"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Create the plot
    plt.figure(figsize=(14, 6))

    # Scatter plot of residuals
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6, color="blue")
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
    plt.title(f'Maza Rice: {model_name}: Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.grid(True)

    # Distribution of residuals
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, color="skyblue", bins=30)
    plt.title(f'Maza Rice: {model_name}: Residual Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)

    # Save the plot
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{model_name}_residuals_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the plot to free memory

    print(f"Plot saved at: {output_path}")

def analyze_model_residuals(models, y_actual, forecasts_with, forecasts_without):
    """
    Perform residual analysis for multiple models.
    
    Args:
        models (list): List of model names.
        y_actual (array-like): Actual target values.
        forecasts_with (dict): Predicted values (with sentiment features) for each model.
        forecasts_without (dict): Predicted values (without sentiment features) for each model.
    """
    
    print("len(y_actual):", len(y_actual))
    print("len(forecasts_with):", len(forecasts_with))
    print("len(forecasts_without):", len(forecasts_without))

    for model_name in models:
        print(f"Residual Analysis for {model_name} (With Sentiment):")
        plot_residuals2(y_actual, forecasts_with[model_name], f'{model_name} (With Sentiment)')

        print(f"Residual Analysis for {model_name} (Without Sentiment):")
        plot_residuals2(y_actual, forecasts_without[model_name], f'{model_name} (Without Sentiment)')

###############################################################################
# Function to Train and Evaluate Models
###############################################################################

def evaluate_models(X_without, X_with, y, model_label):
    # Split chronologically to preserve temporal order
    train_size = int(len(data) * 0.8)

    X_train_without, X_test_without = X_without.iloc[:train_size], X_without.iloc[train_size:]
    X_train_with, X_test_with = X_with.iloc[:train_size], X_with.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    print(f"Size of y_train: {y_train.shape[0]}")
    print(f"Size of y_test: {y_test.shape[0]}")

    # Initialize results dictionary and forecast storage
    results = {}
    forecasts_without = {}
    forecasts_with = {}

    ###############################################################################
    # SARIMAX Model
    ###############################################################################
    sarimax_model_without = SARIMAX(y_train, exog=X_train_without, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False)
    sarimax_results_without = sarimax_model_without.fit(disp=False)
    sarimax_forecast_without = sarimax_results_without.forecast(steps=len(y_test), exog=X_test_without)

    sarimax_model_with = SARIMAX(y_train, exog=X_train_with, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False)
    sarimax_results_with = sarimax_model_with.fit(disp=False)
    sarimax_forecast_with = sarimax_results_with.forecast(steps=len(y_test), exog=X_test_with)
    print("len(sarimax_forecast_with):",len(sarimax_forecast_with))

    results['SARIMAX'] = (
        np.sqrt(mean_squared_error(y_test, sarimax_forecast_without)),
        mean_absolute_error(y_test, sarimax_forecast_without),
        r2_score(y_test, sarimax_forecast_without),
        np.sqrt(mean_squared_error(y_test, sarimax_forecast_with)),
        mean_absolute_error(y_test, sarimax_forecast_with),
        r2_score(y_test, sarimax_forecast_with)
    )
    forecasts_without['SARIMAX'] = sarimax_forecast_without
    forecasts_with['SARIMAX'] = sarimax_forecast_with
    print("R^2 sarimax_model_without:", r2_score(y_test, sarimax_forecast_without))
    print("R^2 sarimax_model_with:", r2_score(y_test, sarimax_forecast_with))

    ###############################################################################
    # Gradient Boosting Model
    ###############################################################################
    gb_model_without = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=3)
    gb_model_without.fit(X_train_without, y_train)
    gb_forecast_without = gb_model_without.predict(X_test_without)

    gb_model_with = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=3)
    gb_model_with.fit(X_train_with, y_train)
    gb_forecast_with = gb_model_with.predict(X_test_with)
    print("len(gb_forecast_with):",len(gb_forecast_with))

    results['Gradient Boosting'] = (
        np.sqrt(mean_squared_error(y_test, gb_forecast_without)),
        mean_absolute_error(y_test, gb_forecast_without),
        r2_score(y_test, gb_forecast_without),
        np.sqrt(mean_squared_error(y_test, gb_forecast_with)),
        mean_absolute_error(y_test, gb_forecast_with),
        r2_score(y_test, gb_forecast_with)
    )
    forecasts_without['Gradient Boosting'] = gb_forecast_without
    forecasts_with['Gradient Boosting'] = gb_forecast_with
    print("R^2 Gradient Boosting without:", r2_score(y_test, gb_forecast_without))
    print("R^2 Gradient Boosting with:", r2_score(y_test, gb_forecast_with))
    ###############################################################################
    # Random Forest Model
    ###############################################################################
    rf_model_without = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=5)
    rf_model_without.fit(X_train_without, y_train)
    rf_forecast_without = rf_model_without.predict(X_test_without)

    rf_model_with = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=5)
    rf_model_with.fit(X_train_with, y_train)
    rf_forecast_with = rf_model_with.predict(X_test_with)
    # After training the Random Forest model
    rf_feature_importance = get_feature_importance(rf_model_with, X_with.columns, "Random Forest")
    
    # After training the Gradient Boosting model
    gb_feature_importance = get_feature_importance(gb_model_with, X_with.columns, "Gradient Boosting")

    # Feature names used as exogenous variables
    feature_names = X_with.columns  # Replace with the actual column names of your exog variables
    
    # Get and visualize SARIMAX feature importance
    sarimax_feature_importance = get_sarimax_feature_importance(sarimax_results_with, feature_names)
    
    print("len(rf_forecast_with):",len(rf_forecast_with))


    results['Random Forest'] = (
        np.sqrt(mean_squared_error(y_test, rf_forecast_without)),
        mean_absolute_error(y_test, rf_forecast_without),
        r2_score(y_test, rf_forecast_without),
        np.sqrt(mean_squared_error(y_test, rf_forecast_with)),
        mean_absolute_error(y_test, rf_forecast_with),
        r2_score(y_test, rf_forecast_with)
    )
    forecasts_without['Random Forest'] = rf_forecast_without
    forecasts_with['Random Forest'] = rf_forecast_with
    
    print("R^2 Random Forest without:", r2_score(y_test, rf_forecast_without))
    print("R^2 Random Forest with:", r2_score(y_test, rf_forecast_with))
    ###############################################################################
    # Ridge Regression Model
    ###############################################################################
    ridge_model_without = Ridge(alpha=1.0)
    ridge_model_without.fit(X_train_without, y_train)
    ridge_forecast_without = ridge_model_without.predict(X_test_without)

    ridge_model_with = Ridge(alpha=1.0)
    ridge_model_with.fit(X_train_with, y_train)
    ridge_forecast_with = ridge_model_with.predict(X_test_with)
    print("len(ridge_forecast_with):",len(ridge_forecast_with))

    results['Ridge Regression'] = (
        np.sqrt(mean_squared_error(y_test, ridge_forecast_without)),
        mean_absolute_error(y_test, ridge_forecast_without),
        r2_score(y_test, ridge_forecast_without),
        np.sqrt(mean_squared_error(y_test, ridge_forecast_with)),
        mean_absolute_error(y_test, ridge_forecast_with),
        r2_score(y_test, ridge_forecast_with)
    )
    forecasts_without['Ridge Regression'] = ridge_forecast_without
    forecasts_with['Ridge Regression'] = ridge_forecast_with
    
    print("R^2 Ridge Regression without:", r2_score(y_test, ridge_forecast_without))
    print("R^2 Ridge Regression with:", r2_score(y_test, ridge_forecast_with))
    ###############################################################################
    # Linear Regression Model
    ###############################################################################
    lr_model_without = LinearRegression()
    lr_model_without.fit(X_train_without, y_train)
    lr_forecast_without = lr_model_without.predict(X_test_without)

    lr_model_with = LinearRegression()
    lr_model_with.fit(X_train_with, y_train)
    lr_forecast_with = lr_model_with.predict(X_test_with)
    print("len(lr_forecast_with):",len(lr_forecast_with))

    results['Linear Regression'] = (
        np.sqrt(mean_squared_error(y_test, lr_forecast_without)),
        mean_absolute_error(y_test, lr_forecast_without),
        r2_score(y_test, lr_forecast_without),
        np.sqrt(mean_squared_error(y_test, lr_forecast_with)),
        mean_absolute_error(y_test, lr_forecast_with),
        r2_score(y_test, lr_forecast_with)
    )
    
    lr_feature_importance = get_linear_regression_feature_importance(lr_model_with, X_with.columns)
    ridge_feature_importance = get_ridge_regression_feature_importance(ridge_model_with, X_with.columns)

    forecasts_without['Linear Regression'] = lr_forecast_without
    forecasts_with['Linear Regression'] = lr_forecast_with
    
    print("R^2 Linear Regression without:", r2_score(y_test, lr_forecast_without))
    print("R^2 Linear Regression with:", r2_score(y_test, lr_forecast_with))
    
    models = ['SARIMAX', 'Gradient Boosting', 'Random Forest', 'Ridge Regression', 'Linear Regression']

    # Perform residual analysis
    analyze_model_residuals(models, y_test, forecasts_with, forecasts_without)

    ###############################################################################
    # Visualization
    ###############################################################################
    import matplotlib.dates as mdates

    for model_name in forecasts_without.keys():
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'].iloc[train_size:], y_test, label='Actual', color='blue')
        plt.plot(data['Date'].iloc[train_size:], forecasts_without[model_name], label=f'Predicted ({model_name}) Without Sentiment', color='orange')
        plt.plot(data['Date'].iloc[train_size:], forecasts_with[model_name], label=f'Predicted ({model_name}) With Sentiment', color='green')

        # Format x-axis to show ticks every 3 months
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.legend()
        plt.title(f'Maza Rice Price: {model_name} Performance: With and Without Sentiment')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    # Save results to CSV
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE Without Sentiment', 'MAE Without Sentiment', 'R^2 Without Sentiment',
                                                                          'RMSE With Sentiment', 'MAE With Sentiment', 'R^2 With Sentiment'])
    results_df.to_csv('model_performance_comparison.csv', index_label='Model')

###############################################################################
# Function to Plot Last 12 Rows for Each Model
###############################################################################


###############################################################################
# Evaluate Models
###############################################################################
results = evaluate_models(
    data[features_without_sentiment],
    data[features_with_sentiment],
    data[target],
    "Model Comparison"
)



# Function to plot actual vs predicted prices
def plot_predictions(y_actual, y_pred_with_sentiment, y_pred_without_sentiment, model_name, last_n=12):
    """
    Plots the actual prices, predicted with sentiment, and predicted without sentiment.
    Args:
        y_actual (array-like): Actual target values.
        y_pred_with_sentiment (array-like): Predicted values with sentiment features.
        y_pred_without_sentiment (array-like): Predicted values without sentiment features.
        model_name (str): Name of the model.
        last_n (int): Number of recent rows to display in the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual[-last_n:], label='Actual Prices', marker='o', color='black')
    plt.plot(y_pred_with_sentiment[-last_n:], label='Predicted (with Sentiment)', marker='x', linestyle='--')
    plt.plot(y_pred_without_sentiment[-last_n:], label='Predicted (without Sentiment)', marker='s', linestyle='--')
    plt.title(f"{model_name}: Actual vs Predicted Prices (Last {last_n} Rows)")
    plt.xlabel("Time")
    plt.ylabel("Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage for each model
# Ridge Regression

