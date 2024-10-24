# Cyclone-Data-Classification-using-Random-Forest

## Authors
- Himansu Agrawal
- Subhalaxmi Nayak

## Overview
The project focuses on classifying cyclones based on various meteorological features such as wind speed, pressure, and geographical coordinates using machine learning techniques. Random Forest, a powerful ensemble learning method, is applied to identify the most important features and predict the cyclone’s status. The process involves feature selection, model training, and evaluation using key metrics like accuracy, precision, recall, and F1-score.

## Dataset Overview
The dataset contains detailed cyclone information including:
- **Geospatial data**: Latitude, Longitude, and hemispherical classification.
- **Meteorological parameters**: Maximum wind speed, minimum pressure, and wind patterns.
- **Time-based features**: Date, Time, Month, and Year.

The cyclones are labeled with various statuses, including tropical depressions, storms, hurricanes, and more, making this a multi-class classification problem.

## Preprocessing Steps
1. **Date and Time Handling**: The `Date` column is converted to datetime format, and a new `Year` column is extracted. Similarly, the `Time` column is formatted to a uniform HHMM format.
   
2. **Hemisphere Encoding**: Latitude and longitude are divided into northern/southern and eastern/western hemispheres respectively, which helps in classifying cyclones by geographical region.

3. **Handling Missing Data**: Missing values (denoted by `-999`) in features are replaced with the mean of the available data to maintain consistency.

4. **Feature Selection**: Initially, unnecessary columns like `ID` and `Event` are removed. Feature selection techniques are applied using Random Forest to identify the top features contributing to the classification task.

5. **Feature Transformation**: The latitude and longitude columns are cleaned, and time-series data is restructured for analysis.

## Model Building

### Random Forest Classifier
Random Forest is used as the primary model for cyclone classification. It is an ensemble learning method that combines multiple decision trees to enhance predictive performance. The following steps outline the model building process:
1. **Training and Testing Split**: The data is split into training and testing sets, with 70% used for training and 30% for testing.
2. **Model Training**: A Random Forest classifier with 1000 trees (`n_estimators=1000`) is trained on the selected features.
3. **Feature Importance**: The model computes the relative importance of each feature. The top features identified are:
   - Maximum Wind
   - Minimum Pressure
   - Latitude
   - Year
   - Longitude
4. **Evaluation Metrics**: After training, the model is evaluated using:
   - **Accuracy**: The overall correctness of the model’s predictions.
   - **Precision, Recall, F1-Score**: Metrics to understand the balance between false positives and false negatives, particularly useful for imbalanced datasets.

### Decision Tree Classifier
For comparison, a Decision Tree classifier is also trained on the same data. The performance of both models is compared using the same metrics.

## Visualizations
- **Pie Chart**: Visualizes the top ten most frequent cyclones based on the number of occurrences.
- **Barplots**: Show the frequency of cyclones by month and category, providing insights into cyclone trends.
- **Heatmap**: Displays the correlation matrix of features, highlighting relationships between meteorological parameters.
  
## Cyclone Classification
The trained Random Forest model predicts the cyclone type based on user-provided input (e.g., maximum wind, pressure, latitude, longitude, and year). The predicted cyclone type is mapped to its description, giving insights into the characteristics and impact of different cyclone types.

## Conclusion
The classification model effectively predicts cyclone types with high accuracy and provides detailed insights into cyclone trends based on historical data. The Random Forest model outperforms the Decision Tree model in terms of accuracy and feature importance, making it a suitable choice for cyclone data classification tasks.

This project demonstrates the power of machine learning in predicting weather patterns and can be further extended to more complex meteorological datasets for improved accuracy and real-time predictions.
