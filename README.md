Polynomial Regression

This project demonstrates how to implement polynomial regression using Python and `scikit-learn`. The code generates synthetic data, fits a polynomial regression model, evaluates the model's performance using R-squared score, and visualizes the results.

### Project Overview

- **Polynomial Regression**: A form of regression analysis where the relationship between the independent variable \(x\) and the dependent variable \(y\) is modeled as an \(n\)-degree polynomial.
- **R-squared Score**: Measures how well the model explains the variance in the target variable. A higher value (closer to 1) indicates a better fit.

### Features

- **Data Generation**: Synthetic data is generated based on a polynomial function with added noise.
- **Polynomial Transformation**: Uses `PolynomialFeatures` from `sklearn.preprocessing` to transform the feature into polynomial features before fitting the model.
- **Model Training**: A linear regression model is fitted to the transformed features.
- **Model Evaluation**: The model's performance is evaluated using the R-squared score.
- **Plotting**: Visualizes the training data, test data, and the fitted polynomial regression curve.

### Installation

To run the code, you'll need to have Python and the required libraries installed. If you don't have them, you can install the dependencies using `pip`.

1. Clone this repository or download the script.
2. Install the necessary Python libraries:

```bash
pip install numpy matplotlib scikit-learn
```

### Code Overview
Data Generation: The generate_data() function generates synthetic data based on a polynomial function of degree n (default is 2), with some added noise.

Data Splitting: The dataset is split into training and testing sets using the split_data() function.

Model Training: A polynomial regression model is trained using the train_model() function, which utilizes PolynomialFeatures to transform the input features.

Model Evaluation: The model is evaluated using the R-squared score, which is calculated by the evaluate_model() function.

Visualization: The results are plotted using matplotlib, showing both the data points and the fitted polynomial curve.

### Example Output
After running the main() function, the R-squared score and a plot of the data with the fitted polynomial regression curve will be displayed. An example output might look like this:

R-squared score of the model: 0.8098

The plot will show the training data (in red), test data (in blue), and the fitted polynomial regression curve (in black dashed line).

### Usage
Adjust Polynomial Degree: You can modify the degree of the polynomial by changing the degree variable in the main() function. For example:

degree = 3 

#### Example of Running the Code
To run the code, simply execute the script:

python polynomial_regression.py


This will generate synthetic data, fit the polynomial regression model, calculate the R-squared score, and plot the results.

### Requirements
Python 3.x
numpy version 1.18 or higher
matplotlib version 3.x or higher
scikit-learn version 0.22 or higher

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Notes:
- Make sure you replace the name of the script `polynomial_regression.py` with the actual name of your file if it's different.
- The README assumes you will run the script directly in the command line, so adjust accordingly if your usage is different.
