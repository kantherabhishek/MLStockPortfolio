# Optimized Portfolio Allocation - README
The code in file main.py demonstrates an application for optimizing portfolio allocation using machine learning techniques. The code uses the Dash framework to create a web application with a user-friendly interface. Users can input a list of stock symbols, select various parameters, and obtain an optimized portfolio allocation based on mean-variance optimization.

# You can view the workings of the code on YouTube via the link below.
 https://www.youtube.com/watch?v=v8YU9fygZac

# Prerequisites
Before running the code, make sure you have the following libraries installed:

 yfinance,
 pandas,
 numpy,
 scikit-learn,
 scipy,
 matplotlib,
 dash
 # You can install the required libraries using pip:
   pip install yfinance pandas numpy scikit-learn scipy matplotlib dash
  # Running the Code
    To run the code, follow these steps:

# Import the required libraries:
   ### import yfinance as yf
   ### import pandas as pd
   ### import numpy as np
   ### from sklearn.model_selection import TimeSeriesSplit
   ### from sklearn.linear_model import LinearRegression
   ### from scipy.optimize import minimize
   ### import matplotlib.pyplot as plt
   ### import dash
   ### from dash import dcc, html
   ### import datetime
   ### from dash.dependencies import Input, Output, State
 # Create a Dash application:
  app = dash.Dash(__name__)
 # Define the layout of the application:
    app.layout = html.Div([
        # ... HTML elements and Dash components ...
    ])
 # Implement the callback function to update portfolio information and pie chart based on user input:
    @app.callback(
        [Output("portfolio-info", "children"),
         Output("pie-chart", "figure")],
        [Input("calculate-button", "n_clicks")],
        [State("symbol-list", "value"),
         State("risk-free-rate", "value"),
         State("portfolio-amount", "value"),
         State("start-date", "date"),
         State("end-date", "date"),
         State("interval", "value"),
         State("max-allocation", "value")]
    )
    def update_portfolio(n_clicks, symbol_list, risk_free_rate, portfolio_amount, start_date, end_date, interval, max_allocation):
        # ... code for portfolio optimization and visualization ...
 # Define the main function to run the application:
    if __name__ == "__main__":
        app.run_server(debug=True)
# Run the code by executing the Python script:
  python filename.py
## Open a web browser and access the application at http://localhost:8050/
## Using the Application
    Once the application is running, you can use it to optimize portfolio allocation. 
## Follow these steps:
   ### 1. Enter a list of stock symbols separated by commas in the "List of Symbols" text area.
   ### 2. Specify the risk-free rate as a decimal value in the "Risk-Free Rate" input field.
   ### 3. Set the portfolio amount in Indian Rupees (₹) in the "Portfolio Amount" input field.
   ### 4. Select the start date and end date for the historical data using the date pickers.
   ### 5. Choose the data interval (daily, weekly, or monthly) from the "Interval" dropdown menu.
   ### 6. Set the maximum allocation allowed for each stock as a decimal value in the "Max Allocation" input field.
   ### 7. Click the "Start Optimization Process" button.
# The application will display the optimized portfolio allocation, including the number of shares to buy for each stock, the latest price, and the latest value. The remaining funds after allocation will also be shown. A pie chart will visualize the allocation of the portfolio across different stocks.

# Customizing the Application
  You can customize the application according to your needs. Here are a few suggestions:

Modify the layout and styling of the web interface using HTML and CSS within the app.layout section.
Adjust the parameters and constraints of the portfolio optimization process to suit your requirements.
Enhance the visualization by adding additional charts or metrics.

# References
   ### yfinance
   ### pandas
   ### numpy
   ### scikit-learn
   ### scipy
   ### matplotlib
   ### Dash
This concludes the README file for the Machine Learning optimized portfolio allocation code. Happy investing!
