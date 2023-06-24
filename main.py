import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
import datetime


from dash.dependencies import Input, Output, State

# Create Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Machine Learning Optimized Portfolio Allocation",style={"text-align":"center"}),
    html.Div(
    children=[
        html.Label("List of Symbols (separated by comma): eg. TCS.NS, ITC.NS, RELIANCE.NS", style={"display": "block"}),
        dcc.Textarea(id="symbol-list", placeholder="Enter symbols", style={"width": "50%","height":"200px"}),
    ],
    style={"margin-bottom": "30px", "text-align": "center"}
),
    html.Div(
        children=[
            html.Label("Risk-Free Rate: eg. for 3.5% = 0.035", style={"display": "block"}),
            dcc.Input(id="risk-free-rate", type="number", min=0, step=0.01, value=0.035, className="input-field", style={"width": "30%", "text-align": "center"}),
        ],
        style={"margin-bottom": "30px", "text-align": "center"}
    ),
    html.Div(
        children=[
            html.Label("Portfolio Amount (in ₹): ", style={"display": "block"}),
            dcc.Input(id="portfolio-amount", type="number", min=0, step=1000, value=1000000, className="input-field", style={"width": "30%", "text-align": "center"}),
        ],
        style={"margin-bottom": "30px", "text-align": "center"}
    ),
    html.Div(
        className="input-row",
        children=[
            html.Label("Start Date: ", className="input-label", style={"display": "block"}),
            dcc.DatePickerSingle(
                id="start-date",
                placeholder="Select a date",
                className="input-field",
                date="2002-12-01"
            ),
        ],
        style={"text-align": "center"}
    ),
    html.Div(
        className="input-row",
        children=[
            html.Label("End Date: ", className="input-label", style={"display": "block"}),
            dcc.DatePickerSingle(
                id="end-date",
                placeholder="Select a date",
                className="input-field",
                date=datetime.date.today()  # Set the default date to today's date
            ),
        ],
        style={"text-align": "center"}
    ),
    html.Div(
        children=[
            html.Label("Interval: ", style={"display": "block"}),
            dcc.Dropdown(
                id="interval",
                options=[
                    {"label": "Daily", "value": "1d"},
                    {"label": "Weekly", "value": "1wk"},
                    {"label": "Monthly", "value": "1mo"}
                ],
                value="1wk",
                style={"width": "100%"}
            )
        ],
        style={"margin-bottom": "20px", "text-align": "center"}
    ),
    html.Div(
        children=[
            html.Label("Max Allocation: Amount allowed to hold in a single security. For 100% allowed use '1.0' and for 10% use '0.1'", style={"display": "block"}),
            dcc.Input(id="max-allocation", type="number", min=0, max=1, step=0.01, value=0.1, style={"width": "30%", "text-align": "center"}),
        ],
        style={"margin-bottom": "20px", "text-align": "center"}
    ),
    html.Button("Start Optimization Process", id="calculate-button", n_clicks=0, style={"width": "100%", "text-align": "center"}),
    html.Div(id="portfolio-info", style={"margin-top": "20px"}),
    dcc.Graph(id="pie-chart")
], style={"max-width": "70%", "margin": "0 auto", "padding": "20px", "font-family": "Arial"})

# Callback for updating portfolio information and pie chart
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
    if n_clicks > 0:
        # Convert symbol list to ticker symbols
        tickers = [symbol.strip() for symbol in symbol_list.split(",")]

        # Download stock price data from Yahoo Finance
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)['Adj Close'].fillna(method='ffill')

        # Calculate log returns for each stock
        log_returns = np.log(data / data.shift(1)).dropna()

        # User-defined Risk Free Rate and Maximum weight for each stock
        rf_rate = float(risk_free_rate)
        Max_allocation_p = float(max_allocation)

        # Create function to perform portfolio optimization using mean-variance optimization framework
        def portfolio_optimization(weights, log_returns, rf_rate):
            port_log_returns = np.sum(log_returns * weights, axis=1)
            port_mean_return = np.mean(port_log_returns)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov(), weights)))
            Sharpe_ratio = (port_mean_return - rf_rate) / port_volatility
            portfolio_return = (1 + port_mean_return) * (1 + rf_rate) - 1
            return np.array([port_mean_return, port_volatility, Sharpe_ratio, portfolio_return])

        # Define objective function for optimization process
        def objective_function(weights, log_returns, rf_rate):
            return portfolio_optimization(weights, log_returns, rf_rate)[0]

        # Define bounds and constraints for optimization process
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: Max_allocation_p - x})

        # Perform cross-validation to choose regression model
        tscv = TimeSeriesSplit(n_splits=5)
        regressor = LinearRegression()
        mean_r_squared = []
        for train_index, test_index in tscv.split(log_returns):
            train_log_returns, test_log_returns = log_returns.iloc[train_index], log_returns.iloc[test_index]
            train_X, train_y = train_log_returns.iloc[:, :-1], train_log_returns.iloc[:, -1]
            test_X, test_y = test_log_returns.iloc[:, :-1], test_log_returns.iloc[:, -1]
            regressor.fit(train_X, train_y)
            mean_r_squared.append(regressor.score(test_X, test_y))
        mean_r_squared = np.mean(mean_r_squared)

        # Perform portfolio optimization using the chosen regression model and maximizing Sharpe Ratio
        weights = minimize(objective_function, len(tickers) * [1 / len(tickers)],
                           args=(log_returns, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints).x
        weights /= np.sum(weights)

        # Print portfolio weights and performance metrics
        portfolio = pd.DataFrame({'Ticker': tickers, 'Weight': weights})
        port_mean_return, port_volatility, Sharpe_ratio, portfolio_return = portfolio_optimization(
            weights, log_returns, rf_rate)

        portfolio_amount = float(portfolio_amount)
        if not np.isnan(portfolio_amount):
            # Get discrete allocation of each share per stock
            allocation = np.multiply(weights, portfolio_amount)

            # Use vlookup to get latest prices
            latest_prices = pd.DataFrame({'Ticker': tickers})
            latest_prices['Latest Price'] = latest_prices['Ticker'].apply(lambda x: data[x].iloc[-1])

            # Divide allocation by latest prices
            latest_prices.set_index('Ticker', inplace=True)
            allocation_df = pd.DataFrame({'Ticker': tickers, 'Allocation': allocation})
            allocation_df.set_index('Ticker', inplace=True)
            allocation_df['Number of shares to buy'] = round(allocation_df['Allocation'] / latest_prices['Latest Price'], 0)
            allocation_df['Latest Price'] = round(latest_prices['Latest Price'],4)
            allocation_df['Latest Value'] = round(allocation_df['Number of shares to buy'] * latest_prices['Latest Price'],4)

            allocation_df.reset_index(inplace=True)

            # Create portfolio_df and print allocation for stocks with weight >= 1
            portfolio_df = allocation_df[['Ticker', 'Number of shares to buy', 'Latest Price', 'Latest Value']]
            high_allocation_df = portfolio_df[portfolio_df['Number of shares to buy'] >= 1]

            # Calculate remaining funds
            remaining_funds = portfolio_amount - np.dot(latest_prices['Latest Price'], allocation_df['Number of shares to buy'])

            # Generate pie chart
            labels = high_allocation_df['Ticker']
            values = high_allocation_df['Latest Value']
            fig = {
                "data": [dict(type="pie", labels=labels, values=values)],
                "layout": dict(title="Portfolio Allocation")
            }

            # Prepare portfolio information for display
            portfolio_info = html.Div([
                html.H2("Portfolio Allocation"),
                html.H3("Number of shares to buy with the amount of ₹" + str(portfolio_amount)),
                html.P("Funds remaining: ₹" + str(round(remaining_funds, 2))),
                html.P("Volatility: " + str(round(port_volatility*100, 2)) + "%"),
                html.P("Sharpe Ratio: " + str(round(Sharpe_ratio, 6)) + ""),
                html.P("Portfolio Returns: " + str(round(portfolio_return*100, 2)) + "%"),
                html.Table(
                    [
                        html.Thead(html.Tr([html.Th(col) for col in high_allocation_df.columns])),
                        html.Tbody([
                            html.Tr([html.Td(high_allocation_df.iloc[i][col]) for col in high_allocation_df.columns])
                            for i in range(len(high_allocation_df))
                        ])
                    ]
                )
                
            ])

            return portfolio_info, fig

    return "", {}


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
