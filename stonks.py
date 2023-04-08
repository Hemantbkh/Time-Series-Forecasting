import streamlit as st
import pickle
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
import plotly.express as px

if __name__ == '__main__':
    st.title('Apple Stock Price Forecasting')

    # load the saved model
    with open('apple_stock_model.pkl', 'rb') as f:
        model_fit = pickle.load(f)

    # load the test dataset
    df_test = pd.read_csv('AAPL.csv', parse_dates=['Date'], index_col='Date')

    # Split the data into train and test sets
    Train = df_test.head(1760)
    Test = df_test.tail(251)

    # Add a widget to trigger the forecasting and chart generation
    forecast_period = st.number_input("Enter the number of days to forecast", min_value=1, max_value=None, value=30)
    if st.button('Generate Forecast'):
        # Fit the Prophet model on the training data
        hwe_model_mul_add = ExponentialSmoothing(Train["Close"],seasonal="mul",trend="add",seasonal_periods=251).fit() 

        # Forecast for the next N days
        forecast_dates = pd.date_range(start=df_test.index[-1], periods=forecast_period, freq='D')
        forecast_data = pd.DataFrame(index=forecast_dates, columns=df_test.columns)
        forecast_data['Open'] = df_test['Open'][-1]
        forecast_data['High'] = df_test['High'][-1]
        forecast_data['Low'] = df_test['Low'][-1]
        forecast_data['Close'] = hwe_model_mul_add.forecast(forecast_period)

        # Make predictions for the test data
        Test = Test.sort_index()
        Test.reset_index(drop=True, inplace=True)  # reset index
        test_dates = pd.to_datetime(Test.index)  # convert index to DateTimeIndex
        pred_hwe_mul_add = hwe_model_mul_add.predict(pd.DataFrame(index=test_dates)).values  # create new DataFrame with DateTimeIndex

        # Plot the actual and predicted values for the test data and forecasted values
        pred_data = pd.DataFrame({'Date': Test.index, 'Close': Test.Close, 'Predicted': pred_hwe_mul_add})

        # Add a checkbox for showing the forecast
        show_forecast = st.checkbox('Show forecast')

        # Plot the actual and predicted values for the test data and forecasted values
        pred_data = pd.DataFrame({'Date': Test.index, 'Close': Test.Close, 'Predicted': pred_hwe_mul_add})
        pred_data = pred_data.melt(id_vars='Date', value_vars=['Close', 'Predicted'], var_name='Variable', value_name='Value')
        fig = px.line(pred_data, x='Date', y='Value', color='Variable')

        if show_forecast:
            fig.add_trace(px.line(forecast_data.reset_index().melt(id_vars='Date', value_vars=['Close']), x='Date', y='value').data[0])
        fig.update_layout(title='Apple Stock Prices', xaxis_title='Date', yaxis_title='Price', legend_title='')
        fig.show()
