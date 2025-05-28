from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

df = pd.read_csv("data/stores_sales_forecasting.csv", encoding='latin1')
df.drop(columns=['Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment','Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name'], inplace=True)
df.drop_duplicates(inplace=True)
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.set_index('Order Date', inplace=True)
monthly_sales = df['Sales'].resample('ME').sum()

# load model
model = joblib.load("model/furniture_sales_forecast_model")

# init fastapi
app = Flask(__name__)
CORS(app)

def get_forecast(months):
    past_dates = monthly_sales.index.strftime('%Y-%m-%d').tolist()
    past_sales = [round(x) for x in monthly_sales.values.tolist()]
    future_dates = pd.date_range(start='2018-01-31', periods=months, freq='ME').strftime('%Y-%m-%d').tolist()
    future_sales = [round(x) for x in model.forecast(steps=months).tolist()]

    return past_dates, past_sales, future_dates, future_sales

@app.route('/forecast', methods=['POST'])
def forecast_sales():
    data = request.get_json()
    months = int(data['months'])
    past_dates, past_sales, future_dates, future_sales = get_forecast(months)

    response = jsonify({
        'past_dates': past_dates,
        'past_sales': past_sales,
        'future_dates': future_dates,
        'future_sales': future_sales
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0')
    result = get_forecast(12)
    print(result)
