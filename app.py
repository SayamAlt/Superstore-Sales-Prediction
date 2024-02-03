from flask import Flask, render_template, request
import joblib, warnings
warnings.filterwarnings('ignore')
import pandas as pd

pipeline = joblib.load('pipeline.pkl')
transformer = pipeline.named_steps['transformer']
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['model']

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        profit = float(request.form['profit'])
        discount_percentage = float(request.form['discount_percentage'])
        postal_code = float(request.form['postal_code'])
        discount = float(request.form['discount'])
        order_day = float(request.form['order_day'])
        order_month = float(request.form['order_month'])
        quantity = float(request.form['quantity'])
        operating_expenses = float(request.form['operating_expenses'])
        ship_day = float(request.form['ship_day'])
        order_weekday = float(request.form['order_weekday'])

        data = [[profit,discount_percentage,postal_code,discount,order_day,order_month,quantity,operating_expenses,ship_day,order_weekday]]
        cols = list(['Profit','Discount Percentage','Postal Code','Discount','Order Day','Order Month','Quantity','Operating Expenses','Ship Day','Order Weekday'])
        data = pd.DataFrame(data, columns=cols)
        transformed_data = transformer.transform(data)
        scaled_data = scaler.transform(transformed_data)
        pred = model.predict(scaled_data)[0]
        output = '%.2f' % pred
        return render_template('index.html', prediction_text=f"The predicted sales amount for the superstore, based on the provided order details, is ${output}.")
    
if __name__ == '__main__':
    app.run(port=8000)