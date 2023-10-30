from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained XGBoost model
loaded_model = pickle.load(open('pharma_model.sav', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict_sales():
    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        drug = int(request.form['drug'])

        # Generate the sales predictions
        df_sales_predictions = predict_sales(start_date, end_date, drug)

        # Extract the predictions as a list
        sales_predictions = df_sales_predictions['predicted_quantity'].tolist()

        return render_template('index.html', sales_predictions=sales_predictions)

    return render_template('index.html')

def predict_sales(start_date, end_date, drug):  # Corrected function name
    # Generate a range of dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create the DataFrame with dates as the index
    df_test = pd.DataFrame(index=dates)
    df_test['Year'] = df_test.index.year
    df_test['Month'] = df_test.index.month
    df_test['Weekday Name'] = df_test.index.weekday
    df_test['day'] = df_test.index.day
    df_test['Drug'] = drug
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # df_test['Weekday Name'] = le.fit_transform(df_test['Weekday Name'])
    df_test['predicted_quantity'] = loaded_model.predict(df_test)
    return df_test
    
if __name__ == '__main__':
    app.run(debug=True)
