from flask import Flask, render_template, request
import pickle
import sklearn
from tensorflow import keras

app = Flask(__name__)
model_lr = pickle.load(open('linear_regression.pkl', 'rb'))
model_nn = keras.models.load_model('neural_net.h5')
model_dt = pickle.load(open('dtree.pkl', 'rb'))
model_lasso = pickle.load(open('Lasso_model.pkl','rb')) #Lasso model added

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form.get('Year'))
        Present_Price = float(request.form.get('Present_Price'))
        Kms_Driven = int(request.form.get('Kms_Driven'))
        Owner = int(request.form.get('Owner'))
        Fuel_Type = int(request.form.get('Fuel_Type_Petrol'))
        car_name = int(request.form.get('Brand'))
        Transmission = float(request.form.get('Transmission_Mannual'))
        Body_type = float(request.form.get('Body_type'))
        warranty = int(request.form.get('Warranty'))

        city = request.form.get('City')

        city_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        city_arr[int(city)] = 1
        input = [[car_name, Year, Fuel_Type, Kms_Driven, Body_type, Transmission, Owner, Present_Price, warranty]]
        for i in city_arr:
            input[0].append(i)
            
        prediction_lr = model_lr.predict(input)
        prediction_nn = model_nn.predict(input)
        prediction_dt = model_dt.predict(input)
        output_lr = round(prediction_lr[0], 2)
        output_nn = round(prediction_nn[0][0], 2)
        output_dt = round(prediction_dt[0], 2)

        price=[]
        price.append(output_lr)
        price.append(output_nn)
        price.append(output_dt)


        min_price = min(price)

        if output_lr < 0:
            return render_template('predict.html', prediction_texts="Sorry you cannot sell this car")
        # else:
        #     return render_template('predict.html', prediction_text_lr="Linear Regression: You can sell the Car at ₹{} ".format(output_lr), prediction_text_nn="Neural Networks: You can sell the Car at ₹{} ".format(output_nn))
        else:return render_template('predict.html', prediction_text_lr="You can sell the Car at approximately ₹{} ".format(min_price))
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
