from flask import Flask, request, render_template, jsonify
from src.DimondPricePrediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.DimondPricePrediction.logger import logging

app = Flask(__name__)

@app.route('/')
def home_page():
    logging.info("Calling index.html page")
    return render_template('index.html')
    

@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        logging.info("Calling form.html page")
        return render_template("form.html")
        
    
    else:
        carat = float(request.form.get('carat'))
        depth = float(request.form.get('depth'))
        table = float(request.form.get('table'))
        x = float(request.form.get('x'))
        y = float(request.form.get('y'))
        z = float(request.form.get('z'))
        cut = request.form.get('cut')
        color = request.form.get('color')
        clarity = request.form.get('clarity')
        data = CustomData(carat,cut,color,clarity,depth,table,x,y,z)
        df = data.get_data_as_dataframe()
        print(df)
        logging.info(f"Getting all the data from the form and converting it into DataFrame:\n {df.to_string()}")

        predict_point = PredictPipeline()
        pred = predict_point.predict(df)
        

        result = round(pred[0], 2)
        logging.info(f"Predicted value is: {result}")

        return render_template('result.html', final_result = result)


if __name__ == "__main__":
    app.run(debug=True)