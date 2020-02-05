from flask import Flask, request, jsonify
import pandas as pd
import pickle
app = Flask(__name__)
pkl_model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict',methods=['GET'])
def predict():
    data = pd.DataFrame(request.json)
    index = data.index
    prediction = pkl_model.predict(data)
    return pd.Series(prediction,name='target',index=index).to_json()
if __name__ == '__main__':       #calls app.run
    app.run(port=8080, debug=True)
    app.run()
