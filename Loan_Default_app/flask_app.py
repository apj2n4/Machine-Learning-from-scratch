import numpy as np
from nn_sgd import dense_NN_sgd
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class predict_default(Resource):
    def post(self):
        inp_json = request.get_json()
        X = np.array(inp_json['input'])
        nn4 = dense_NN_sgd()
        nn4.load_model(name="dnn_SGD.pkl")
        y_prob_nn4,y_class_nn4 = nn4.predict(X)
        result = {"y_prob":y_prob_nn4[0],"y_class":y_class_nn4[0]}
        
        return result

api.add_resource(predict_default,'/')

if __name__ =='__main__':
    app.run(debug=True)