from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model with the correct dtype
with open('forest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the expected dtype for the model's tree nodes
expected_dtype = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples', 'missing_go_to_left'],
    'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'],
    'offsets': [0, 8, 16, 24, 32, 40, 48, 56],
    'itemsize': 64
})

# Check and convert the dtype of the model's tree nodes if needed
if model.tree_.dtype != expected_dtype:
    model.tree_ = model.tree_.astype(expected_dtype)

@app.route('/')
def hello():
    return render_template('mlupload.html')

@app.route('/predict', methods=['POST'])
def predict():
    arr=[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601]
    int_str = list(map(float, arr))
    a = np.array(int_str)
    pred = model.predict([a])
    return render_template('result.html', data=pred)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
