from flask import Flask,render_template,request
import pickle
import numpy as np
import joblib

# Loading the model
#model = joblib.load('forest.pkl')

model=pickle.load(open('logistic.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('mlupload.html')
#
@app.route('/predict',methods=['POST'])
def predict():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11= request.form['k']
    data12= request.form['l']
    data13= request.form['m']
    data14= request.form['n']
    data15= request.form['o']
    data16 = request.form['p']
    data17= request.form['q']
    data18= request.form['r']
    data19= request.form['s']
    data20= request.form['t']
    data21= request.form['u']
    data22= request.form['v']
    data23= request.form['w']
    data24= request.form['x']
    data25= request.form['y']
    data26= request.form['z']
    data27= request.form['a1']
    data28= request.form['b1']
    data29= request.form['c1']
    data30= request.form['d1']
    arr = [data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23,data24,data25,data26,data27,data28,data29,data30]
    #arr=[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601]
    int_str=list(map(float,arr))
    a=np.array(int_str)
    pred = model.predict([a])
    return render_template('result.html', data=pred)

# Remove app.run() and add the following block
if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0', port=5000)  # You can specify the host and port here if needed

