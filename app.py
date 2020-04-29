from flask import Flask, render_template, request
from searchModule import Predict
import os
app = Flask(__name__)

@app.route('/')
def hello():
	return render_template('index.html')


@app.route('/',methods=['POST'])
def hi():
	if request.method == 'POST':
		f = request.files['userfile']
		path = "./static/{}".format(f.filename)# ./static/images.jpg
		f.save(path)
		prediction = "alai_minar"
		prediction = Predict(path)
		os.remove(path)
		return render_template('index.html',your_prediction=prediction)

if __name__ == '__main__':
	app.run(debug=True)