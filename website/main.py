from flask import Flask, render_template, request
from googlemaps import Client
from numpy import exp
from sklearn.externals import joblib

with open('api_key.txt', 'r') as file: # Google Maps API Key
    api_key = file.read()

app = Flask(__name__)

def comma(num):
    if type(num)==int:
        return '{:,}'.format(num)
    elif type(num)==float:
        return '{:,.2f}'.format(num)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def my_form_post():
    street = request.form['street']
    number = request.form['number']
    return train(street, number)

@app.route('/train', methods=['GET'])
def train(street, number):
    address = '{} {}, NYC'.format(street, number)
    coordinates = client.geocode(address)[0]['geometry']['location']
    latitude, longitude = coordinates['lat'], coordinates['lng']
    prediction = comma(int(exp(reg.predict([float(latitude), float(longitude), 1, 0, 1])[0]))*1000)
    text = '$ {}'.format(prediction)
    return render_template('train.html', prediction=text, latitude=latitude, longitude=longitude)

if __name__ == '__main__':
	try:
		client = Client(api_key)
		reg = joblib.load('model/model.pkl')
		print('model loaded')

	except Exception as e:
	    print('No model here')
	    print('Train first')
	    clf = None

	app.run(debug=True)