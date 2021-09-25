from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

#dic = {0 : 'Bad Cars', 1 : 'Good Cars'}

model = load_model('Cardetect.h5')

model.make_predict_function()

def predict_label(img_path):
		i = image.load_img(img_path,target_size=(150,150))
		i = image.img_to_array(i)/255
		i = np.expand_dims(i, axis=0)
		i = np.vstack([i])
		p = model.predict(i)
		#return dic[p[0]]
		if p >= 0.4:
			return "Good Car"
		
		return "Damaged Car"


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/Readme")
def about_page():
	return "This was a tough task for me but i am glad i did it! I will update this Readme page soon. Ibukunoluwa"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "Test/" + img.filename	
		img.save(img_path)

		print(img_path)
		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)

@app.route("/Test/<path:imageName>", methods=['GET'])
def getImages(imageName):
	return send_from_directory('Test',f'{imageName}')

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)