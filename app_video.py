from __future__ import division , print_function
import os
import numpy as np
from tensorflow.keras import backend as K
K.clear_session()

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask , request , render_template
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')

MODEL_PATH = r"F:\Python\Projects\models_app\my_model.h5"

model= load_model(MODEL_PATH)
print('Model loaded. Start serving')

def model_predict(img_path , model):
    img = image.load_img(img_path , target_size=(50,50))
    
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    pred = np.argmax(preds, axis=1)
    return pred

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        #get the file from post request
        f= request.files['file']
        
        #save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        #make prediction
        pred = model_predict(file_path, model)
        os.remove(file_path)
        
        #arrange the correct return according to the model
        #in this model 1 is pneumonia and 0 is normal
        return render_template('base.html', data=pred)
    else:
        None
if __name__ == '__main__':
    app.run(debug=True)
    
    
    
