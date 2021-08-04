from flask import Flask, render_template,request
from tensorflow.keras.models import load_model
import pickle 
import os
import tensorflow as tf
graph = tf.compat.v1.get_default_graph()
with open(r'count_vec.pkl','rb') as file:
    cv=pickle.load(file)
cla = load_model('phone.h5')
cla.compile(optimizer='adam',loss='binary_crossentropy')
PEOPLE_FOLDER = os.path.join('static', 'people_photo')
app=Flask(__name__)

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/',methods=["POST"])
#@app.route('/', methods = ['GET','POST'])


#@app.route('/')
#@app.route('/index')
#def show_index():
#    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'up1.png')
 #   return render_template("index.html", user_image = full_filename)


def prediction():
        topic = request.form['review']
        name =request.form['name']
        if (name!=""):
            name=name+"'s"
        print("Hey " + topic)
        topic1=cv.transform([topic])
        
        print("\n"+str(topic1.shape)+"\n")
        with graph.as_default():
             cla = load_model('phone.h5')
             cla.compile(optimizer='adam',loss='binary_crossentropy')
             y_pred = cla.predict(topic1)
             print("pred is "+str(y_pred))
        if(y_pred > 0.5):
            topic3 = os.path.join(app.config['UPLOAD_FOLDER'], 'Positive.gif')
            topic2 = "POSITIVE REVIEW"
        else:
            topic3 = os.path.join(app.config['UPLOAD_FOLDER'], 'Negative.gif')
            topic2 = "NEGATIVE REVIEW"

        return render_template('index.html',ypred = topic2, im =topic3, cont= topic, clna= "ans-container",rev= name+" Review",sub="sub-container scrollable-content")
        


if __name__=='__main__':
    app.run(debug=True)

