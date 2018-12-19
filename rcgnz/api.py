import requests
import config
import tensorflow as tf
import Main
from keras.models import load_model
from keras.optimizers import RMSprop

adding_filds = ['number_of_calls', 'call_hour', 'cnvrs_key', 'pst', 'ngt', 'again_key', 'first_ques',
                'stage', 'phone_for_offer', 'mileage', 'serv_hist','city']

model = load_model('char-reg.h5')
model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.005), loss='categorical_crossentropy',
              metrics=['accuracy'])
graph = tf.get_default_graph()

from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/snd', methods=['POST', 'GET'])
def snd():
    pers = {'photo_url': "https://www.pyimagesearch.com/wp-content/uploads/2015/01/google_logo.png"}
    requests.post('https://0d5cbef5.ngrok.io/extract_num', json=pers)


@app.route('/extract_num', methods=['POST', 'GET'])
def extract_num():
    photo_url = request.get_json()['photo_url']

    global graph
    with graph.as_default():
        c, _ = Main.main('http://www.letchworthmini.co.uk/s/cc_images/cache_71011477.JPG')
        c = Main.validate_for_britain(c)
    req = request.get_json()

    for i in range(len(adding_filds)):
        if i < 3:
            req[adding_filds[i]] = 0
        elif i < 7:
            req[adding_filds[i]] = False
        elif i == 7:
            req[adding_filds[i]] = 1
        else:
            req[adding_filds[i]] = None
    req['reg_num'] = c
    del req['photo_url']
    print(req)
    requests.post(config.add_pers_url, json=req)
    # return c


if __name__ == "__main__":
    app.run()
