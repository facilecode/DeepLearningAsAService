from flask import Flask, request
from flask_cors import CORS
from io import BytesIO
import base64
import json
import requests

url = "127.0.0.1"

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/inference', methods=["POST", "GET"])
def receive():

    print("in post req")
    data = None

    if request.method == "POST":

        #data = json.loads(request.get_data())
        data = request.get_data()
        print(data)
        
        #data = json.loads(request.get_data())["image"]
        #print("Data ", data)

        #Thread(target=detect, args=(data,)).start()

        return "received"

app.run(host=url, port=8000)