from modules.NhanDien import NhanDien
import cv2
import os
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

nhanDienCayDau = NhanDien(model_type=0)
nhanDienCayNgo = NhanDien(model_type=1)


@app.route('/cay-ngo', methods=['GET', 'POST'])
def cay_ngo():
    if request.method == 'GET':
        return render_template('index.html', data=None)
    f = request.files['fileNgo']
    save_path = f'static/image.png'
    f.save(save_path)
    image = cv2.imread(save_path)
    score, name = nhanDienCayNgo.predict(image)
    score = str(int(score*100))+"%"
    response = {'image_path': save_path,
                'score': score, 'name': name, 'type': 1}
    return render_template('index.html', data=response)


@app.route('/cay-dau', methods=['GET', 'POST'])
def cay_dau():
    if request.method == 'GET':
        return render_template('indexx.html', data=None)
    f = request.files['fileDau']
    save_path = f'static/image.png'
    f.save(save_path)
    image = cv2.imread(save_path)
    score, name = nhanDienCayDau.predict(image)
    score = str(int(score*100))+"%"
    response = {'image_path': save_path,
                'score': score, 'name': name, 'type': 0}
    return render_template('index.html', data=response)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', data=None)


if __name__ == '__main__':
    app.run(debug=True)
