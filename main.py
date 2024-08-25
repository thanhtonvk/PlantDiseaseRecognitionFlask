from modules.NhanDien import NhanDien
import cv2
import os
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
CAY_DAU = ["Đốm lá góc cạnh", "Rỉ sét", "Khoẻ mạnh"]
CAY_NGO = ["Cháy lá", "Rỉ sét thông thường", "Đốm lá xám", "Khoẻ mạnh"]
nhanDienCayDau = NhanDien(model_type=0)
nhanDienCayNgo = NhanDien(model_type=1)
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.route('/cay-ngo', methods=['GET', 'POST'])
def cay_ngo():
    if request.method == 'GET':
        return render_template('index.html', data=None)
    f = request.files['fileNgo']
    save_path = f'static/image.png'
    f.save(save_path)
    image = cv2.imread(save_path)
    scores = nhanDienCayNgo.predict(image)
    score_max = max(scores)
    max_idx = scores.index(score_max)
    result = ''
    if max_idx == 3:
        result = f"{CAY_NGO[max_idx]} : {score_max}%"
    else:
        for i in range(len(scores)-1):
            result += f"{CAY_NGO[i]} : {scores[i]}%\n"
    image_base64 = encode_image(save_path)
    response = {'image_path': image_base64,
                'result': result, 'type': 1}
    return render_template('index.html', data=response)


@app.route('/cay-dau', methods=['GET', 'POST'])
def cay_dau():
    if request.method == 'GET':
        return render_template('index.html', data=None)
    f = request.files['fileDau']
    save_path = f'static/image.png'
    f.save(save_path)
    image = cv2.imread(save_path)
    scores = nhanDienCayDau.predict(image)
    score_max = max(scores)
    max_idx = scores.index(score_max)
    result = ''
    if max_idx == 2:
        result = f"{CAY_DAU[max_idx]} : {score_max}%"
    else:
        for i in range(len(scores)-1):
            result += f"{CAY_DAU[i]} : {scores[i]}%\n"
    image_base64 = encode_image(save_path)
    response = {'image_path': image_base64,
                'result': result, 'type': 0}
    return render_template('index.html', data=response)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', data=None)


if __name__ == '__main__':
    app.run(debug=True)
