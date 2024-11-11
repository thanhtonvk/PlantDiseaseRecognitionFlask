from modules.NhanDien import NhanDien
from modules.NhanDienLaCam import predictLaCam
from modules.NhanDienQuaCam import predictQuaCam
import cv2
import os
from flask import Flask, render_template, request, redirect
from unidecode import unidecode

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

@app.route('/la-cam', methods = ['GET', 'POST'])
def la_cam():
    if request.method == 'GET':
        return render_template('index.html', data = None)
    f = request.files['fileLaCam']
    save_path = f'static/image.png'
    f.save(save_path)
    image = cv2.imread(save_path)

    rectangle_color = (0, 255, 0)  # Green color
    rectangle_thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (0, 0, 255)  # Red color
    text_thickness = 2
    boxes,classes,scores = predictLaCam(image)
    result = ''
    for box,cls,score in zip(boxes,classes, scores):
        xmin,ymin,xmax,ymax = box
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(image, top_left, bottom_right, rectangle_color, rectangle_thickness)
        text_position = (xmin, ymin-10)
        cv2.putText(image, unidecode(cls), text_position, font, font_scale, text_color, text_thickness)
        result+=f'{cls} - {int(score*100)}% \n'
    cv2.imwrite(save_path,image)
    image_base64 = encode_image(save_path)
    response = {'image_path': image_base64,
                'result': result, 'type': 2}
    return render_template('index.html', data=response) 
        
@app.route('/qua-cam', methods = ['GET', 'POST'])
def qua_cam():
    if request.method == 'GET':
        return render_template('index.html', data = None)
    f = request.files['fileQuaCam']
    save_path = f'static/image.png'
    f.save(save_path)
    image = cv2.imread(save_path)

    rectangle_color = (0, 255, 0)  # Green color
    rectangle_thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (0, 0, 255)  # Red color
    text_thickness = 2
    boxes,classes,scores = predictQuaCam(image)
    result = ''
    for box,cls,score in zip(boxes,classes, scores):
        xmin,ymin,xmax,ymax = box
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(image, top_left, bottom_right, rectangle_color, rectangle_thickness)
        text_position = (xmin, ymin-10)
        cv2.putText(image, unidecode(cls), text_position, font, font_scale, text_color, text_thickness)
        result+=f'{cls} - {int(score*100)}% \n'
    cv2.imwrite(save_path,image)
    image_base64 = encode_image(save_path)
    response = {'image_path': image_base64,
                'result': result, 'type': 3}
    return render_template('index.html', data=response) 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', data=None)


if __name__ == '__main__':
    app.run(debug=True)
