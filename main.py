from modules.NhanDien import NhanDien
from modules.NhanDienLaCam import predictLaCam
from modules.NhanDienQuaCam import predictQuaCam
from modules.NhanDienCayQue import predictCayQue
from modules.NhanDienCayCafe import predictCafe
import cv2
import os
from flask import Flask, render_template, request, redirect, jsonify
from unidecode import unidecode
app = Flask(__name__)
CAY_DAU = ["Đốm lá góc cạnh", "Rỉ sét", "Khoẻ mạnh"]
CAY_NGO = ["Cháy lá", "Rỉ sét thông thường", "Đốm lá xám", "Khoẻ mạnh"]
nhanDienCayDau = NhanDien(model_type=0)
nhanDienCayNgo = NhanDien(model_type=1)

f = open("chua_benh/cafe.txt", "r", encoding="utf-8")
lines = f.read()
chua_benh_cafe = lines.split("#")
f.close()


f = open("chua_benh/cay_dau.txt", "r", encoding="utf-8")
lines = f.read()
chua_benh_cay_dau = lines.split("#")
f.close()

f = open("chua_benh/cay_ngo.txt", "r", encoding="utf-8")
lines = f.read()
chua_benh_cay_ngo = lines.split("#")
f.close()

f = open("chua_benh/cay_que.txt", "r", encoding="utf-8")
lines = f.read()
chua_benh_cay_que = lines.split("#")
f.close()

f = open("chua_benh/la_cam.txt", "r", encoding="utf-8")
lines = f.read()
chua_benh_la_cam = lines.split("#")
f.close()

f = open("chua_benh/qua_cam.txt", "r", encoding="utf-8")
lines = f.read()
chua_benh_qua_cam = lines.split("#")
f.close()

import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@app.route("/cay-ngo", methods=["GET", "POST"])
def cay_ngo():
    if request.method == "GET":
        return render_template("index.html", data=None)
    f = request.files["fileNgo"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)
    scores = nhanDienCayNgo.predict(image)
    score_max = max(scores)
    max_idx = scores.index(score_max)
    result = f"{CAY_NGO[max_idx]} : {score_max}%"
    image_base64 = encode_image(save_path)
    response = {
        "image_path": image_base64,
        "result": result,
        "type": 1,
        "chua_benh": chua_benh_cay_ngo[max_idx],
    }
    return render_template("index.html", data=response)


@app.route("/cay-dau", methods=["GET", "POST"])
def cay_dau():
    if request.method == "GET":
        return render_template("index.html", data=None)
    f = request.files["fileDau"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)
    scores = nhanDienCayDau.predict(image)
    score_max = max(scores)
    max_idx = scores.index(score_max)
    result = f"{CAY_DAU[max_idx]} : {score_max}%"
    image_base64 = encode_image(save_path)
    response = {
        "image_path": image_base64,
        "result": result,
        "type": 0,
        "chua_benh": chua_benh_cay_dau[max_idx],
    }
    return render_template("index.html", data=response)


@app.route("/la-cam", methods=["GET", "POST"])
def la_cam():
    if request.method == "GET":
        return render_template("index.html", data=None)
    f = request.files["fileLaCam"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)

    rectangle_color = (0, 255, 0)  # Green color
    rectangle_thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (0, 0, 255)  # Red color
    text_thickness = 2
    boxes, classes, scores,nhan_label = predictLaCam(image)
    result = ""
    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(
            image, top_left, bottom_right, rectangle_color, rectangle_thickness
        )
        text_position = (xmin, ymin - 10)
        cv2.putText(
            image,
            unidecode(cls),
            text_position,
            font,
            font_scale,
            text_color,
            text_thickness,
        )
        result += f"{cls} - {int(score*100)}% \n"
    if len(classes) > 0:
        chua_benh = chua_benh_la_cam[nhan_label[0]]
    else:
        chua_benh = "Khoẻ mạnh"
    cv2.imwrite(save_path, image)
    image_base64 = encode_image(save_path)
    response = {
        "image_path": image_base64,
        "result": result,
        "type": 2,
        "chua_benh": chua_benh,
    }
    return render_template("index.html", data=response)


@app.route("/qua-cam", methods=["GET", "POST"])
def qua_cam():
    if request.method == "GET":
        return render_template("index.html", data=None)
    f = request.files["fileQuaCam"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)

    rectangle_color = (0, 255, 0)  # Green color
    rectangle_thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (0, 0, 255)  # Red color
    text_thickness = 2
    boxes, classes, scores,nhan_label = predictQuaCam(image)
    result = ""
    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(
            image, top_left, bottom_right, rectangle_color, rectangle_thickness
        )
        text_position = (xmin, ymin - 10)
        cv2.putText(
            image,
            unidecode(cls),
            text_position,
            font,
            font_scale,
            text_color,
            text_thickness,
        )
        result += f"{cls} - {int(score*100)}% \n"
    if len(classes) > 0:
        chua_benh = chua_benh_qua_cam[nhan_label[0]]
    else:
        chua_benh = "Khoẻ mạnh"
    cv2.imwrite(save_path, image)
    image_base64 = encode_image(save_path)
    response = {
        "image_path": image_base64,
        "result": result,
        "type": 3,
        "chua_benh": chua_benh,
    }
    return render_template("index.html", data=response)


@app.route("/cay-que", methods=["GET", "POST"])
def cay_que():
    if request.method == "GET":
        return render_template("index.html", data=None)
    f = request.files["fileCayQue"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)

    rectangle_color = (0, 255, 0)  # Green color
    rectangle_thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (0, 0, 255)  # Red color
    text_thickness = 2
    boxes, classes, scores,nhan_label = predictCayQue(image)
    result = ""
    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(
            image, top_left, bottom_right, rectangle_color, rectangle_thickness
        )
        text_position = (xmin, ymin - 10)
        cv2.putText(
            image,
            unidecode(cls),
            text_position,
            font,
            font_scale,
            text_color,
            text_thickness,
        )
        result += f"{cls} - {int(score*100)}% \n"
    if len(classes) > 0:
        chua_benh = chua_benh_cay_que[nhan_label[0]]
    else:
        chua_benh = "Khoẻ mạnh"
    cv2.imwrite(save_path, image)
    image_base64 = encode_image(save_path)
    response = {
        "image_path": image_base64,
        "result": result,
        "type": 4,
        "chua_benh": chua_benh,
    }
    return render_template("index.html", data=response)


@app.route("/la-cafe", methods=["GET", "POST"])
def la_cafe():
    if request.method == "GET":
        return render_template("index.html", data=None)
    f = request.files["fileCafe"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)

    rectangle_color = (0, 255, 0)  # Green color
    rectangle_thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (0, 0, 255)  # Red color
    text_thickness = 2
    boxes, classes, scores,nhan_label = predictCafe(image)
    result = ""
    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(
            image, top_left, bottom_right, rectangle_color, rectangle_thickness
        )
        text_position = (xmin, ymin - 10)
        cv2.putText(
            image,
            unidecode(cls),
            text_position,
            font,
            font_scale,
            text_color,
            text_thickness,
        )
        result += f"{cls} - {int(score*100)}% \n"
    if len(classes) > 0:
        chua_benh = chua_benh_cafe[nhan_label[0]]
    else:
        chua_benh = "Khoẻ mạnh"
    cv2.imwrite(save_path, image)
    image_base64 = encode_image(save_path)
    response = {
        "image_path": image_base64,
        "result": result,
        "type": 5,
        "chua_benh": chua_benh,
    }
    return render_template("index.html", data=response)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", data=None)


@app.route("/api/cay-ngo", methods=["POST"])
def api_cay_ngo():
    f = request.files["image"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)
    scores = nhanDienCayNgo.predict(image)
    score_max = max(scores)
    max_idx = scores.index(score_max)
    result = f"{CAY_NGO[max_idx]} : {score_max}%"
    return result


@app.route("/api/cay-dau", methods=["POST"])
def api_cay_dau():
    f = request.files["image"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)
    scores = nhanDienCayDau.predict(image)
    score_max = max(scores)
    max_idx = scores.index(score_max)
    result = f"{CAY_DAU[max_idx]} : {score_max}%"
    return result


@app.route("/api/la-cam", methods=["POST"])
def api_la_cam():
    f = request.files["image"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)
    boxes, classes, scores = predictLaCam(image)
    if len(classes) > 0:
        result += f"{classes[0]} - {int(scores[0]*100)}% \n"
    else:
        result = ""
    return result


@app.route("/api/qua-cam", methods=["POST"])
def api_qua_cam():
    f = request.files["image"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)
    boxes, classes, scores = predictQuaCam(image)
    result = ""
    for box, cls, score in zip(boxes, classes, scores):
        result += f"{cls} - {int(score*100)}% \n"
    return result


@app.route("/api/cay-que", methods=["POST"])
def api_cay_que():
    f = request.files["image"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)
    boxes, classes, scores = predictCayQue(image)
    result = ""
    for box, cls, score in zip(boxes, classes, scores):
        result += f"{cls} - {int(score*100)}% \n"
    return result


@app.route("/api/cay-cafe", methods=["POST"])
def api_cay_cafe():
    f = request.files["image"]
    save_path = f"static/image.png"
    f.save(save_path)
    image = cv2.imread(save_path)
    boxes, classes, scores = predictCafe(image)
    result = ""
    for box, cls, score in zip(boxes, classes, scores):
        result += f"{cls} - {int(score*100)}% \n"
    return result


import socket


def get_local_ipv4():
    try:
        # Tạo kết nối giả để tìm địa chỉ IP thực
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ipv4 = s.getsockname()[0]
        s.close()
        return ipv4
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    ipv4 = get_local_ipv4()
    app.run(host=ipv4, port=5000, debug=True)
