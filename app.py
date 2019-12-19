import os
import time
from datetime import timedelta

import cv2
from flask import Flask, request, jsonify, make_response
from flask import render_template
import config
from werkzeug.utils import secure_filename
from model import *
from colorlization  import *


app = Flask(__name__)
app.config.from_object(config)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=0)
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = False

@app.route('/')
def index():
    return render_template("index.html")


"""
处理图片的显示，使用到了ajax技术
"""
# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/color/", methods=['GET','POST'])
def coloring():
    if request.method == 'POST':
        print(request.files)
        f = request.files['file0']
        if not (f and allowed_file((f.filename))):
            return jsonify({'error': '1001', 'msg': "请检查上传的文件类型"})
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        # upload_path = os.path.join(basepath, r'static\images', secure_filename(f.filename))
        # upload_path = os.path.join(basepath, r'static\imgs\images', "test.jpg")
        upload_path = os.path.join(basepath, r'static/imgs/images', f.filename)

        f.save(upload_path)
        name = f.filename.split('.')[0]
        print(name)
        img = cv2.imread(upload_path)
        src1 = r"static/imgs/images/" + name + '.jpg'
        src2 = r"static/imgs/images/" +"g_" + name + '.jpg'
        tsc1 = r"imgs/images/" + name + '.jpg'
        tsc2 = r"imgs/images/" + "g_" + name + '.jpg'
        cv2.imwrite(os.path.join(basepath, src1), img)
        gan = Pix2Pix()

        print(tsc1, tsc2)
        predict_single_image(gan, src1, src2)
        return render_template('coloring_ok.html' ,src1=tsc1, src2=tsc2, val1=time.time())

    else:
        return render_template("coloring.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", threaded=False)
