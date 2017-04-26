# coding=utf-8
from flask import Flask, request, jsonify
from celery import Celery
import re
from os import mkdir
from os.path import join, exists
import time
import base64
import commands

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])


@app.route('/transform', methods=['POST'])
def transform():
    # 获取参数
    json_data = request.get_json()
    filename = json_data.get('filename')
    image = json_data.get('image')
    email = json_data.get('email')
    model = json_data.get('model')

    # 检查参数
    if filename is None or image is None or email is None or model is None:
        return jsonify(status='PARAMS ERROR')
    if os.path.splitext(filename)[1] not in app.config['ALLOWED_EXTENSIONS']:
        return jsonify(status='FILENAME NOT SUPPORT')
    if re.match("^.+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)$", email) is None:
        return jsonify(status='EMAIL FORMAT ERROR')
    if model not in app.config['MODEL_FILES']:
        return jsonify(status='MODEL NOT EXISTS')

    try:
        image = base64.b64decode(image)
    except TypeError:
        return jsonify(status='IMAGE ERROR')

    # 异步转换
    transform_async.delay(filename, image, email, model)
    return jsonify(status='SUBMIT_SUCCESS')


@celery.task
def transform_async(filename, image, email, model):
    # 保存图片
    filename = str(time.time()) + '-' + filename
    with open(join(app.config['UPLOAD_FOLDER'], filename)) as f:
        f.write(image)

    # 执行任务
    content_file_path = join(app.config['UPLOAD_FOLDER'], filename)
    model_file_path = join(app.config['MODEL_FOLDER'], model)
    output_folder = app.config['OUTPUT_FOLDER']
    command = 'python eval.py --CONTENT_IMAG %s --MODEL_PATH %s -- OUTPUT_FOLDER %s' % (
        content_file_path, model_file_path, output_folder)
    status, _ = commands.getstatusoutput(command)

    # 返回状态
    if status == 0:
        pass
    else:
        pass


@app.route('/status', methods=['GET'])
def status():
    pass


def config(app):
    app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
    app.config['UPLOAD_FOLDER'] = 'raw-images/'
    app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

    app.config['MODEL_FOLDER'] = 'models/'
    app.config['MODEL_FILES'] = set(['fast-style-transfer.ckpt-done'])
    app.config['OUTPUT_FOLDER'] = 'output-images/'

    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    celery.conf.update(app.config)

    if not exists(app.config['UPLOAD_FOLDER']):
        mkdir(app.config['UPLOAD_FOLDER'])
    if not exists(app.config['OUTPUT_FOLDER']):
        mkdir(app.config['OUTPUT_FOLDER'])


if __name__ == '__main__':
    config(app)
    app.run()
