# coding=utf-8
from flask import Flask, request, jsonify
from flask_mail import Mail, Message
from celery import Celery
from os import mkdir, remove, environ
from os.path import join, exists, splitext
import re
import time
import base64
import commands
import sys

# 解决编码问题
reload(sys)
sys.setdefaultencoding('utf-8')

app = Flask(__name__)
app.config.from_pyfile('default_config.py')
app.config['MAIL_USERNAME'] = environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = environ.get('MAIL_PASSWORD')
mail = Mail(app)
celery = Celery(
    app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])


@app.route('/transform', methods=['POST'])
def transform():
    # 获取参数
    json_data = request.get_json() 
    filename = json_data.get('filename')
    image = json_data.get('image')
    email = json_data.get('email')
    model = json_data.get('model')

    print json_data

    # 检查参数
    if filename is None or image is None or email is None or model is None:
        return jsonify(status='PARAMS ERROR')
    if splitext(filename)[1] not in app.config['ALLOWED_EXTENSIONS']:
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
    with open(join(app.config['UPLOAD_FOLDER'], filename), 'wb') as f:
        f.write(image)

    # 转换图片
    content_file_path = join(app.config['UPLOAD_FOLDER'], filename)
    model_file_path = join(app.config['MODEL_FOLDER'], model)
    output_folder = app.config['OUTPUT_FOLDER']
    output_file_path = join(output_folder, filename)
    command = 'python eval.py --CONTENT_IMAG %s --MODEL_PATH %s -- OUTPUT_FOLDER %s' % (
        content_file_path, model_file_path, output_folder)
    status, output = commands.getstatusoutput(command)

    print status, output

    # 发送邮件
    if status == 0:
        msg = Message("IMAGE-STYLE-TRANSFER",
                      sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = filename
        with app.open_resource(output_file_path) as f:
            mime_type = 'image/jpg' if splitext(
                filename)[1] is not '.png' else 'image/png'
            msg.attach(filename, mime_type, f.read())
        with app.app_context():
            mail.send(msg)
    else:
        pass

    remove_files.apply_async(
        args=[[content_file_path, output_file_path]], countdown=60)


@celery.task
def remove_files(file_list):
    for file in file_list:
        if exists(file):
            remove(file)


if __name__ == '__main__':
    if not exists(app.config['UPLOAD_FOLDER']):
        mkdir(app.config['UPLOAD_FOLDER'])
    if not exists(app.config['OUTPUT_FOLDER']):
        mkdir(app.config['OUTPUT_FOLDER'])

    app.run()
