# coding=utf-8
from flask import Flask, request, jsonify
from flask_mail import Mail, Message
from celery import Celery
from os import mkdir, remove
from os.path import join, exists, splitext
import re
import time
import base64
import commands
import json


app = Flask(__name__)
app.config.from_pyfile('default_config.py')
mail = Mail(app)
celery = Celery(
    app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('index.html')


@app.route('/models', methods=['GET'])
def models():
    pass


@app.route('/help', methods=['GET'])
def help():
    return jsonify(status='HELP_SUCCESS', models=list(app.config['MODEL_FILES']), \
        format='/transform: {email:xxx(receive output), filename:xxx(jpg or png), model:xxx(see models), image:xxxx(base64 encode image)}')


@app.route('/transform', methods=['POST'])
def transform():
    # 获取参数
    json_data = json.loads(request.get_data())
    filename = json_data.get(u'filename').encode('ascii')
    image = json_data.get(u'image').encode('ascii')
    email = json_data.get(u'email').encode('ascii')
    model = json_data.get(u'model').encode('ascii')

    print len(image), filename, email, model

    # 检查参数
    if filename is None or image is None or email is None or model is None:
        return jsonify(status='PARAMS ERROR')
    if re.match("^[a-zA-Z0-9_\\-.]+$", filename) is None or splitext(filename)[1] not in app.config['ALLOWED_EXTENSIONS']:
        return jsonify(status='FILENAME NOT SUPPORT')
    if re.match("^.+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)$", email) is None:
        return jsonify(status='EMAIL FORMAT ERROR')
    if model not in app.config['MODEL_FILES']:
        return jsonify(status='MODEL NOT EXISTS')

    try:
        image = base64.b64decode(image)
    except TypeError:
        return jsonify(status='IMAGE ERROR')

    # 保存图片
    filename = str(time.time()) + '-' + filename
    with open(join(app.config['UPLOAD_FOLDER'], filename), 'wb') as f:
        f.write(image)

    # 异步转换
    transform_async.delay(filename, email, model)
    return jsonify(status='SUBMIT_SUCCESS')


@celery.task
def transform_async(filename, email, model):
    # 开始转换
    content_file_path = join(app.config['UPLOAD_FOLDER'], filename)
    model_file_path = join(app.config['MODEL_FOLDER'], model)
    output_folder = app.config['OUTPUT_FOLDER']

    output_filename = filename
    (shotname, extension) = splitext(output_filename)
    output_filename = shotname + '-' + model + extension
    output_file_path = join(output_folder, output_filename)

    command = 'python eval.py --CONTENT_IMAG %s --MODEL_PATH %s -- OUTPUT_FOLDER %s' % (
        content_file_path, model_file_path, output_folder)
    status, output = commands.getstatusoutput(command)

    print status, output

    # 发送邮件
    if status == 0:
        with app.app_context():
            msg = Message("IMAGE-STYLE-TRANSFER",
                          sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = filename
            with app.open_resource(output_file_path) as f:
                mime_type = 'image/jpg' if splitext(
                    filename)[1] is not '.png' else 'image/png'
                msg.attach(filename, mime_type, f.read())
            mail.send(msg)
    else:
        with app.app_context():
            msg = Message("IMAGE-STYLE-TRANSFER",
                          sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = "CONVERT ERROR\n" + filename + "\n HELP - http://host:port/help"
            mail.send(msg)

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

    app.run(host='0.0.0.0', debug=True)
