# coding=utf-8
from flask import Flask
from celery import Celery


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@app.route('/')
def index():
    pass


@app.route('/transform')
def transform():
    pass


@celery.task
def transform_async():
    pass


if __name__ == '__main__':
    app.run()
