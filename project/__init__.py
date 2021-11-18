from flask import Flask, render_template, request, url_for
from os import path
from project import ArbitraryStyleTransfer

IMG_WIDTH=IMG_HEIGHT=256
def create_app():

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    from .route import views
    app.register_blueprint(views, url_prefix='/')

    return app
