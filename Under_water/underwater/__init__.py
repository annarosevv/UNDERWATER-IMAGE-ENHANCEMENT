from flask import Flask

app = Flask(__name__)

app.config['UPLOAD_FOLDER']="image/static/uploads"

app.config['SECRET_KEY'] = '8ea2a86e42689205dde0ba81f31138b6'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skin.db'

from underwater import routes

    
app.app_context().push()