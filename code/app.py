# -*- coding: utf-8 -*-
import os
import datetime
import time

from flask import Flask, request, render_template
from flask_wtf import FlaskForm
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, RadioField
from wtforms.validators import DataRequired

app = Flask(__name__)
path_demo = '../datasets/Demo'
app.config['UPLOADED_PHOTOS_DEST'] = path_demo  # 文件储存地址
app.config['SECRET_KEY'] = 'I have a dream'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 5

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app) 


EDSR = {
    'n_blocks': {'2': 12, '4': 12, '2+4': 10},
    'n_feats': 96
}

DBPN = {
    'n_blocks': {'2': 3, '4': 3, '2+4': 2},
    'n_feats': 32,
    'n_init': 128
}
class UploadForm(FlaskForm):
    model = RadioField('Model', choices=[('EDSR', 'EDSR'), ('DBPN', 'DBPN')], validators=[DataRequired()])
    scale = RadioField('Scale', choices=[('2', '2'), ('4', '4'),('2+4', '2+4')], validators=[DataRequired()])
    photo = FileField(validators=[
        FileAllowed(photos, u'只能上传图片！'), 
        FileRequired(u'文件未选择！')])
    submit = SubmitField(u'上传')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    os.system('rm -rf ' + path_demo)
    file_url = None

    if form.is_submitted():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)

        since = time.time()
        selected_model = form.model.data
        selected_scale = form.scale.data

        if selected_model == 'EDSR':
            os.system("python main.py --test --data_test Demo --pre_train ../models/EDSR_{}.pt --n_blocks {} --save --description 'demo' --upscale {} --model EDSR --n_feats {} --result_path ./static/images/".format(selected_scale, EDSR['n_blocks'][selected_scale], selected_scale, int(EDSR['n_feats'])))
        
        elif selected_model == 'DBPN':
            os.system("python main.py --test --data_test Demo --pre_train ../models/DBPN_{}.pt --n_blocks {} --save --description 'demo' --upscale {} --model DBPN --n_feats {} --result_path ./static/images/ --n_init".format(selected_scale, DBPN['n_blocks'][selected_scale], selected_scale, int(DBPN['n_feats']),int(DBPN['n_init'])))

        time_elapsed = time.time() - since 
        
        return render_template('result.html', time = time_elapsed, scale = selected_scale)
        
    return render_template('index.html', form = form)


if __name__ == '__main__':
    
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug = True)