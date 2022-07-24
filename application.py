import random
import os
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf
import plotly.graph_objs as go
import nibabel as nib
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import config
from Model import DiceCoefficientLoss
import plotly

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import datetime
import json
import io
import base64
from base64 import decodestring

"""#### Loading model"""


def dice(y_true, y_pred):
    # computes the dice score on two tensors

    sum_p = K.sum(y_pred, axis=0)
    sum_r = K.sum(y_true, axis=0)
    sum_pr = K.sum(y_true * y_pred, axis=0)
    dice_numerator = 2 * sum_pr
    dice_denominator = sum_r + sum_p
    dice_score = (dice_numerator + K.epsilon()) / (dice_denominator + K.epsilon())
    return dice_score


def dice_whole_metric(y_true, y_pred):
    # computes the dice for the whole tumor

    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))
    y_whole = K.sum(y_true_f[..., 1:], axis=1)
    p_whole = K.sum(y_pred_f[..., 1:], axis=1)
    dice_whole = dice(y_whole, p_whole)
    return dice_whole


def dice_en_metric(y_true, y_pred):
    # computes the dice for the enhancing region

    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))
    y_enh = y_true_f[:, -1]
    p_enh = y_pred_f[:, -1]
    dice_en = dice(y_enh, p_enh)
    return dice_en


def dice_core_metric(y_true, y_pred):
    ##computes the dice for the core region

    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))

    # workaround for tf
    # y_core=K.sum(tf.gather(y_true_f, [1,3],axis =1),axis=1)
    # p_core=K.sum(tf.gather(y_pred_f, [1,3],axis =1),axis=1)

    y_core = K.sum(y_true_f[:, 2:], axis=1)
    p_core = K.sum(y_pred_f[:, 2:], axis=1)
    dice_core = dice(y_core, p_core)
    return dice_core

def gen_dice_score(y_true, y_pred):
  y_true_f = K.reshape(y_true,shape=(-1,4))
  y_pred_f = K.reshape(y_pred,shape=(-1,4))
  sum_p=K.sum(y_pred_f,axis=-2)
  sum_r=K.sum(y_true_f,axis=-2)
  sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
  weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
  generalised_dice_numerator =2*K.sum(weights*sum_pr)
  generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
  generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
  return generalised_dice_score


def gen_dice_loss(y_true, y_pred):
  return 1 - gen_dice_score(y_true, y_pred)


# model_path=config.MODEL_PATH

model = tf.keras.models.load_model('finalvalaug.h5', custom_objects={'gen_dice_loss': gen_dice_loss, 'dice_whole_metric':dice_whole_metric, 'dice_en_metric': dice_en_metric, 'dice_core_metric': dice_core_metric})

"""### Prediction """


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    return out


def normalize(image):
    img1 = itensity_normalize_one_volume(image[..., 0])
    img2 = itensity_normalize_one_volume(image[..., 1])
    img3 = itensity_normalize_one_volume(image[..., 2])
    img4 = itensity_normalize_one_volume(image[..., 3])
    img = np.stack((img1, img2, img3, img4), axis=-1)
    return img


def input_image(image):
    image_path = os.path.join(config.IMAGES_DATA_DIR, image)
    img = nib.load(image_path)
    image_data = img.dataobj
    image_data = np.asarray(image_data)

    image_data = image_data[34:194, 22:214, 13:141, ]
    image_data = normalize(image_data)
    # Reshaping the Input Image and Ground Truth(Mask)
    reshaped_image_data=image_data.reshape(1,160,192,128,4)

    print(reshaped_image_data.shape)
    print(type(reshaped_image_data))

    # Prediction - Our Segmentation
    Y_hat = model.predict(x=reshaped_image_data)
    Y_hat = np.argmax(Y_hat, axis=-1)
    print(f"Y_hat shape - {Y_hat.shape}")

    # Read the Input Image and Predicted Mask
    image = reshaped_image_data[0, :, :, :, 0].T
    mask = Y_hat[0].T

    # For Colorscale
    pl_bone=[[0.0, 'rgb(0, 0, 0)'],
             [0.05, 'rgb(10, 10, 14)'],
             [0.1, 'rgb(21, 21, 30)'],
             [0.15, 'rgb(33, 33, 46)'],
             [0.2, 'rgb(44, 44, 62)'],
             [0.25, 'rgb(56, 55, 77)'],
             [0.3, 'rgb(66, 66, 92)'],
             [0.35, 'rgb(77, 77, 108)'],
             [0.4, 'rgb(89, 92, 121)'],
             [0.45, 'rgb(100, 107, 132)'],
             [0.5, 'rgb(112, 123, 143)'],
             [0.55, 'rgb(122, 137, 154)'],
             [0.6, 'rgb(133, 153, 165)'],
             [0.65, 'rgb(145, 169, 177)'],
             [0.7, 'rgb(156, 184, 188)'],
             [0.75, 'rgb(168, 199, 199)'],
             [0.8, 'rgb(185, 210, 210)'],
             [0.85, 'rgb(203, 221, 221)'],
             [0.9, 'rgb(220, 233, 233)'],
             [0.95, 'rgb(238, 244, 244)'],
             [1.0, 'rgb(255, 255, 255)']]

    r,c = image[0].shape
    n_slices = image.shape[0]
    height = (image.shape[0]-1) / 10
    grid = np.linspace(0, height, n_slices)
    slice_step = grid[1] - grid[0]

    rm,cm = mask[0].shape
    nm_slices = mask.shape[0]
    height_m = (mask.shape[0]-1) / 10
    grid_m = np.linspace(0, height_m, nm_slices)
    slice_step_m = grid_m[1] - grid_m[0]

    initial_slice = go.Surface(
                         z=height*np.ones((r,c)),
                         surfacecolor=np.flipud(image[-1]),
                         colorscale=pl_bone,
                         showscale=False)

    initial_slice_m = go.Surface(
                         z=height_m*np.ones((rm,cm)),
                         surfacecolor=np.flipud(mask[-1]),
                         colorscale=pl_bone,
                         showscale=False)

    frames = [go.Frame(data=[dict(type='surface',
                              z=(height-k*slice_step)*np.ones((r,c)),
                              surfacecolor=np.flipud(image[-1-k]))],
                              name=f'frame{k+1}') for k in range(1, n_slices)]

    frames_m = [go.Frame(data=[dict(type='surface',
                              z=(height_m-k*slice_step_m)*np.ones((rm,cm)),
                              surfacecolor=np.flipud(mask[-1-k]))],
                              name=f'frame{k+1}') for k in range(1, nm_slices)]

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [dict(steps = [dict(method= 'animate',
                                  args= [[f'frame{k+1}'],
                                        dict(mode= 'immediate', frame= dict(duration=20, redraw= True),transition=dict(duration= 0))
                                        ],
                                  label=f'{k+1}'
                                  )for k in range(n_slices)],
                    active=17,
                    transition= dict(duration= 0),
                    x=0, # slider starting position
                    y=0,
                    currentvalue=dict(font=dict(size=12),
                                      prefix='slice: ',
                                      visible=True,
                                      xanchor= 'center'
                                     ),
                   len=1.0) #slider length
               ]

    layout3d = dict(title_text='Slices of Brain in volumetric data: Input Image', title_x=0.5,
                    template="plotly_dark",
                    width=600,
                    height=600,
                    scene_zaxis_range=[-0.1, 12.8],
                    updatemenus = [
                        {
                            "buttons": [
                                {
                                    "args": [None, frame_args(50)],
                                    "label": "&#9654;", # play symbol
                                    "method": "animate",
                                },
                                {
                                    "args": [[None], frame_args(0)],
                                    "label": "&#9724;", # pause symbol
                                    "method": "animate",
                                },
                            ],
                            "direction": "left",
                            "pad": {"r": 0, "t": 60},
                            "type": "buttons",
                            "x": 0,
                            "y": 0,
                        }
                     ],
                     sliders=sliders
                )

    layout3d_m = dict(title_text='Slices of Mask: Brain Segmentation', title_x=0.5,
                    template="plotly_dark",
                    width=600,
                    height=600,
                    scene_zaxis_range=[-0.1, 12.8],
                    updatemenus = [
                        {
                            "buttons": [
                                {
                                    "args": [None, frame_args(50)],
                                    "label": "&#9654;", # play symbol
                                    "method": "animate",
                                },
                                {
                                    "args": [[None], frame_args(0)],
                                    "label": "&#9724;", # pause symbol
                                    "method": "animate",
                                },
                            ],
                            "direction": "left",
                            "pad": {"r": 0, "t": 60},
                            "type": "buttons",
                            "x": 0,
                            "y": 0,
                        }
                     ],
                     sliders=sliders
                )

    fig1 = go.Figure(data=[initial_slice], layout=layout3d, frames=frames)
    fig2 = go.Figure(data=[initial_slice_m], layout=layout3d_m, frames=frames_m)

    return fig1, fig2

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

colors = {
    'background': '#111111'
}
fig_1,fig_2 = input_image("test4d.nii.gz")
index_page = html.Div(style={'backgroundColor': colors['background']}, children=[html.Div("Brain Tumor Segmentation",style= {"color": "white",
                                                      "text-align": "center","background-color": colors['background'], "font-size": "40px"}),
    dcc.Link(html.Button('Test your Brain Image'), href='/upload'),
    html.Br(),
    html.Div([
        html.Div([
            dcc.Graph(id='g1', figure=fig_1)
        ], className="six columns"),

        html.Div([
            dcc.Graph(id='g2', figure=fig_2)
        ], className="six columns"),
    ], className="row")
])

page_1_layout = html.Div(style={'backgroundColor': colors['background']}, children=[html.Div("Brain Tumor Segmentation",style= {"color": "white",
                                                      "text-align": "center","background-color": colors['background'], "font-size": "40px"}),

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={'color': 'white',
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
    dcc.Link(html.Button('Home Page'), href='/')
]),

def parse_contents(contents):
    img, msk= input_image(contents)
    return html.Div([
        html.Div([
            dcc.Graph(id='g1', figure=img)
        ], className="six columns"),

        html.Div([
            dcc.Graph(id='g2', figure=msk)
        ], className="six columns"),
    ], className="row")


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')])

def update_output(image):
    if not image:
        return

    for i, image_str in enumerate(image):
        data = image_str.encode("utf8").split(b";base64,")[1]
        with open(f"BrainTumorData/imagesTest/image_{i+1}.nii", "wb") as fp:
            fp.write(base64.decodebytes(data))

    children = [parse_contents("image_1.nii")]
    return children

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/upload':
        return page_1_layout
    else:
        return index_page

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
