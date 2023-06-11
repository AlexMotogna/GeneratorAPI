from flask import Flask, request, send_file, jsonify
from BookCoverGenerator import getBookCovers, load_encoding, load_models
from NewTitleGenerator import loadVocab
from io import BytesIO
from PIL import Image
from base64 import encodebytes
import io


app = Flask(__name__)
wordtoix = None
text_encoder = None
netG = None
netD = None
vocab = None


def get_response_image(img):
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG') 
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') 
    return encoded_img


@app.route("/generateCovers", methods=['POST'])
def generateBookCovers():
    data = request.json
    imgs = getBookCovers(data["title"], wordtoix, netG, netD, text_encoder, vocab, 6)

    encoded_img = []
    for img in imgs:
        encoded_img.append(get_response_image(img))

    return jsonify({'result': encoded_img})


@app.route("/generateCover", methods=['POST'])
def generateBookCover():
    data = request.json
    imgs = getBookCovers(data["title"], wordtoix, netG, netD, text_encoder, vocab, 1)
    img = imgs[0]
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    _, wordtoix, n_words = load_encoding('captions.pickle')
    text_encoder, netG, netD = load_models(n_words)
    vocab = loadVocab()
    app.run(host="0.0.0.0", port=5000, debug=True)
