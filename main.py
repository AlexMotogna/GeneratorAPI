from flask import Flask, request, send_file
from BookCoverGenerator import getBookCovers, load_encoding, load_models
from NewTitleGenerator import loadVocab
from io import BytesIO

app = Flask(__name__)
wordtoix = None
text_encoder = None
netG = None
netD = None
vocab = None


@app.route("/", methods=['GET'])
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
