import io
import os
import pickle
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler

import PIL.Image
from tornado.escape import xhtml_escape
import http

from tornado.options import define, options
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from keras import Model
from keras.models import load_model
from matplotlib import pyplot as plt

from API.DNNFeatureExtractor import DNNFeatureExtractor
import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import socket
# config options
define('port', default=8082, type=int, help='port to run web server on')
define('debug', default=True, help='start app in debug mode')
define('route_to_index', default=False, help='route all requests to index.html')
options.parse_command_line(final=True)

PORT = options.port
DEBUG = options.debug
ROUTE_TO_INDEX = options.route_to_index
def fromParamToIndexPath(Dataset, NumPivots, Method):
    path = "../RetrievalFineTuning/Indexes/Dataset="+str(Dataset)+"/Method="+str(Method)+"/NumPivots="+str(NumPivots)
    return path

def GetIndex(IndexNamePath):
    try:
        index = None
        os.makedirs(os.path.dirname(IndexNamePath), exist_ok=True)
        with open(IndexNamePath, 'rb') as i:
            index = pickle.load(i)
        return index
    except:
        return None

DataName = "CombinedVGG16block2FineTuned1024_0.txt"
indexName = fromParamToIndexPath(DataName, 100, "Kmedoids")
print("Getting index...", indexName)
index = GetIndex(indexName)
if index == None:
    exit()

class FeaturesExtractor():

    def __init__(self, modelName):
        model = load_model(modelName)
        newModel = Model(inputs=model.input, outputs=model.get_layer("global_average_pooling2d").output)
        self.input_size = (224, 224)

        self.featureExtractor = DNNFeatureExtractor(newModel, self.input_size)

    def extractFeatures(self, queryObject):

        features = self.featureExtractor.extractFeaturesFromImage(queryObject)
        return features

Extractor = FeaturesExtractor("../RetrievalFineTuning/VGG16block2FineTuned1024_0")
class WebSocketHadler( tornado.websocket.WebSocketHandler):

    def open(self):

        print('Nuova connessione')

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        if ch == 3:
            return rgba

        assert ch == 4, 'RGBA image has 4 channels.'

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def fromGray2RGB(self, img):
        row, col = img.shape

        if len(img.shape) > 2:
            return img
        img2 = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        return img2
    def on_message(self, message):
        print("Received new request")
        jsonMessage = json.loads(message)
        imageB64 = jsonMessage["image"][len("data:image/png;base64,"):]
        img = PIL.Image.open(io.BytesIO(base64.b64decode(imageB64)))
        img = img.resize((224, 224), PIL.Image.ANTIALIAS)
        img = np.array(img)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = self.rgba2rgb(img)
        if len(img.shape) == 2:
            img = self.fromGray2RGB(img)
        print("Extracting query features...")
        features = Extractor.extractFeatures(img)
        p = index.query(features, k=10, z=200, method="perturbation")
        response = {
            "data": []
        }
        print("Packing results..")
        for o in p:
            image = {
                "className": o[0]
            }
            if o[0] != None:
                with open("../sketches/png/"+o[0]+"/"+o[1], "rb") as f:
                    image["imageData"] = base64.b64encode(f.read()).decode()
            else:
                with open("../mirflickr25k/mirflickr/random/"+o[1],"rb") as f:
                    image["imageData"] = base64.b64encode(f.read()).decode()
            response["data"].append(image)
        print("Sending...")
        self.write_message(json.dumps(response).encode())


    def on_close(self):
        # metodo eseguito alla chiusura della connessione
        print('Connessione chiusa')
    def check_origin(self, origin):
        return True

class MyWebSocketServer(tornado.web.StaticFileHandler):
    def validate_absolute_path(self, root, absolute_path):
        if ROUTE_TO_INDEX and self.request.uri != '/' and not '.' in self.request.uri:
            uri = self.request.uri
            if self.request.uri.endswith('/'):
                uri = uri[:-1]

            absolute_path = absolute_path.replace(uri, '/index.html')

        if os.path.isdir(absolute_path):
            index = os.path.join(absolute_path, 'index.html')
            if os.path.isfile(index):
                return index

            return absolute_path

        return super(MyWebSocketServer, self).validate_absolute_path(root, absolute_path)

    def get_content_type(self):
        if self.absolute_path.endswith('.vtt'):
            return 'text/vtt'

        if self.absolute_path.endswith('.m3u8'):
            return 'application/vnd.apple.mpegurl'

        content_type = super(MyWebSocketServer, self).get_content_type()

        # default to text/html
        if content_type == 'application/octet-stream':
            return 'text/html'

        return content_type

    @classmethod
    def get_content(cls, abspath, start=None, end=None):
        relative_path = abspath.replace(os.getcwd(), '') + '/'

        if os.path.isdir(abspath):
            html = '<html><title>Directory listing for %s</title><body><h2>Directory listing for %s</h2><hr><ul>' % (
            relative_path, relative_path)
            for filename in os.listdir(abspath):
                force_slash = ''
                full_path = filename
                if os.path.isdir(os.path.join(relative_path, filename)[1:]):
                    full_path = os.path.join(relative_path, filename)
                    force_slash = '/'

                html += '<li><a href="%s%s">%s%s</a>' % (
                xhtml_escape(full_path), force_slash, xhtml_escape(filename), force_slash)

            return html + '</ul><hr>'

        if os.path.splitext(abspath)[1] == '.md':
            try:
                import codecs
                import markdown
                input_file = codecs.open(abspath, mode='r', encoding='utf-8')
                text = input_file.read()
                return markdown.markdown(text)
            except:
                import traceback
                traceback.print_exc()
                pass

        return super(MyWebSocketServer, cls).get_content(abspath, start=start, end=end)


settings = {
    'debug': DEBUG,
    'gzip': True,
    'static_handler_class': MyWebSocketServer
}

application = tornado.web.Application([
    (r'/ws', WebSocketHadler),
    (r"/(.*)", MyWebSocketServer, {'path': './web'})

], **settings)

def StartServer():

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(PORT)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    StartServer()




