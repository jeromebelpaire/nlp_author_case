from src.apply import apply_fastai_model_on_sentence
from flask import Flask, request
from flask_restful import Resource, Api
from src.config import Config

class Info(Resource):
    def get(self):
        return {
            'Message': ["This is the author classifier"],
            "Available endpoints": ["/sentence?sentence=XXX"]
        }

class ApplyOnSentence(Resource):
    def get(self):
        sentence = request.args.get('sentence')
        result = apply_fastai_model_on_sentence(sentence)
        return {
            'result': [result]
        }

def launch_api():
    '''
    Launch api from config file
    Attention: if there are problems, check if your 
    AntiVirus software does not black the port
    '''
    # Instantiate the app
    app = Flask(__name__)
    api = Api(app)
    config = Config()
    # Create routes
    api.add_resource(Info, '/') 

    # Create routes
    api.add_resource(ApplyOnSentence, '/sentence') 

    host = config.get_api_address()
    port = config.get_api_port()
    app.run(host=host, debug=False, port=port)


if __name__ == '__main__':
    launch_api()