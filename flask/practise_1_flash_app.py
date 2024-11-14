from flask import Flask, request, jsonify
from functools import wraps
import time
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

user = []

def time_middlerware(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        start = time.time()
        response = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Time taken: {end - start}")
        return response
    return decorated_function

@app.errorhandler(404)
def page_note_f(e):
    return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/')
@time_middlerware
def home():
    return jsonify({"message": "Welcome to the home page"})

@app.route('/static/<path:filename>')
def static_file(filename):
    logger.info(f"Filename: {filename}")
    return app.send_static_file(filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)