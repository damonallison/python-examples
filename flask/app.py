"""A simple flask app.

To run the app on a local dev server (port 5000 by default)

# FLASK_APP is the name of the module (flask app) to import and run. Defaults to app.py
# FLASK_ENV=development will enable hot-reload, also enable the debugger.

$ python -m flask run
$ env FLASK_APP=app.py python -m flask run

#
# To run the app exposed to the world (all public IPs)
#
$ flask run --host=0.0.0.0
"""
from flask import Flask, url_for
from flask import request

# The name of the application's module or package.
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def hello_world():
    return f"Hello, world. Tell me your name at : {url_for('hello_yall', username='yourname')}"

@app.route('/user/<string:username>')
def hello_yall(username):
    """Use <varname> to include variables in a URI

    Type specifiers:

    string: (default) accepts any string without '/'
    int: positive integers
    float: positive floating point numbers
    path: like string, but accepts "/"
    uuid: uuid strings
    """
    return f"Hello there, {username}"

@app.route('/test', methods=["POST"])
def test_request():
    """Retrieve a JSON payload."""
    print(request.get_json())
    return "thanks!"