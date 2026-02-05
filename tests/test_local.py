import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app

def test_api_requires_token():

    response = app.respond(
        "Bright and citrusy",
        512,
        0,
        True,
    )
    
    #assert "please log in" not in first.lower()  # shouldn't get warning
    #assert isinstance(first, str)