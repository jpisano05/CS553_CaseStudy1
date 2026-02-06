import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app

def test_local():
    
    hf_token = os.environ.get("CASESTUDY1HF")

    response = app.respond(
        "Bright and citrusy",
        "",
        512,
        hf_token,
        True,
    )
    
    assert "Ethiopian Yirgacheffe"  in response  # this is what it recommends when asking for bright and citrusy, so getting this means it works
    #assert isinstance(first, str)