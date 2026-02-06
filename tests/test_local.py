import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app

def test_local():
    hf_token = os.environ.get("CASESTUDY1HF")

    gen = app.respond(
        "Bright and citrusy",
        [],
        512,
        hf_token,
        True,
    )

    response = "".join(gen)
    assert "ethiopian yirgacheffe" in response.lower() #this is the type of coffee the local model recommends when asking for "bright and citrusy", so if it returns this it worked