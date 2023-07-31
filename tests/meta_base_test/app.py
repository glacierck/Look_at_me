from flask import Flask, render_template
import jwt
import time

app = Flask(__name__)

METABASE_SITE_URL = "http://localhost:12345"
METABASE_SECRET_KEY = "f3fd79f57069e7c949c8bd10d165962c6ed0804b8cbef95b93a7deac246b8807"


@app.route('/')
def dashboard():
    payload = {
        "resource": {"dashboard": 2},
        "params": {},
        "exp": round(time.time()) + (60 * 10)  # 10 minute expiration
    }
    token = jwt.encode(payload, METABASE_SECRET_KEY, algorithm="HS256")
    iframeUrl = METABASE_SITE_URL + "/embed/dashboard/" + token + "#theme=night&bordered=true&titled=true"

    return render_template('dashboard.html', iframeUrl=iframeUrl)


if __name__ == '__main__':
    app.run(debug=True)
