import requests
import json
import base64
import argparse
import sys
from os.path import exists, basename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--FILEPATH", help="Image file path", type=str, required=True)
    parser.add_argument("--SERVER", help="Server ip:port (example: 127.0.0.1:5000)", type=str, required=True)
    parser.add_argument("--EMAIL", help="Email to receive output image", type=str, required=True)
    parser.add_argument("--MODEL", help="Model name", type=str, required=True)
    args = parser.parse_args()

    headers = {"Content-Type":"application/json"}
    data = {"filename":basename(args.FILEPATH), "model":args.MODEL, "email":args.EMAIL}
    if not exists(args.FILEPATH):
        print("{} FILE NOT EXISTS".format(args.FILEPATH))
        sys.exit()
    with open(args.FILEPATH, "rb") as f:
        data["image"] = base64.b64encode(f.read())

    url = "http://" + args.SERVER + "/transform"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.text)
