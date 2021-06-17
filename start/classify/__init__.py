import logging

import azure.functions as func
import json

# Import helper script
from .predictonnx import predict_image_from_file


def main(req: func.HttpRequest) -> func.HttpResponse:
    file = None

    try:
        file = req.files['file']
        logging.info(file)
    except:
        logging.error('File not found')

    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    if file:
        logging.info('Image file received: ' + file.filename)

        results = predict_image_from_file(file)

        return func.HttpResponse(json.dumps(results, indent=4), headers=headers)
    else:
        return func.HttpResponse(
            "Please pass a file in the request",
            headers=headers, status_code=400
        )
