import logging

import azure.functions as func
import json

# Import helper script
from .predictonnx import predict_image_from_file, predict_image_from_url


def main(req: func.HttpRequest) -> func.HttpResponse:
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    image_url = req.params.get("img_url")

    if image_url is not None:
        logging.info("Image URL recieved: " + image_url)
        results = predict_image_from_url(image_url)

        return func.HttpResponse(json.dumps(results, indent=4), headers=headers)
    else:
        try:
            file = req.files['file']
            results = predict_image_from_file(file)

            return func.HttpResponse(json.dumps(results, indent=4), headers=headers)
        except:
            logging.error("Error opening file")

    return func.HttpResponse(
        "Please pass a file or img_url in the request",
        headers=headers, status_code=400
    )
