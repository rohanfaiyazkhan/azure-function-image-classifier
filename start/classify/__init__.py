import logging

import azure.functions as func
import json
# from requests_toolbelt.multipart import decoder

# Import helper script
from .predictonnx import ImageNotOpenableError, predict_image_from_file, predict_image_from_url


def main(req: func.HttpRequest) -> func.HttpResponse:
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    content_type = req.headers['Content-Type']

    if content_type == "application/json":
        req_body = req.get_json()
        image_url = req_body['img_url']
        logging.info("Image URL recieved: " + image_url)

        try:
            results = predict_image_from_url(image_url)
            return func.HttpResponse(json.dumps(results), headers=headers)
        except ImageNotOpenableError as e:
            return func.HttpResponse(
                "Unable to open image from URL",
                headers=headers, status_code=400
            )
        except:
            return func.HttpResponse(
                "Sorry something went wrong",
                headers=headers, status_code=500
            )

    elif content_type.startswith("multipart/form-data"):
        try:

            file = req.files['file']

            results = predict_image_from_file(file)

            return func.HttpResponse(json.dumps(results), headers=headers)
        except Exception as e:
            logging.error(e)
            logging.error("Error opening file")

            return func.HttpResponse(
                "Unable to open file passed",
                headers=headers, status_code=400
            )

    else:
        return func.HttpResponse(
            "Please pass a file or img_url in the request",
            headers=headers, status_code=400
        )
