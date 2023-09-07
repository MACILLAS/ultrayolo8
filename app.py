import io
from PIL import Image
import flask
import numpy as np
import onnxruntime as ort
import cv2
import json

app = flask.Flask(__name__)
ort_sess = ort.InferenceSession('yolov8n.onnx')
model_inputs = ort_sess.get_inputs()
input_shape = model_inputs[0].shape
input_width = input_shape[2]
input_height = input_shape[3]
confidence_thres = 0.5
iou_tresh = 0.5

def preprocess(img):

    # Resize the image to match the input shape
    img = img.resize((input_height, input_height))

    # Normalize the image data by dividing it by 255.0
    image_data = np.array(img) / 255.0

    # Transpose the image to have the channel dimension as the first dimension
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

    # Expand the dimensions of the image data to match the expected input shape
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    # Return the preprocessed image data
    return image_data

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
def postprocess(output, img_width, img_height):
    # Transpose and squeeze the output to match the expected shape
    outputs = np.transpose(np.squeeze(output[0]))

    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor = img_width / input_width
    y_factor = img_height / input_height

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]

        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= confidence_thres:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_tresh)
    data = []
    for i in indices:
        my_dict = {"class_id": class_ids[i], "boxes": boxes[i], "scores": scores[i]}
        data.append(my_dict)

    return json.dumps(data, default=np_encoder)


@app.route('/predict', methods=['POST'])
def predict():
    if flask.request.method == 'POST':
        if flask.request.files.get("img"):
            img = Image.open(io.BytesIO(flask.request.files["img"].read()))
            img_width, img_height = img.width, img.height
            img = preprocess(img)
            output = ort_sess.run(None, {model_inputs[0].name: img})
            output = postprocess(output, img_width, img_height)
        else:
            output = "Error no img"
    else:
        output = "Error no POST"
    return output

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

