import object_detection_api
import os
from PIL import Image
from flask import Flask, request, Response

app = Flask(__name__)

PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]


@app.route('/')
def index():
    return Response('Tensor Flow object detection')


@app.route('/image', methods=['POST'])
def process_image():
    """Process image."""
    try:
        # Get the image.
        image_file = request.files['image']
        # Set an image confidence threshold value to limit returned data.
        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = object_detection_api.THRESHOLD
        else:
            threshold = float(threshold)

        # Run the image through Tensorflow object detection.
        image_object = Image.open(image_file)
        objects = object_detection_api.get_objects(image_object, threshold)
        return objects

    except Exception as e:
        print('POST /image error: %e' % e)
        return e


@app.route('/test')
def test():
    """Test API."""

    image = Image.open(TEST_IMAGE_PATHS[0])
    objects = object_detection_api.get_objects(image)
    return objects


@app.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")


@app.route('/video')
def remote():
    return Response(open('./static/video.html').read(), mimetype="text/html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)
