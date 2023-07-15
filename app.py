from flask import Flask, request,make_response
from predict import *
from flask_cors import CORS
import base64
import os
from PIL import Image
from io import BytesIO
import requests
import imagehash
import subprocess
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
url ="https://back-express.onrender.com/api/validity"

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def compute_image_hashes(folder_path):
    image_hashes = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                image = Image.open(file_path)
                hash_value = imagehash.average_hash(image)
                image_hashes[hash_value] = file_path
            except IOError:
                pass  # Skip any files that are not valid images or cannot be opened
    return image_hashes

def b64():
    file_name = next(filename for filename in os.listdir("./saved2") if filename.endswith('.png'))  # Adjust file extensions as needed

    # Construct the full path to the image
    image_path = os.path.join("./saved2", file_name)

    # Open the image file using PIL
    image = Image.open(image_path).resize((200,200))

    # Convert the image to a buffer
    image_buffer = BytesIO()
    image.save(image_buffer, format='png')  # Adjust format as needed

    # Get the buffer value
    buffer_value = image_buffer.getvalue()

    # Encode the buffer as base64
    base64_image = base64.b64encode(buffer_value).decode('utf-8')

    # Print the base64-encoded image
    return base64_image, file_name.split(".")[0].split('_')



image_hashes = compute_image_hashes("./images")
delete_files_in_folder("./saved")
delete_files_in_folder("./saved2")
@app.route('/api/mask', methods=['POST'])
def tile():
    try:
        if(requests.post(url,json={'mfa':request.headers['mfa']}).json()['code']):
            ext = request.files['image'].filename.split('.')[-1]
            image = Image.open(request.files['image'])
            matching_image = None
            image.save("./saved/wtv.png")
            print(request.headers['cell'],type(request.headers['cell']))
            if (request.headers['cell']=="1"):
                hsh = imagehash.average_hash(image)
                matching_image_path = image_hashes.get(hsh)
                if matching_image_path:
                    hsh = imagehash.average_hash(image)
                    matching_image_path = image_hashes.get(hsh)
                    matching_image = Image.open(matching_image_path.replace("images","masks"))
                    image_io2 = BytesIO()
                    matching_image.save(image_io2, format=ext)
                    image_io2.seek(0)
                    matching_image = base64.b64encode(image_io2.getvalue()).decode('utf-8')
            try:
                subprocess.call(["python", "predict.py"])
            except Exception as e:
                print(e)
                return {'code':2}

            f,c = b64()
            print(c)
            response = {
                'match' : matching_image,
                'code' : 0,
                'image': f,
                'n': c[0],
                'w': c[1],
                'm': c[2],
                's': c[3]
            }
            resp = make_response(response)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            delete_files_in_folder("./saved")
            delete_files_in_folder("./saved2")
            return resp
        else:
            return {'code':1}
    except Exception as e:
        print(e)
        return {'code':2}

if __name__ == '__main__':
    app.run()
