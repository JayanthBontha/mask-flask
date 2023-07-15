# import os
# def allred(n,w,m,s):
#     n=int(n)
#     w=int(w)
#     m=int(m)
#     s=int(s)

#     inten =int( (w+m*2+s*3)/(n+w+m+s))
#     prop = (w+m+s)/(n+w+m+s)
#     if prop==0:
#         return inten,prop
#     if prop<0.01:
#         return 1,inten
#     if prop<0.1:
#         return 2,inten
#     if prop<0.334:
#         return 3,inten
#     if prop<0.667:
#         return 4,inten
#     return 5,inten
# def rename_images(folder_path):
#     # Iterate through all files in the folder
#     for filename in os.listdir(folder_path):
#         # Check if the file is an image
#         if filename.endswith(".png"):
#             # Generate a new name for the image
#             x,y = allred(*filename.split('.')[0].split('_')[3:])
#             new_name = str(x) + '_' + str(y)+ '_' + str(x+y) + '_' + filename
#             print(new_name)
#             # Get the full path of the file
#             file_path = os.path.join(folder_path, filename)
#             new_file_path = os.path.join(folder_path, new_name)
#             # Rename the file
#             os.rename(file_path, new_file_path)

# # Provide the path to the folder containing the images
# folder_path = "./"
# rename_images(folder_path)


from flask import Flask, request,make_response
from predict import *
from flask_cors import CORS
import base64
import os
from PIL import Image
from io import BytesIO
import requests
import imagehash
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

image_hashes = compute_image_hashes("./images")
delete_files_in_folder("./saved")
@app.route('/api/mask', methods=['POST'])
def tile():
    # if(requests.post(url,json={'mfa':request.headers['mfa']}).json()['code']):
    if True:
        ext = request.files['image'].filename.split('.')[-1]
        image = Image.open(request.files['image'])
        matching_image = None
        print(request.headers['cell'])
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

        image.save("saved/"+request.files['image'].filename)
        res = prediict()
        buffer = BytesIO()
        res[0].savefig(buffer, format=ext, bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        response = {
            'match' : matching_image,
            'code' : 0,
            'image': base64.b64encode(buffer.read()).decode('utf-8'),
            'n': res[1],
            'w': res[2],
            'm': res[3],
            's': res[4],
            'all' : res[1]+res[2]+res[3]+res[4]
        }
        resp = make_response(response)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        delete_files_in_folder("./saved")
        return resp
    else:
        return {'code':1}

if __name__ == '__main__':
    app.run()






















from PIL import Image, ImageDraw, ImageFont
import os

# Define the folder paths
main_folder = "./images"
masks_folder = "./stage1_test"
output_folder = "./show2"

# Define the colors for the legend
legend_colors = {
    'N': (0, 255, 0),   # Green
    'W': (0, 0, 255),   # Blue
    'M': (255, 165, 0), # Orange
    'S': (255, 0, 0)    # Red
}

# Define the labels for the legend
legend_labels = {
    'N': 'N - Green',
    'W': 'W - Blue',
    'M': 'M - Orange',
    'S': 'S - Red'
}

# Define the text values
alrred_score = 0.8
proportional_score = 0.75
intensity_score = 0.9

# Iterate through the images in the main folder
for filename in os.listdir(main_folder):
    if filename.endswith(".png"):
        # Extract the base filename without the extension
        base_name = os.path.splitext(filename)[0]
        proportional_score, intensity_score, alrred_score = base_name.split('_')[:3]
        # Find the corresponding mask file
        mask_filename = "_".join(base_name.split('_')[3:6]) + ".png"
        mask_filepath = os.path.join(masks_folder, mask_filename)
        lab = base_name.split('_')[6:]
        legend_labels = {
                        'N': 'N - '+lab[0],
                        'W': 'W - '+lab[1],
                        'M': 'M - '+lab[2],
                        'S': 'S - '+lab[3]
                        }

        # Open the original image and the mask image
        original_image = Image.open(os.path.join(main_folder, filename))
        mask_image = Image.open(mask_filepath)

        # Resize the original image to 400x400 pixels
        original_image = original_image.resize((400, 400))

        # Resize the mask image to 400x400 pixels
        mask_image = mask_image.resize((400, 400))

        # Create a new image with double width to accommodate the original and mask images side by side
        combined_image = Image.new('RGB', (original_image.width * 2 + 300, original_image.height))

        # Paste the original image on the left side and the mask image on the right side
        combined_image.paste(original_image, (0, 0))
        combined_image.paste(mask_image, (original_image.width, 0))

        # Draw labels for "Original" and "Prediction"
        draw = ImageDraw.Draw(combined_image)
        label_font = ImageFont.truetype("arial.ttf", 20)
        draw.text((10, 10), "Original", fill=(255, 255, 255), font=label_font)
        draw.text((original_image.width + 10, 10), "Prediction", fill=(255, 255, 255), font=label_font)

        # Draw legend with color codes and labels
        legend_x = combined_image.width - 290
        legend_y = 10  # Starting position of the legend
        legend_spacing = 30
        legend_font = ImageFont.truetype("arial.ttf", 16)
        for label, color in legend_colors.items():
            draw.rectangle([(legend_x, legend_y), (legend_x + 20, legend_y + 20)], fill=color)
            draw.text((legend_x + 30, legend_y), legend_labels[label], fill=(255, 255, 255), font=legend_font)
            legend_y += legend_spacing

        # Add text elements for scores on the right side
        scores_font = ImageFont.truetype("arial.ttf", 16)
        text_x = combined_image.width - 290
        text_y = combined_image.height - 90
        draw.text((text_x, text_y), f"Alrred Scoring: {alrred_score}", fill=(255, 255, 255), font=scores_font)
        draw.text((text_x, text_y + 20), f"Proportional Score: {proportional_score}", fill=(255, 255, 255), font=scores_font)
        draw.text((text_x, text_y + 40), f"Intensity Score: {intensity_score}", fill=(255, 255, 255), font=scores_font)

        # Save the modified image with a new filename
        new_filename = base_name + ".png"
        combined_image.save(os.path.join(output_folder, new_filename))

        print(f"Processed {filename} and saved as {new_filename}.")
