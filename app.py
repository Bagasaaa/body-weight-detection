import cv2 as cv
import json

from flask import Flask, request, render_template, jsonify

from helper import zoom_at, scale_factor

from flask import Flask, request, jsonify, render_template

from ultralytics import YOLO

app = Flask(__name__, template_folder='templates')

FONT = cv.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)

app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_photo(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/timbangan', methods=['GET', 'POST'])
def timbangan():
    if request.method == 'POST':
        timbangan = request.files.get("timbangan")
        
        # Simpan file yang diunggah
        uploaded_filepath = "user_weight_scale/uploaded_image.jpg"
        timbangan.save(uploaded_filepath)
        
        img = cv.imread(uploaded_filepath)
        img_size = img.shape
        
        new_x, new_y = scale_factor(img_size)

        if img_size == (1920, 1080, 3):
            zoomed_cropped_image = zoom_at(img, 6, coord=(550, 750))
            output_filepath = "user_weight_scale/zoom_and_cropped.jpg"
            cv.imwrite(output_filepath, zoomed_cropped_image)
        else:
            zoomed_cropped_image = zoom_at(img, 6, coord=(new_x, new_y))
            output_filepath = "user_weight_scale/zoom_and_cropped.jpg"
            cv.imwrite(output_filepath, zoomed_cropped_image)
        
        image_path = output_filepath

        model_yolo = YOLO("model/best_1.pt")
        model_yolo.fuse()
        predict_raw = model_yolo.predict(image_path, conf=0.1)
        names = model_yolo.names

        sorted_predictions = []

        for predict in predict_raw:
            boxes = predict.boxes.xyxy
            class_indices = predict.boxes.cls
            
            # Buat daftar pasangan (x_min, class_name) dari setiap prediksi
            for i in range(len(boxes)):
                x_min = boxes[i][0]
                class_index = int(class_indices[i])
                class_name = names[class_index]
                sorted_predictions.append((x_min, class_name))

        # Urutkan prediksi berdasarkan koordinat x_min
        sorted_predictions.sort(key=lambda x: x[0])

        print(sorted_predictions)

        # Simpan class_name ke dalam list
        class_names = [class_name for _, class_name in sorted_predictions]
        print(class_names)

        body_weight_raws = []
        if len(class_names) == 3:
            body_weight_raws.append(class_names[0] + class_names[1] + "." + class_names[2])
        elif len(class_names) == 4:
            body_weight_raws.append(class_names[0] + class_names[1] + class_names[2] + "." + class_names[3])
        elif len(class_names) == 2:
            body_weight_raws.append(class_names[0] + class_names[1])
        else:
            return jsonify({"user_body_weight": "angka belum terdeteksi"})

        # Buat data JSON dari body_weight_raws
        json_data = [{"user_body_weight": body_weight_raws[0] + " kg"}] if body_weight_raws else []

        # Simpan data dalam format JSON
        json_filename = "./results/weight_prediction.json"
        with open(json_filename, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        return jsonify(json_data)
    
    else:
        return render_template('timbangan.html')
    
if __name__ == '__main__':
    app.run()