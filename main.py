import io
from tensorflow import keras
import rembg
import numpy as np
import PIL.Image
import PIL.ImageOps
from keras.preprocessing import image
import json
from keras.preprocessing import image as keras_image
from getClassByIndex import getClassByIndex, checkPlant
import PIL.Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import matplotlib.patches as patches
import os
from fastapi import (
    FastAPI,
    File,
    UploadFile,
)

app = FastAPI(docs_url="/model/docs", redoc_url="/model/redoc")

# Apple_model = keras.models.load_model("./models/Apple_model.h5")
# Cassava_model = keras.models.load_model("./models/Cassava_model.h5")
# Cherry_model = keras.models.load_model("./models/Cherry_model.h5")
# Chili_model = keras.models.load_model("./models/Chili_model.h5")
# Coffee_model = keras.models.load_model("./models/Coffee_model.h5")
# Corn_model = keras.models.load_model("./models/Corn_model.h5")
# Cucumber_model = keras.models.load_model("./models/Cucumber_model.h5")
# Gauva_model = keras.models.load_model("./models/Gauva_model.h5")
# Grape_model = keras.models.load_model("./models/Grape_model.h5")
# Jamun_model = keras.models.load_model("./models/Jamun_model.h5")
# Lemon_model = keras.models.load_model("./models/Lemon_model.h5")
# # Mango_model = keras.models.load_model("./models/Mango_model.h5")
# Peach_model = keras.models.load_model("./models/Peach_model.h5")
# Pepper_bell_model = keras.models.load_model("./models/Pepper_bell_model.h5")
# Pomegranate_model = keras.models.load_model("./models/Pomegranate_model.h5")
# Potato_model = keras.models.load_model("./models/Potato_model.h5")
# Rice_model = keras.models.load_model("./models/Rice_model.h5")
# Soybean_model = keras.models.load_model("./models/Soybean_model.h5")
# Strawberry_model = keras.models.load_model("./models/Strawberry_model.h5")
# Sugarcane_model = keras.models.load_model("./models/Sugarcane_model.h5")
# Tea_model = keras.models.load_model("./models/Tea_model.h5")
# Tomato_model = keras.models.load_model("./models/Tomato_model.h5")
# Wheat_model = keras.models.load_model("./models/Wheat_model.h5")
# all_models = {
#     "Apple": Apple_model,
#     "Cassava": Cassava_model,
#     "Cherry": Cherry_model,
#     "Chili": Chili_model,
#     "Coffee": Coffee_model,
#     "Corn": Corn_model,
#     "Cucumber": Cucumber_model,
#     "Gauva": Gauva_model,
#     "Grape": Grape_model,
#     "Jamun": Jamun_model,
#     "Lemon": Lemon_model,
#     # "Mango": Mango_model,
#     "Peach": Peach_model,
#     "Pepper_bell": Pepper_bell_model,
#     "Pomegranate": Pomegranate_model,
#     "Potato": Potato_model,
#     "Rice": Rice_model,
#     "Soybean": Soybean_model,
#     "Strawberry": Strawberry_model,
#     "Sugarcane": Sugarcane_model,
#     "Tea": Tea_model,
#     "Tomato": Tomato_model,
#     "Wheat": Wheat_model,
# }


def preprocess(input_image, target_size=(224, 224)):
    try:
        # Use rembg to remove the background
        print("1")
        output_image = rembg.remove(input_image)
        print("2")
        # Convert the output image to a NumPy array
        output_array = keras_image.img_to_array(output_image)
        print("3")
        # Ensure the array has shape (height, width, 3) (RGB format)
        output_array_rgb = output_array[:, :, :3]
        print("4")
        # Resize the output array
        output_array_resized = image.smart_resize(output_array_rgb, target_size)
        # output_array_resized = output_array_rgb
        print("5")
        # Add a batch dimension
        output_array_resized = np.expand_dims(output_array_resized, axis=0)
        # Normalize the pixel values
        output_array_resized /= 255.0
        return output_array_resized
    except Exception as e:
        print("The error in preprocessing: ", e)
        return str(e)


@app.get("/model/")
async def root():
    return {"hello"}


def filter_dict_by_indices(dictionary, indices):
    filtered_dict = {}
    for key, value in dictionary.items():
        # if key in ["orig_shape"]:
        #     filtered_dict[key] = value
        if key in ["shape"]:
            continue
        if isinstance(value, list):
            filtered_dict[key] = [value[i] for i in indices if i < len(value)]
    return filtered_dict


yolo_model = YOLO("./models/Yolo_model.pt")


@app.post("/model/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        ## loadin the photo
        contents = await file.read()
        pil_image = PIL.Image.open(io.BytesIO(contents))
        print("image Uploaded")
        ## check if its plant
        # yolo_model.overrides["conf"] = 0.7  # NMS confidence threshold
        yolo_model.overrides["iou"] = 0.45  # NMS IoU threshold
        yolo_model.overrides["agnostic_nms"] = False  # NMS class-agnostic
        yolo_model.overrides["max_det"] = 1000  # maximum number of detections per image
        results = yolo_model.predict(pil_image, conf=0.5)
        if len(results[0].boxes) == 0:
            return {"Message": "not plant"}
        detection_name = []
        names = yolo_model.names
        print("names ", names)
        for r in results:
            for c in r.boxes.cls:
                detection_name.append(names[int(c)])
        print("detection names", detection_name)
        may = []
        idx = 0
        for i in detection_name:
            if checkPlant(i):
                if i in ["soyabean"]:
                    may.append(("Saoybean", idx))
                else:
                    may.append((i.capitalize(), idx))
            idx += 1
        print("maybe", may)
        if len(may) == 0:
            return {"Message": "wrong detection"}
        name, _ = may[0]
        idx = [t[1] for t in may]
        ## preprocess
        print("Starting Preprocess")
        pil_image = preprocess(pil_image)
        image_to_plot = np.squeeze(pil_image, axis=0)
        tmp_img2 = image_to_plot.tolist()
        ## get the correct class
        print("Choosing model")
        model = keras.models.load_model("./models/" + name.capitalize() + "_model.h5")
        # model = Apple_model
        ## go to the correct and predict the disease
        print("predicting")
        predictions = model.predict(pil_image)
        class_index = int(np.argmax(predictions[0]))
        print("getting class")
        predicted_class = getClassByIndex(class_index, name.capitalize())
        yolo_res = {
            # "cls": results[0].boxes.cls.tolist(),
            "conf": results[0].boxes.conf.tolist(),
            # "data": results[0].boxes.data.tolist(),
            "id": results[0].boxes.id,
            # "is_track": results[0].boxes.is_track,
            # "orig_shape": results[0].boxes.orig_shape,
            # "shape": list(results[0].boxes.shape),
            "xywh": results[0].boxes.xywh.tolist(),
            "xywhn": results[0].boxes.xywhn.tolist(),
            "xyxy": results[0].boxes.xyxy.tolist(),
            "xyxyn": results[0].boxes.xyxyn.tolist(),
        }
        filtered_yolo_res = filter_dict_by_indices(yolo_res, idx)
        dic = {
            # "Message": "correct",
            "orig_shape": results[0].boxes.orig_shape,
            "predicted_class": predicted_class,
            "preprocessd image": tmp_img2,
            "Yolo result": filtered_yolo_res,
        }
        print(filtered_yolo_res)
        json_obj = json.dumps(dic)
        return dic
    except Exception as e:
        print("The error is: ", e)
        return str(e)


@app.post("/model/live-detection/")
async def live(file: UploadFile = File(...)):
    try:
        print("image Uploading")
        contents = await file.read()
        pil_image = PIL.Image.open(io.BytesIO(contents))
        ## check if its plant
        # yolo_model = YOLO("./models/Yolo_model.pt")
        print("Predicting")
        results = yolo_model.predict(pil_image)
        print(results[0].boxes.cls)
        if len(results[0].boxes) == 0:
            return {"Message": "not plant"}
        detection_name = []
        names = yolo_model.names
        # print("names ", names)
        for r in results:
            for c in r.boxes.cls:
                detection_name.append(names[int(c)])
        yolo_res = {
            # "cls": results[0].boxes.cls.tolist(),
            "conf": results[0].boxes.conf.tolist(),
            # "data": results[0].boxes.data.tolist(),
            "id": results[0].boxes.id,
            # "is_track": results[0].boxes.is_track,
            "orig_shape": results[0].boxes.orig_shape,
            # "shape": list(results[0].boxes.shape),
            "xywh": results[0].boxes.xywh.tolist(),
            "xywhn": results[0].boxes.xywhn.tolist(),
            "xyxy": results[0].boxes.xyxy.tolist(),
            "xyxyn": results[0].boxes.xyxyn.tolist(),
        }
        dic = {"Yolo result": yolo_res}
        json_obj = json.dumps(dic)
        return json_obj
    except Exception as e:
        print("Error in live detection: ", e)
        return str(e)


import io
from PIL import Image
import shutil

seg_model = YOLO("./models/best_small.pt")


@app.post("/model/segment/")
async def segment(file: UploadFile = File(...)):
    try:
        upload_dir = "./uploads"
        result_dir = "./results"

        contents = await file.read()
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            # contents = await file.read()
            f.write(contents)
        im = PIL.Image.open(file_path)

        p_path = "uploads"
        name = "cur"
        res = seg_model.predict(im, save=True)
        if len(res[0].boxes) == 0:
            return {"Message": "Not in our classes"}
        print(res[0])

        new_im = PIL.Image.open("./runs/segment/predict/" + file.filename)
        # plt.imshow(new_im)
        # plt.show()
        # buf = io.BytesIO()
        # new_im.save(buf, format="JPEG")
        new_im = PIL.Image.open("./runs/segment/predict/" + file.filename)
        # plt.imshow(new_im)
        # plt.show()
        # buf = io.BytesIO()
        # new_im.save(buf, format="JPEG")
        x, y = new_im.size
        new_im = keras_image.img_to_array(new_im)
        new_im = new_im[:, :, :3].tolist()
        # new_im = np.expand_dims(new_im, axis=0).tolist()
        # new_im = np.array(new_im).reshape((x, y, -1)).tolist()
        print(len(new_im), len(new_im[0]))
        # buf = io.BytesIO()
        # new_im.save(buf, format="JPEG")
        # new_im = new_im.tolist()
        pred_classes = res[0].boxes.cls.tolist()
        labels = {
            0: "Chili_whitefly",
            1: "Coffee_healthy",
            2: "Coffee_rust",
            3: "Strawberry_healthy",
            4: "Strawberry_leaf_scorch",
        }
        classes = []
        for i in pred_classes:
            classes.append(labels[int(i)])
        dic = {"Photo": new_im, "predicted_class": classes}
        json_obj = json.dumps(dic)
        try:
            folder_path = "./runs/segment/predict"
            shutil.rmtree(folder_path)
            print("Folder and its content removed")
        except:
            print("Folder not deleted")
        return dic

    except Exception as e:
        print("Error in segmentation", e)
        return str(e)
