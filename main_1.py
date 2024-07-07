from pathlib import Path
#from configs.config import MainConfig
from confz import BaseConfig, FileSource
import os
import torch
import numpy as np
from tqdm import tqdm
#from utils.utils import load_detector, load_classificator, open_mapping, extract_crops
import pandas as pd
from itertools import repeat
import csv
from tkinter import *
from PIL import Image
from PIL.ExifTags import TAGS
import customtkinter
import time
from confz import BaseConfig
from typing import Tuple
import sys
from PyQt5.QtWidgets import *


class DetectorArgs(BaseConfig):
    weights: str
    iou: float
    conf: float
    imgsz: Tuple[int, int]
    batch_size: int


class ClassificatorArgs(BaseConfig):
    weights: str
    imgsz: Tuple[int, int]
    batch_size: int


class MainConfig(BaseConfig):
    src_dir: str
    mapping: str
    device: str
    detector: DetectorArgs
    classificator: ClassificatorArgs




from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
from torchvision.transforms.functional import normalize
import numpy as np
from confz import BaseConfig
import cv2
from pathlib import Path


MEAN = [123.675, 116.28, 103.535]
STD = [58.395, 57.12, 57.375]


def load_detector(config: BaseConfig):
    # Loading YOLOv8 weights
    model = YOLO(config.weights)
    return model


def load_classificator(config: BaseConfig):
    # Loading timm model
    model = torch.load(config.weights)
    model.eval()
    return model


def open_mapping(path_mapping: str) -> dict[int, str]:
    with open(path_mapping, 'r') as txt_file:
        lines = txt_file.readlines()
        lines = [i.strip() for i in lines]
        dict_map = {k: v for k, v in enumerate(lines)}
    return dict_map


def extract_crops(results: list[Results], config: BaseConfig) -> dict[str, torch.Tensor]:
    dict_crops = {}
    for res_per_img in results:
        if len(res_per_img) > 0:
            crops_per_img = []
            for box in res_per_img.boxes:
                x0, y0, x1, y1 = box.xyxy.cpu().numpy().ravel().astype(np.int32)
                crop = res_per_img.orig_img[y0: y1, x0: x1]

                # Do squared crop
                # crop = letterbox(img=crop, new_shape=config.imgsz, color=(0, 0, 0))
                crop = cv2.resize(crop, config.imgsz, interpolation=cv2.INTER_LINEAR)

                # Convert Array crop to Torch tensor with [batch, channels, height, width] dimensions
                crop = torch.from_numpy(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
                crop = normalize(crop.float(), mean=MEAN, std=STD)
                crops_per_img.append(crop)

            dict_crops[Path(res_per_img.path).name] = torch.cat(crops_per_img) # if len(crops_per_img) else None
    return dict_crops


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # if not scaleup:  # only scale down, do not scale up (for better test mAP)
    #     r = min(r, 1.0)

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # elif scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img





win = customtkinter.CTk()
win.geometry("1030x600")







def main():
    # Load main config
    main_config = MainConfig(config_sources=FileSource(file=os.path.join("configs", "config.yml")))
    device = main_config.device

    # Load imgs from source dir
    pathes_to_imgs = [i for i in Path(main_config.src_dir).glob("*")
                      if i.suffix.lower() in [".jpeg", ".jpg", ".png"]]

    # Load mapping for classification task
    mapping = open_mapping(path_mapping=main_config.mapping)

    # Separate main config
    detector_config = main_config.detector
    classificator_config = main_config.classificator

    # Load models
    detector = load_detector(detector_config).to(device)
    classificator = load_classificator(classificator_config).to(device)

    # Inference
    if len(pathes_to_imgs):

        list_predictions = []

        num_packages_det = np.ceil(len(pathes_to_imgs) / detector_config.batch_size).astype(np.int32)
        with torch.no_grad():
            for i in tqdm(range(num_packages_det), colour="green"):
                # Inference detector
                batch_images_det = pathes_to_imgs[detector_config.batch_size * i:
                                                  detector_config.batch_size * (1 + i)]
                results_det = detector(batch_images_det,
                                       iou=detector_config.iou,
                                       conf=detector_config.conf,
                                       imgsz=detector_config.imgsz,
                                       verbose=False,
                                       device=device)

                if len(results_det) > 0:
                    # Extract crop by bboxes
                    dict_crops = extract_crops(results_det, config=classificator_config)

                    # Inference classificator
                    for img_name, batch_images_cls in dict_crops.items():
                        # if len(batch_images_cls) > classificator_config.batch_size:
                        num_packages_cls = np.ceil(len(batch_images_cls) / classificator_config.batch_size).astype(
                            np.int32)
                        for j in range(num_packages_cls):
                            batch_images_cls = batch_images_cls[classificator_config.batch_size * j:
                                                                classificator_config.batch_size * (1 + j)]
                            logits = classificator(batch_images_cls.to(device))
                            probabilities = torch.nn.functional.softmax(logits, dim=1)
                            top_p, top_class_idx = probabilities.topk(1, dim=1)

                            # Locate torch Tensors to cpu and convert to numpy
                            top_p = top_p.cpu().numpy().ravel()
                            top_class_idx = top_class_idx.cpu().numpy().ravel()

                            class_names = [mapping[top_class_idx[idx]] for idx, _ in enumerate(batch_images_cls)]

                            list_predictions.extend([[name, cls, prob] for name, cls, prob in
                                                     zip(repeat(img_name, len(class_names)), class_names, top_p)])

        # Create Dataframe with predictions
        table = pd.DataFrame(list_predictions, columns=["image_name", "class_name", "confidence"])
        table.to_csv("table.csv", index=False) # Раскомментируйте, если хотите увидеть результаты предсказания
        # нейронной сети по каждому найденному объекту

        agg_functions = {
            'class_name': ['count'],
            "confidence": ["mean"]
        }
        groupped = table.groupby(['image_name', "class_name"]).agg(agg_functions)
        img_names = groupped.index.get_level_values("image_name").unique()

        final_res = []
        for img_name in img_names:
            imagename = f"{folder}/{img_name}"
            image = Image.open(imagename)
            exifdata = image.getexif()
            for tag_id in exifdata:
                # получить имя тега вместо идентификатора
                tag = TAGS.get(tag_id, tag_id)
                if(tag == "DateTime"):
                    data = exifdata.get(tag_id)
                    # декодировать байты
                    if isinstance(data, bytes):
                        data = data.decode()

                    data_con = data[11:13] * 3600 + data[14:16] * 60 + data[17:19]

                    print(data)
                    print(class_names)

            for i in range(0, -1*len(folder), -1):
                if(folder[i] == "/"):
                    folderf = folder[i+1::]
                    print(folderf)
                    break



            groupped_per_img = groupped.query(f"image_name == '{img_name}'")
            max_num_objects = groupped_per_img["class_name", "count"].max()
            #max_confidence = groupped_per_img["class_name", "confidence"].max()
            statistic_by_max_objects = groupped_per_img[groupped_per_img["class_name", "count"] == max_num_objects]

            if len(statistic_by_max_objects) > 1:
                statistic_by_max_mean_conf = statistic_by_max_objects.reset_index().max().values
                statistic_by_max_mean_conf = statistic_by_max_objects.loc[[statistic_by_max_objects["confidence", "mean"].idxmax()]]
                final_res.extend(statistic_by_max_mean_conf.reset_index().values)
            else:
                final_res.extend(statistic_by_max_objects.reset_index().values)
        groupped.to_csv("table_agg.csv", index=True) # Раскомментируйте, если хотите увидеть результаты аггрегации
        final_table = pd.DataFrame(final_res, columns=["image_name", "class_name", "count", "confidence"])
        final_table.to_csv("table_final.csv", index=False)





def choose_folder():
    class Form(QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)

        def get_directory(self):  # <-----
            return QFileDialog.getExistingDirectory(self, "Выбрать папку", ".")

    if __name__ == '__main__':
        global folder
        app = QApplication(sys.argv)
        ex = Form()
        folder = ex.get_directory()

        folder = folder.replace(" ", "")
        vibor_papk.destroy()

        otmena = customtkinter.CTkButton(win, width=200, height=60, fg_color='#34be2b', hover_color='#24951c',
                                         command=remove_path, text="Отмена", corner_radius=10, border_width=1)
        otmena.place(x=790, y=150)
        puti = customtkinter.CTkLabel(win, width=720, height=60, fg_color='#34be2b', text=folder, corner_radius=10)
        puti.place(x=40, y=50)

def remove_path():
    otmena.destroy()
    puti = customtkinter.CTkLabel(win, width=720, height=60, fg_color='#c3d6e4', text="", corner_radius=10)
    puti.place(x=40, y=50)
    vibor_papk = customtkinter.CTkButton(win, width=200, height=60, fg_color='#34be2b', hover_color='#24951c',
                                         command=choose_folder, text="Выбрать папку", corner_radius=10, border_width=1)
    vibor_papk.place(x=790, y=150)








otmena = customtkinter.CTkButton(win, width=200, height=60, fg_color='#34be2b', hover_color='#24951c', command=remove_path, text="Отмена", corner_radius=10, border_width=1)


puti = customtkinter.CTkLabel(win, width=720, height=60, fg_color='#c3d6e4', text="", corner_radius=10)
puti.place(x=40, y=50)

vibor_papk = customtkinter.CTkButton(win, width=200, height=60, fg_color='#34be2b', hover_color='#24951c', command=choose_folder, text="Выбрать папку", corner_radius=10, border_width=1)
vibor_papk.place(x=790, y=150)

download = customtkinter.CTkButton(win, text="Скачать", corner_radius=10, command=main, fg_color='#34be2b', border_width=1, hover_color='#24951c', width=200, height=60)
download.place(x=790, y=50)

win.mainloop()









