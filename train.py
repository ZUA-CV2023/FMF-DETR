import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rtdetr-r18-WT-Z.yaml')
    # model.load('')  # loading pretrain weights
    model.train(
                data='dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=1,
                workers=4,
                device='0',
                patience=50,
                # resume='runs/train/exp/weights/last.pt', # last.pt path
                project='runs/train',
                name='exp',
                )