import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/exp24/weights/best.pt')
    # model = RTDETR('runs/distill/distill3/weights/best.pt')
    model.val(data='dataset/VisDrone.yaml',
              # split='val',
              split='test',
              imgsz=1280,
              batch=8,
            #   save_json=True, # if you need to cal coco metrice
            #   project='runs/val',
              project='runs/test',
              name='exp',
              )