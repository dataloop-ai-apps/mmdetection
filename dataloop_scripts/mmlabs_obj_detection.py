import dtlpy as dl
from mmdet.apis import init_detector, inference_detector
import os
import urllib.request

class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        with open("../labels.txt", "r") as file:
            self.coco_labels = [line.replace('\n', '') for line in file.readlines()]
        print(self.coco_labels)

    def detect_obj(self, item):
        image_path = item.download()
        config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
        checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
        os.system("mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .")
        model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

        builder = item.annotations.builder()
        detections = inference_detector(model, image_path).pred_instances
        all_bboxes = detections.bboxes
        all_labels = detections.labels

        for i, score in enumerate(detections.scores):
            detection_score = float(score)
            if detection_score >= 0.4:
                min_x = int(all_bboxes[i][0])
                min_y = int(all_bboxes[i][1])
                max_x = int(all_bboxes[i][2])
                max_y = int(all_bboxes[i][3])
                label_id = int(all_labels[i])
                builder.add(annotation_definition=dl.Box(top=min_y,
                                                         left=min_x,
                                                         bottom=max_y,
                                                         right=max_x,
                                                         label=self.coco_labels[label_id]),
                            model_info={'name': "mmdetection", 'confidence': detection_score})

        item.annotations.upload(builder)


if __name__ == '__main__':
    model = dl.models.get(model_id='65a9723545ad6d4b305a8546')
    adapter = Adapter(model_entity=model)
    item = dl.items.get(item_id='659dad958625651d89ca8e25')
    adapter.predict_items(items=[item])
