import os

import dtlpy as dl
from mmdet.apis import init_detector, inference_detector


@dl.Package.decorators.module(description='Model Adapter for mmlabs object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        with open(os.path.join(local_path, "./labels.txt"), "r") as file:
            self.coco_labels = [line.replace('\n', '') for line in file.readlines()]
        config_file = os.path.join(local_path, 'rtmdet_tiny_8xb32-300e_coco.py')
        checkpoint_file = os.path.join(local_path, 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')
        self.model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    def predict(self, batch, **kwargs):
        batch_annotations = list()
        for image in batch:
            image_annotations = dl.AnnotationCollection()
            detections = inference_detector(self.model, image).pred_instances
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
                    image_annotations.add(annotation_definition=dl.Box(top=min_y,
                                                                       left=min_x,
                                                                       bottom=max_y,
                                                                       right=max_x,
                                                                       label=self.coco_labels[label_id]),
                                          model_info={'name': "mmdetection", 'confidence': detection_score})

            batch_annotations.append(image_annotations)
        return batch_annotations
