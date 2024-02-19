import os
import subprocess
import dtlpy as dl
from mmdet.apis import init_detector, inference_detector
import logging
import torch

logger = logging.getLogger('MMDetection')


@dl.Package.decorators.module(description='Model Adapter for mmlabs object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class MMDetection(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):

        model_name = self.model_entity.configuration.get('model_name',
                                                         'rtmdet_tiny_8xb32-300e_coco')
        config_file = self.model_entity.configuration.get('config_file',
                                                          'rtmdet_tiny_8xb32-300e_coco.py')
        checkpoint_file = self.model_entity.configuration.get('checkpoint_file',
                                                              'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')

        if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
            logger.info("Downloading mmdet artifacts")
            download_status = subprocess.Popen(f"mim download mmdet --config {model_name} --dest .",
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               shell=True)
            download_status.wait()
            if download_status.returncode != 0:
                (out, err) = download_status.communicate()
                raise Exception(f'Failed to download mmdet artifacts: {err}')

        logger.info("MMDetection artifacts downloaded successfully, Loading Model")
        device = self.model_entity.configuration.get('device', 'cuda:0')
        if device == 'cuda:0':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.confidence_thr = self.model_entity.configuration.get('confidence_thr', 0.4)
        logger.info("MMDetection artifacts downloaded successfully, Loading Model")
        self.model = init_detector(config_file, checkpoint_file, device=device)  # or device='cuda:0'
        logger.info("Model Loaded Successfully")

    def predict(self, batch, **kwargs):
        logger.info(f"Predicting on batch of {len(batch)} images")
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
                                                                       label=self.model_entity.labels[label_id]),
                                          model_info={'name': self.model_entity.name,
                                                      'model_id': self.model_entity.id,
                                                      'confidence': detection_score})
            batch_annotations.append(image_annotations)
            logger.info(f"Found {len(image_annotations)} annotations in image")
        return batch_annotations
