# MMDetection Model Adapter

## Introduction

An [MMDetection ](https://github.com/open-mmlab/mmdetection/tree/main) Model Adapter implementation for Dataloop

## Installing in the Platform
Navigate to eh marketplace, search for mmdetection and click "Install"!


## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy it, so it can be used
for prediction.

### Deploying with the Platform

In the Model Management page of your project, find a pretrained or fine-tuned version of your <Model Name> model and
click the three dots in the right of the model's row and select "Deploy".

Here you can choose the instance, minimum and maximum number of replicas and queue size of the service that will run the
deployed model (for more information on these parameters,
check [the documentation](https://developers.dataloop.ai/tutorials/faas/advance/chapter/#autoscaler)):

After this, your model is deployed and ready to run inference.

### Deploying with the SDK

To deploy with the default service configuration defined in the package:

```python
import dtlpy as dl

model_entity = dl.models.get(model_id='<model-id>')
model_entity.deploy()
```

For more information and how to set specific service settings for the deployed model, check
the [documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#clone-and-deploy-a-model)
.

## Prediction

### Predicting in the Platform

The best way to perform predictions in the platform is to add a "Predict Node" to a pipeline.

Click [here](https://developers.dataloop.ai/onboarding/08_pipelines/) for more information on Dataloop Pipelines.

### Predicting with the SDK

The deployed model can be used to run prediction on batches of images:

```python
import dtlpy as dl

model_entity = dl.models.get(model_id='<model-id>')
item_id_0 = '<item-id-0>'
results = model_entity.predict_items([item_id_0])
print(results)
```

For more information and
options, [check the documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predict-items).#
