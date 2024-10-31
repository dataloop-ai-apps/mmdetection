# MMDetection Model Adapter

## Introduction

An [MMDetection ](https://github.com/open-mmlab/mmdetection/tree/main) Model Adapter implementation for Dataloop

## Running locally

### Requirements

```commandline
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    mim install mmengine
    mim install "mmcv>=2.0.0"
    mim install mmdet
```

### Running the adapter

```python
import dtlpy as dl

model = dl.models.get(model_id='<model-id>')
item = dl.items.get(item_id='<item-id>')
adapter = MMDetection(model=model)
adapter.predict_items([item_id])
```

### Installing in the Platform

In the Marketplace, search for `MMDetection Model`
Then follow the [documentation](https://docs.dataloop.ai/docs/dl-hub-models) for installing the model.

## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy it, so it can be used
for prediction.

### Using the model on the platform

Follow the dataloop [documentation on the Model Management](https://docs.dataloop.ai/docs/model-management-overview)

### Deploying with the SDK

To deploy with the default service configuration defined in the package:

```python
import dtlpy as dl

model_entity = dl.models.get(model_id='<model-id>')
model_entity.deploy()
```

For more information and
options, [check the documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predict-items).
