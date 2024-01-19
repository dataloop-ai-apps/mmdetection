import dtlpy as dl
import os
from pathlib import Path

from model_adapter import Adapter


def deploy_package(project_id, package_name, code_base_folder_name):
    code_base_path = Path(os.getcwd()).parent
    project = dl.projects.get(project_id=project_id)
    codebase = project.codebases.pack(directory=os.path.join(code_base_path, code_base_folder_name))
    metadata = dl.Package.get_ml_metadata(cls=Adapter,
                                          default_configuration={},
                                          output_type=dl.AnnotationType.BOX
                                          )
    module = dl.PackageModule(
        class_name='Adapter',
        name='model-adapter',
        entry_point='model_adapter.py',
        init_inputs=[dl.FunctionIO(type=dl.PackageInputType.MODEL, name='model_entity')],
        functions=[
            dl.PackageFunction(
                name='predict',
                inputs=[
                    dl.FunctionIO(name='items', type=dl.PackageInputType.ITEMS)
                ],
                outputs=[
                    dl.FunctionIO(name='batch_annotations', type=dl.PackageInputType.ANNOTATIONS)
                ]
            ),
            dl.PackageFunction(
                name='predict_items',
                inputs=[
                    dl.FunctionIO(name='items', type=dl.PackageInputType.ITEMS)
                ],
                outputs=[
                    dl.FunctionIO(name='annotations', type=dl.PackageInputType.ANNOTATIONS)
                ]
            )
        ]
    )
    package = project.packages.push(package_name=package_name,
                                    src_path=os.getcwd(),
                                    package_type='ml',
                                    codebase=codebase,
                                    modules=[module],
                                    is_global=False,
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_S,
                                                                        runner_image='shadimahameeddl/mm3ddetection:latest',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=1,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)
    return package


def create_model(package, artifacts, dataset_id, using_local_artifacts=False):
    try:
        model = package.models.create(model_name=package.name,
                                      dataset_id=dataset_id,
                                      description='mmdetaction model',
                                      tags=['pretrained', 'tutorial'],
                                      configuration={
                                          'weights_filename': 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
                                          'config_filename': 'rtmdet_tiny_8xb32-300e_coco.py',
                                      },
                                      project_id=package.project.id,
                                      model_artifacts=artifacts
                                      )

        model.status = 'trained'
        model.update()
    except dl.exceptions.BadRequest:
        model = package.models.get(model_name=package.name)

    if using_local_artifacts is True:
        for artifact in artifacts:
            model.artifacts.upload(filepath=artifact.local_path)

    if model.status == 'deployed':
        for service_id in model.metadata.get('system', dict()).get('deploy', dict()).get('services', list()):
            service = dl.services.get(service_id=service_id)
            service.update()
    else:
        model.deploy(service_config=package.service_config)
    return model


if __name__ == '__main__':
    dl.setenv('rc')
    code_base_folder_name = 'mmdetection'
    project_id = 'b005023f-290c-4863-9439-d5e1d65842ca'
    dataset_id = '654a8600166cf4636c32434a'
    package_name = 'mmdet-object-detection'
    package = deploy_package(project_id=project_id,
                             package_name=package_name,
                             code_base_folder_name=code_base_folder_name)
    # TODO: Need to update to link artifacts
    artifacts = [dl.LocalArtifact(local_path=r'C:\Users\Liron\PycharmProjects\mmdetection\rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'),
                 dl.LocalArtifact(
                     local_path=r'C:\Users\Liron\PycharmProjects\mmdetection\rtmdet_tiny_8xb32-300e_coco.py'),
                 dl.LocalArtifact(
                     local_path=r'C:\Users\Liron\PycharmProjects\mmdetection\labels.txt')
                 ]

    create_model(package=package, artifacts=artifacts, dataset_id=dataset_id, using_local_artifacts=True)