import dtlpy as dl
import pathlib

project_name = 'Rekognition_Vision_DEV'
package_name = 'mmlabsobjdetection'


def get_package(project: dl.Project, deploy_new_package):
    ###############
    #   package   #
    ###############
    src_path = str(pathlib.Path('.').resolve())

    if deploy_new_package:

        modules = [
            dl.PackageModule(
                class_name='ServiceRunner',
                name=package_name,
                entry_point='mmlabs_obj_detection.py',
                functions=[
                    dl.PackageFunction(name='detect_obj',
                                       display_name='detect objects',
                                       description='detect objects',
                                       inputs=[dl.FunctionIO(type=dl.PACKAGE_INPUT_TYPE_ITEM, name='item')]
                                       )
                ]
            )
        ]

        package = project.packages.push(package_name=package_name,
                                        modules=modules,
                                        src_path=src_path,
                                        service_config={
                                            'runtime': dl.KubernetesRuntime(num_replicas=1,
                                                                            concurrency=10,
                                                                            runner_image='shadimahameeddl/mm3ddetection:latest',
                                                                            autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                                minReplicas=1,
                                                                                max_replicas=1,
                                                                                queue_length=10)).to_json()
                                        })
        print('New Package has been deployed')
    else:
        package = project.packages.get(package_name=package_name)
        print('Got last package')
    return package


def deploy_service(package: dl.Package):
    project = package.project

    ###############
    #     bot     #
    ###############

    try:
        bot = project.bots.get(bot_name=package.name)
        print("Package {} Bot {} {} has been gotten".format(package.name, bot.name, bot.email))
    except dl.exceptions.NotFound:
        bot = project.bots.create(name=package.name)
        print("New bot has been created: {} email: {}".format(bot.name, bot.email))


    ###########
    # service #
    ###########
    try:
        service = package.services.get(service_name=package.name)
        print("Service has been gotten: ", service.name)
    except dl.exceptions.NotFound:
        service = package.services.deploy(service_name=package.name,
                                          bot=bot,
                                          module_name=package.name)

        print("New service has been created: ", service.name)

    print("package.version: ", package.version)
    print("service.package_revision: ", service.package_revision)
    print("service.runtime.concurrency: ", service.runtime.concurrency)
    service.runtime.autoscaler.print()

    if package.version != service.package_revision:
        service.package_revision = package.version
        service.update()
        print("service.package_revision has been updated: ", service.package_revision)

    else:
        print('No need to update service.package_revision')
    try:
        service.activate_slots(project_id=project.id)
        print("Slot has ben activated")
    except:
        print("Slot is already existing")


def main(project_name):
    project = dl.projects.get(project_name=project_name)
    package = get_package(project, deploy_new_package=True)

    # init params:
    deploy_service(package=package)


if __name__ == "__main__":
    dl.setenv("rc")
    main(project_name=project_name)
