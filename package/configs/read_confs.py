import yaml

class configs_:
    def __init__(self) -> None:

        with open('package/configs/camera.yml') as confs:
            self.camera_conf = yaml.safe_load(confs)
            confs.close()

        with open('package/configs/camera.yml') as confs:
            self.device_conf = yaml.safe_load(confs)
            confs.close()

        with open('package/configs/ledboard.yml') as confs:
            self.ledboard_conf = yaml.safe_load(confs)
            confs.close()