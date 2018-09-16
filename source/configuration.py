import yaml
from munch import munchify


class Configuration:
    """
    Each experiment has own experiment.yaml configuration file. This
    configuration object passes through all methods which required
    additional parameters.
    """
    def __init__(self, file_path):
        """ All parameters saved in .yaml file convert to dot accessible """
        self._file_path = file_path
        self._data = self.__read_yaml_file()
        self.__set_dot_accessible_attributes()


    def __set_dot_accessible_attributes(self):
        """ Set all parameters saved in .yaml file as object attributes """
        for name, value in munchify(self._data).items():
            setattr(self, name, value)


    def __read_yaml_file(self):
        """ Read YAML configuration file """
        with open(self._file_path, 'r') as stream:
            return yaml.load(stream)
