import yaml


class Configuration:
    """
    Each experiment has own experiment.yaml configuration file. This
    configuration object passes through all methods which required
    additional parameters.
    """
    def __init__(self, file_path):
        """ All parameters saved in .yaml file convert to dot accessible """
        self._file_path = file_path
        self._data = self._read_yaml_file()
        self._check_file(required_keys=['alphabet', 'features_extractor', 'model', 'callbacks', 'optimizer', 'decoder'])
        self.alphabet = self._data.get('alphabet')
        self.features_extractor = self._data.get('features_extractor')
        self.model = self._data.get('model')
        self.callbacks = self._data.get('callbacks')
        self.optimizer = self._data.get('optimizer')
        self.decoder = self._data.get('decoder')


    def _read_yaml_file(self):
        """ Read YAML configuration file """
        with open(self._file_path, 'r') as stream:
            return yaml.load(stream)


    def _check_file(self, required_keys):
        if not all(key in self._data for key in required_keys):
            raise KeyError(f'Configuration file should have all required keys: {required_keys}')
