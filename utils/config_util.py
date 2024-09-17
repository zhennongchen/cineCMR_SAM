
import collections
import yaml
import re
import os

class AttrDict(dict):
    """Dict as attribute trick."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

class Config(AttrDict):
    r"""Configuration class. This should include every human specifiable
    hyperparameter values for your training."""

    def __init__(self, filename=None, verbose=False):
        super(Config, self).__init__()
        # Update with given configurations.
        assert os.path.exists(filename), "File {} not exist.".format(filename)
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u"tag:yaml.org,2002:float",
            re.compile(
                u"""^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list(u"-+0123456789."),
        )

        try:
            with open(filename, "r") as f:
                cfg_dict = yaml.load(f, Loader=loader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        recursive_update(self, cfg_dict)

    
def recursive_update(d, u):
    """Recursively update AttrDict d with AttrDict u"""
    for key, value in u.items():
        if isinstance(value, collections.abc.Mapping):
            d.__dict__[key] = recursive_update(d.get(key, AttrDict({})), value)
        elif isinstance(value, (list, tuple)):
            if isinstance(value[0], dict):
                d.__dict__[key] = [AttrDict(item) for item in value]
            else:
                d.__dict__[key] = value
        else:
            d.__dict__[key] = value
    return d
