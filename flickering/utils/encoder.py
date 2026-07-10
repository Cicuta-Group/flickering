from numpyencoder import NumpyEncoder
from json import JSONEncoder

class MultiJsonEncoder(NumpyEncoder):
    def default(self, obj):
        if isinstance(obj, list):
            return JSONEncoder.default(obj)

        try:
            return super().default(obj)
        except:
            return f"<not serializable: {str(type(obj))}"