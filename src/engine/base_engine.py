from abc import abstractclassmethod
from utils.register import register_class

@register_class(alias="Engine.Base")
class Engine:
    def __init__(self):
        pass
    
    @staticmethod 
    def add_parser_args(parser): 
        pass

    @abstractclassmethod 
    def get_response(self, messages):
        pass