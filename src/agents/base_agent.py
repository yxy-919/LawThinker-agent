from utils.register import register_class


@register_class(alias="Agent.Base")
class Agent(object):
    def __init__(self, engine):
        self.engine = engine
        self.memories = [("system", self.system_prompt)]

    def memorize(self, message):
        self.memories.append(message)

    def forget(self):
        self.memories = [("system", self.system_message)]