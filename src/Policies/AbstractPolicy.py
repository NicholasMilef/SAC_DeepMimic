import os
import abc

class AbstractPolicy(abc.ABC):
    def __init__(self):
        self.name = None
        
    @abc.abstractmethod
    def loadPolicy(self, name):
        pass

    @abc.abstractmethod
    def run(self, state):
        pass