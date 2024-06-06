from abc import ABC, abstractmethod

class Observable(ABC):
    _identifier_n = None
    _id_symbol = None

    @property
    def name(self):
        if self._identifier_n is None:
            self._identifier_n = self.__class__.__name__
        return self._identifier_n

    @name.setter
    def name(self, new_name):
        self._identifier_n = new_name

    @property
    def symbol(self):
        if self._id_symbol is None:
            self._id_symbol = self.__class__.__name__
        return self._id_symbol
    
    @symbol.setter
    def symbol(self, new_symbol):
        self._id_symbol = new_symbol

    def __str__(self) -> str:
        return self.symbol
    
    def __repr__(self) -> str:
        return self.name
    

    @abstractmethod
    def compute(self, samples, model):
        raise NotImplementedError