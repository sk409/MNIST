

class Config:

    __train = True
    __i_type = "int64"
    __f_type = "float64"

    def prediction_scope(self):
        self.__train = False
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__train = True

    @property
    def train(self):
        return self.__train

    @property
    def i_type(self):
        return self.__i_type

    @property
    def f_type(self):
        return self.__f_type


config = Config()