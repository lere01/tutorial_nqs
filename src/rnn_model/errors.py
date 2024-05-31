class ModelTypeError(Exception):
    """Exception raised for errors in the input model type."""

    def __init__(self, model_type, message="Invalid model type. Choose 'rnn' or 'transformer'."):
        self.model_type = model_type
        self.message = message
        super().__init__(self.message)
