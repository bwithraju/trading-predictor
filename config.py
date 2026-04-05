# Configuration classes

class APIConfig:
    BASE_URL = 'https://api.example.com'
    TIMEOUT = 30

class DataConfig:
    DATA_SOURCE = 'data_source'
    DATA_PATH = 'data/data.csv'

class ModelConfig:
    MODEL_PATH = 'models/model.pkl'
    INPUT_FEATURES = ['feature1', 'feature2', 'feature3']