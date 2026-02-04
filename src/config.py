MODEL_TYPE = "GIN"          # GIN / GAIN / GGRNet 
NUM_GIN_LAYERS = 3
INPUT_DIM = 9               # atom feature dimension (dummy)
HIDDEN_DIM = 64

EPSILON = 0.0               
POOLING_TYPE = "sum"        # sum / mean / max

TASKS = {
    "toxicity": 1,          
    "solubility": 1,        
    "logP": 1               
}
