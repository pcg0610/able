from enum import Enum

class BlockType(Enum):
    TRANSFORM = "Transform"
    LAYER = "Layer"
    ACTIVATION = "Activation"
    LOSS = "Loss"
    OPERATION = "Operation"
    OPTIMIZER = "Optimizer"
    MODULE = "Module"