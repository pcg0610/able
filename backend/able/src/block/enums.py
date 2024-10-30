from enum import Enum

class BlockType(str, Enum):
    TRANSFORM = "Transform"
    LAYER = "Layer"
    ACTIVATION = "Activation"
    LOSS = "Loss"
    OPERATION = "Operation"
    OPTIMIZER = "Optimizer"
    MODULE = "Module"
    DATA = "Data"
    INTERPRETER = "Interpreter"