from enum import Enum

class DeployStatus(str, Enum):
    STOP="stop"
    RUNNING="running"