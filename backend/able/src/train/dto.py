from src.canvas.schemas import Canvas
from src.response.schemas import ImmutableBaseModel
from src.train.schemas import Device


class TrainRequest(ImmutableBaseModel):
    project_name: str
    epoch: int
    batch_size: int
    device: Device = Device(index = -1, name='cpu')
    canvas: Canvas

class TrainResponse(ImmutableBaseModel):
    pass

class TrainResultRequest(ImmutableBaseModel):
    project_name: str
    train_result_name: str

class DeviceListResponse(ImmutableBaseModel):
    devices: list[Device]