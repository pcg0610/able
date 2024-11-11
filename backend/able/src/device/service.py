import torch

from src.device.dto import DeviceListResponse
from src.device.schema import Device

def get_device_list() -> DeviceListResponse:
    device_list = [Device(index=-1, name='cpu')]

    for index in range(torch.cuda.device_count()):
        device_list.append(Device(index=index, name=torch.cuda.get_device_name(index)))



    return DeviceListResponse(devices=device_list)