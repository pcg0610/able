import { Response } from '@shared/types/response.type';

export interface Device {
  index: number;
  name: string;
}

export interface DeviceResponse extends Response {
  data: {
    devices: Device[];
  };
}
