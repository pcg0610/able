import type { Option } from '@shared/types/common.type';

export interface ApiSchema {
  projectName: string;
  trainResult: string;
  checkpoint: string;
  uri: string;
  description: string;
}

export interface DeployConfig {
  apiPath: string;
  apiDescription: string;
  selectedOption: Option;
}
