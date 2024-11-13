export interface DeployInfo {
  apiVersion: string;
  port: string;
  status: 'running' | 'stop';
}

export interface ApiListItem {
  projectName: string;
  trainResult: string;
  checkpoint: string;
  uri: string;
  description: string;
}
