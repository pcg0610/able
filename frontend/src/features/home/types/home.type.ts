export interface HistoryItem {
  index: number;
  date: string;
  accuracy: string;
  status: string;
}
export interface HistoryResponse {
  totalTrainLogs: number;
  trainSummaries: HistoryItem[];
}
export interface Project {
  title: string;
  description: string;
  cudaVersion: string;
  pythonKernelPath: string;
  thumbnail: string;
}

export interface ProjectNameStore {
  projectName: string;
  resultName: string;
  epochName: string;
  setProjectName: (project: string) => void;
  setResultName: (result: string) => void;
  setEpochName: (epoch: string) => void;
}

export interface ProjectStore {
  currentProject: Project | null;
  setCurrentProject: (project: Project) => void;
  resetProject: () => void;
}
