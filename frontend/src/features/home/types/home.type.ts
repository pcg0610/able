export interface HistoryItem {
  id: number;
  date: string;
  accuracy: string;
  status: string;
}

export interface HistoryListProps {
  items: HistoryItem[];
}

export interface Project {
  title: string;
  description: string;
  cudaVersion: string;
  pythonKernelPath: string;
}

export interface ProjectNameStore {
  projectName: string;
  setProjectName: (project: string) => void;
}

export interface ProjectStore {
  currentProject: Project | null;
  setCurrentProject: (project: Project) => void;
  resetProject: () => void;
}
