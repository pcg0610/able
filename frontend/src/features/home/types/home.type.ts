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
