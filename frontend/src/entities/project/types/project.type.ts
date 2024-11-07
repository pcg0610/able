import type { Project } from '@features/home/types/home.type';

export interface ProjectNameState {
  projectName: string;
  setProjectName: (project: string) => void;
}

export interface ProjectState {
  currentProject: Project | null;
  setCurrentProject: (project: Project) => void;
  resetProject: () => void;
}
