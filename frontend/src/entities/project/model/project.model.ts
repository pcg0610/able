import { create } from 'zustand';

import {
  Project,
  ProjectStore,
  ProjectNameStore,
} from '@features/home/types/home.type';

export const useProjectStateStore = create<ProjectNameStore>((set) => ({
  projectName: '',
  resultName: '',
  epochName: '',
  setProjectName: (project: string) => set({ projectName: project }),
  setResultName: (result: string) => set({ resultName: result }),
  setEpochName: (epoch: string) => set({ epochName: epoch }),
}));

export const useProjectStore = create<ProjectStore>((set) => ({
  currentProject: null,
  setCurrentProject: (project: Project) => set({ currentProject: project }),
  resetProject: () => set({ currentProject: null }),
}));
