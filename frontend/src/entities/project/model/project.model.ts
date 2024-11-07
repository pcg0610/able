import { create } from 'zustand';

import {
  Project,
  ProjectStore,
  ProjectNameStore,
} from '@features/home/types/home.type';

export const useProjectStateStore = create<ProjectNameStore>((set) => ({
  projectName: '',
  setProjectName: (project: string) => set({ projectName: project }),
}));

export const useProjectStore = create<ProjectStore>((set) => ({
  currentProject: null,
  setCurrentProject: (project: Project) => set({ currentProject: project }),
  resetProject: () => set({ currentProject: null }),
}));
