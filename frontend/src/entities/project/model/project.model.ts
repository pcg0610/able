import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

import { Project } from '@features/home/types/home.type';
import { ProjectNameState, ProjectState } from '@entities/project/types/project.type';

export const useProjectNameStore = create<ProjectNameState>()(
  persist(
    (set) => ({
      projectName: '',
      resultName: '',
      epochName: '',
      setProjectName: (project: string) => set({ projectName: project }),
      setResultName: (result: string) => set({ resultName: result }),
      setEpochName: (epoch: string) => set({ epochName: epoch }),
    }),
    {
      name: 'project-storage',
      storage: createJSONStorage(() => sessionStorage),
    }
  )
);

export const useProjectStore = create<ProjectState>((set) => ({
  currentProject: null,
  setCurrentProject: (project: Project) => set({ currentProject: project }),
  resetProject: () => set({ currentProject: null }),
}));
