import { useMutation, useQueryClient } from '@tanstack/react-query';
import axiosInstance from '@shared/api/config/axios-instance';

import homeKey from '@features/home/api/home-key';
import { Project } from '@features/home/types/home.type';
import { useProjectNameStore } from '@entities/project/model/project.model';

const createProject = async (projectData: Project): Promise<Project> => {
  const response = await axiosInstance.post('/projects', {
    title: projectData.title,
    description: projectData.description,
    cudaVersion: projectData.cudaVersion,
    pythonKernelPath: projectData.pythonKernelPath,
  });

  if (response.status == 201) {
    return projectData;
  }

  return response.data;
};

export const useCreateProject = () => {
  const queryClient = useQueryClient();
  const { setProjectName } = useProjectNameStore();

  return useMutation({
    mutationFn: createProject,

    onSuccess: (data) => {
      setProjectName(data.title);

      // 프로젝트 리스트 쿼리를 무효화하여 새로고침되도록 설정
      queryClient.invalidateQueries({ queryKey: homeKey.list() });
    },
  });
};
