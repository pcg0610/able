import { useQuery } from '@tanstack/react-query';
import axiosInstance from '@shared/api/axios-instance';
import homeKey from '@/features/home/api/home-key';

const fetchProjects = async () => {
  try {
    const response = await axiosInstance.get('/projects/');
    const projects = response.data?.data?.projects;
    return projects || [];
  } catch (error) {
    console.error('Failed to fetch projects:', error);
    throw error;
  }
};

const fetchProject = async (title: string) => {
  try {
    const response = await axiosInstance.get(`/projects/${title}`);
    const project = response.data?.data?.project;
    return project || null;
  } catch (error) {
    console.error('Failed to fetch projects:', error);
    throw error;
  }
};

export const useProjects = () => {
  return useQuery({
    queryKey: homeKey.list(),
    queryFn: () => fetchProjects(),
  });
};

export const useProject = (title: string) => {
  return useQuery({
    queryKey: homeKey.project(title),
    queryFn: () => fetchProject(title),
    enabled: !!title,
  });
};
