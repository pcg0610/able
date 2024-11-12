import { useQuery } from '@tanstack/react-query';

import axiosInstance from '@shared/api/config/axios-instance';
import trainKey from '@features/train/api/train-key';
import type { GraphResponse } from '@features/train/types/analyze.type';

const fetchGraphs = async (projectName: string, resultName: string) => {
  try {
    const response = await axiosInstance.get(`/trains/result/${projectName}/${resultName}`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch epochs:', error);
    throw error;
  }
};

export const useGraphs = (projectName: string, resultName: string) => {
  return useQuery<GraphResponse>({
    queryKey: trainKey.graph(projectName, resultName),
    queryFn: async () => {
      const response = await fetchGraphs(projectName, resultName);
      console.log('dsdas', response.data);
      return response.data;
    },
  });
};
