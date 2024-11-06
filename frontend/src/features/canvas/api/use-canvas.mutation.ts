import axiosInstance from '@/shared/api/axios-instance';
import { useMutation } from '@tanstack/react-query';

const saveCanvas = async (projectName: string) => {
  try {
    const response = await axiosInstance.post('/canvas', {
      params: { projectName },
      data: {},
    });
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

export const useCanvas = (projectName: string) => {
  return useMutation({
    mutationFn: () => saveCanvas(projectName),
    onError: () => {
      console.error('캔버스 저장 실패');
    },
  });
};
