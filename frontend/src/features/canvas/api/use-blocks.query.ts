import axiosInstance from '@shared/api/axios-instance';
import { useQuery } from '@tanstack/react-query';

// get-fetch, post-create, put-update, delete-delete
const fetchBlocks = async (type: string) => {
  try {
    const response = await axiosInstance.get('/blocks', { params: { type } });
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

export const useBlocks = (type: string, isOpen: boolean) => {
  return useQuery({
    queryKey: ['blocks', type],
    queryFn: () => {
      return fetchBlocks(type);
    },
    enabled: isOpen,
  });
};
