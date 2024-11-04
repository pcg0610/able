import { useQuery } from '@tanstack/react-query';

import axiosInstance from '@shared/api/axios-instance';
import canvasKey from '@features/canvas/api/canvas-key';
import { BlocksResponse } from '../types/block-types.type';

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
  return useQuery<BlocksResponse>({
    queryKey: canvasKey.blocks(type),
    queryFn: () => fetchBlocks(type),
    enabled: isOpen,
  });
};
