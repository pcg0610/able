import { useMutation } from '@tanstack/react-query';

import axiosInstance from '@/shared/api/axios-instance';
import {
  BlockResponse,
  EdgeResponse,
} from '@features/canvas/types/canvas.type';

interface SaveCanvasProps {
  projectName: string;
  canvas: { blocks: BlockResponse[]; edges: EdgeResponse[] };
}

const saveCanvas = async ({ projectName, canvas }: SaveCanvasProps) => {
  try {
    const response = await axiosInstance.post(
      '/canvas',
      { canvas },
      {
        params: { projectName },
      }
    );
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

export const useSaveCanvas = () => {
  return useMutation({
    mutationFn: saveCanvas,
    onError: () => {
      console.error('캔버스 저장 실패');
    },
  });
};
