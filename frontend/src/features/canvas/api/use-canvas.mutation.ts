import { useMutation } from '@tanstack/react-query';

import axiosInstance from '@/shared/api/axios-instance';
import type {
  BlockSchema,
  EdgeSchema,
} from '@features/canvas/types/canvas.type';

interface SaveCanvasProps {
  projectName: string;
  canvas: { blocks: BlockSchema[]; edges: EdgeSchema[] };
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
    onSuccess: () => {
      console.log('저장 성공');
    },
  });
};
