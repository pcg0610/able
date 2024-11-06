import { useMutation, useQueryClient } from '@tanstack/react-query';

import axiosInstance from '@/shared/api/axios-instance';
import type {
  BlockSchema,
  CanvasResponse,
  EdgeSchema,
} from '@features/canvas/types/canvas.type';
import canvasKey from '@features/canvas/api/canvas-key';

interface SaveCanvasProps {
  projectName: string;
  canvas: { blocks: BlockSchema[]; edges: EdgeSchema[] };
}

const saveCanvas = async ({ projectName, canvas }: SaveCanvasProps) => {
  await axiosInstance.post(
    '/canvas',
    { canvas },
    {
      params: { projectName },
    }
  );
};

export const useSaveCanvas = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: saveCanvas,
    onError: () => {
      console.error('캔버스 저장 실패');
    },
    onSuccess: (_data, variables) => {
      queryClient.setQueryData<CanvasResponse>(
        canvasKey.canvas(variables.projectName),
        (oldData) => ({
          ...oldData,
          canvas: {
            blocks: variables.canvas.blocks,
            edges: variables.canvas.edges,
          },
        })
      );
    },
  });
};
