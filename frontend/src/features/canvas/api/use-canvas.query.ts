import { useQuery } from '@tanstack/react-query';

import axiosInstance from '@shared/api/axios-instance';
import canvasKey from '@features/canvas/api/canvas-key';
import type { TransformedCanvas } from '@features/canvas/types/canvas.type';
import { transformCanvasResponse } from '@features/canvas/utils/canvas-transformer.util';

const fetchCanvas = async (projectName: string) => {
  try {
    const response = await axiosInstance.get('/canvas', {
      params: { projectName },
    });

    if (response.status === 204) {
      return null;
    }

    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

export const useFetchCanvas = (projectName: string) => {
  return useQuery<TransformedCanvas>({
    queryKey: canvasKey.canvas(projectName),
    queryFn: async () => {
      const response = await fetchCanvas(projectName);
      return transformCanvasResponse(response);
    },
    enabled: !!projectName,
  });
};
