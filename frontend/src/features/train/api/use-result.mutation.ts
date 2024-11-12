import { useMutation, useQueryClient } from '@tanstack/react-query';

import axiosInstance from '@/shared/api/config/axios-instance';
import trainKey from '@features/train/api/train-key';

const fetchDeployList = async ({ projectName, resultName }: FeatureMapProps): Promise<string | null> => {
  try {
    const response = await axiosInstance.post(`/checkpoints/${projectName}/${resultName}`);

    if (response.status == 204) {
      return null;
    }

    return response.data.data.featureMap;
  } catch (error) {
    console.error('Feature map fetch error:', error);
    return '';
  }
};

export const useFetchDeployList = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: fetchDeployList,
    onSuccess: (featureMap: string | null, variables) => {
      queryClient.setQueryData<string | null>(
        trainKey.deployList(variables.projectName, variables.resultName),
        () => featureMap
      );
    },
  });
};
