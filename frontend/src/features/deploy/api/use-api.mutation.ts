import { useMutation, useQueryClient } from '@tanstack/react-query';

import axiosInstance from '@/shared/api/config/axios-instance';
import deployKey from '@features/deploy/api/deploy-key';
import { ApiSchema } from '@features/deploy/type/deploy.type';

const registerAPI = async ({
  projectName,
  trainResult,
  checkpoint,
  uri,
  description,
}: ApiSchema): Promise<boolean> => {
  try {
    const response = await axiosInstance.post(`/deploy/apis`, {
      project_name: projectName,
      train_result: trainResult,
      checkpoint: checkpoint,
      uri: uri,
      description: description,
    });

    if(response.status == 200){
      return true;
    }

    return false;
    
  } catch (error) {
    console.error('regist api error:', error);
    return false;
  }
};

export const useRegisterAPI = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: registerAPI,
    onSuccess: (featureMap: string | null, variables) => {
      queryClient.setQueryData<string | null>(
        deployKey.regist(variables.projectName, variables.resultName, variables.),
        () => featureMap
      );
    },
  });
};
