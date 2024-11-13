import { useMutation } from '@tanstack/react-query';

import axiosInstance from '@/shared/api/config/axios-instance';

const runServer = async () => {
  const response = await axiosInstance.get('/deploy/run');
  return response.data;
};

export const useStartServer = () => {
  return useMutation({
    mutationFn: runServer,
  });
};

const stopServer = async () => {
  const response = await axiosInstance.get('/deploy/stop');
  return response.data;
};

export const useStopServer = () => {
  return useMutation({
    mutationFn: stopServer,
  });
};

const restartServer = async () => {
  await axiosInstance.post('/deploy/restart');
};

export const useRestartServer = () => {
  return useMutation({
    mutationFn: restartServer,
  });
};
