import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { RouterProvider } from 'react-router-dom';
import { router } from '@app/router';
import { Global } from '@emotion/react';

import { queryClient } from '@shared/api/query-client';
import { globalStyle } from '@shared/styles/global.style';
import toast, { Toaster, useToasterStore } from 'react-hot-toast';
import { useEffect } from 'react';

const App = () => {
  const { toasts } = useToasterStore();
  const TOAST_LIMIT = 1;

  useEffect(() => {
    toasts
      .filter((t) => t.visible)
      .filter((_, i) => i >= TOAST_LIMIT)
      .forEach((t) => toast.dismiss(t.id));
  }, [toasts]);

  return (
    <>
      <Toaster position='top-center' />
      <QueryClientProvider client={queryClient}>
        <Global styles={globalStyle} />
        <RouterProvider router={router} />
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </>
  );
};

export default App;
