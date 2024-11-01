import { QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router-dom';
import { router } from '@app/router';
import { Global } from '@emotion/react';

import { queryClient } from '@shared/api/query-client';
import { globalStyle } from '@shared/styles/global.style';

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <Global styles={globalStyle} />
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
};

export default App;
