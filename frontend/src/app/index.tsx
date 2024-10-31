import { RouterProvider } from 'react-router-dom';
import { router } from '@/app/router';
import { Global } from '@emotion/react';

import { globalStyle } from '@/shared/styles/global.style';

const App = () => {
  return (
    <>
      <Global styles={globalStyle} />
      <RouterProvider router={router} />
    </>
  );
};

export default App;
