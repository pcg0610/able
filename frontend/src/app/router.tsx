import { createBrowserRouter } from 'react-router-dom';

import Home from '@/pages/home.page';
import Canvas from '@pages/canvas.page';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Home />,
  },
  {
    path: '/canvas',
    element: <Canvas />,
  },
]);
