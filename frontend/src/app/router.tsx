import { createBrowserRouter } from 'react-router-dom';

import HomePage from '@/pages/home.page';
import CanvasPage from '@pages/canvas.page';
import TrainPage from '@/pages/train/train.page';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <HomePage />,
  },
  {
    path: '/canvas',
    element: <CanvasPage />,
  },
  {
    path: '/train',
    element: <TrainPage />,
  },
]);
