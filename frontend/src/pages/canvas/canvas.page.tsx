import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

import * as S from '@pages/canvas/canvas.style';

import CanvasSidebar from '@features/canvas/ui/sidebar/canvas-sidebar';
import CanvasEditor from '@features/canvas/ui/canvas-editor';
import PageHeader from '@widgets/header/page-header';

const CanvasPage = () => {
  return (
    <DndProvider backend={HTML5Backend}>
      <S.PageContainer>
        <PageHeader title='프로젝트' date='2024.08.12' />
        <S.Content>
          <CanvasSidebar />
          <CanvasEditor />
        </S.Content>
      </S.PageContainer>
    </DndProvider>
  );
};

export default CanvasPage;
