// src/pages/canvas/canvas-page.tsx
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { ReactFlowProvider } from '@xyflow/react';

import * as S from '@pages/canvas/canvas.style';
import Common from '@/shared/styles/common';

import CanvasSidebar from '@features/canvas/ui/sidebar/canvas-sidebar';
import CanvasEditor from '@features/canvas/ui/canvas-editor';
import PageHeader from '@widgets/header/page-header';
import BasicButton from '@shared/ui/button/basic-button';
import PlayIcon from '@icons/play.svg?react';
import SaveIcon from '@icons/save.svg?react';

const CanvasPage = () => {
  return (
    <DndProvider backend={HTML5Backend}>
      <ReactFlowProvider>
        <S.PageContainer>
          <PageHeader title='프로젝트' />
          <S.Content>
            <CanvasSidebar />
            <CanvasEditor />
            <S.OverlayButton>
              <BasicButton
                text='실행'
                icon={<PlayIcon width={13} height={16} />}
                width='5.5rem'
              />
              <BasicButton
                text='저장'
                color={Common.colors.primary}
                backgroundColor={Common.colors.secondary}
                icon={<SaveIcon />}
                width='5.5rem'
              />
            </S.OverlayButton>
          </S.Content>
        </S.PageContainer>
      </ReactFlowProvider>
    </DndProvider>
  );
};

export default CanvasPage;
