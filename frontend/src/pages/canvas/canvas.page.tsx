import * as S from '@pages/canvas/canvas.style';

import CanvasSidebar from '@features/canvas/ui/sidebar/canvas-sidebar';
import CanvasEditor from '@features/canvas/ui/canvas-editor';
import PageHeader from '@widgets/header/page-header';

const CanvasPage = () => {
  return (
    <S.PageContainer>
      <PageHeader title='프로젝트' date='2024.08.12' />
      <S.Content>
        <CanvasSidebar />
        <CanvasEditor />
      </S.Content>
    </S.PageContainer>
  );
};

export default CanvasPage;
