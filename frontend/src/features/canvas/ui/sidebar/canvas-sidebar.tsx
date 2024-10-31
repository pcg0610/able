import * as S from '@/features/canvas/ui/sidebar/canvas-sidebar.style';
import { BLOCK_MENU } from '@features/canvas/costants/block-types.constant';
import MenuAccordion from './menu-accordion';

const CanvasSidebar = () => {
  return (
    <S.SidebarContainer>
      {BLOCK_MENU.map((menu) => (
        <MenuAccordion title={menu.name} />
      ))}
    </S.SidebarContainer>
  );
};

export default CanvasSidebar;
