import * as S from '@/features/canvas/ui/sidebar/canvas-sidebar.style';
import { BLOCK_MENU } from '@features/canvas/costants/block-types.constant';
import MenuAccordion from './menu-accordion';

const CanvasSidebar = () => {
  const capitalizeFirstLetter = (text: string) => {
    return text.charAt(0).toUpperCase() + text.slice(1);
  };

  return (
    <S.SidebarContainer>
      {BLOCK_MENU.map((menu) => (
        <MenuAccordion
          key={menu.name}
          label={capitalizeFirstLetter(menu.name)}
          Icon={menu.icon}
        />
      ))}
    </S.SidebarContainer>
  );
};

export default CanvasSidebar;
