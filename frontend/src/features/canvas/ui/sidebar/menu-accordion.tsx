import { useState } from 'react';

import * as S from '@features/canvas/ui/sidebar/menu-accordion.style';
import { MENU_ICON_MAP } from '@features/canvas/costants/block-types.constant';

import ArrowButton from '@/shared/ui/button/arrow-button';

interface MenuAccordionProps {
  label: string;
  icon: keyof typeof MENU_ICON_MAP;
}

const MenuAccordion = ({ label, icon }: MenuAccordionProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const MenuIcon = MENU_ICON_MAP[icon];

  const handleToggleOpen = () => {
    setIsOpen(!isOpen);
  };

  const capitalizeFirstLetter = (text: string) => {
    return text.charAt(0).toUpperCase() + text.slice(1);
  };

  return (
    <>
      <S.Container>
        <S.LabelWrapper>
          {icon && <MenuIcon />}
          {capitalizeFirstLetter(label)}
        </S.LabelWrapper>
        <ArrowButton
          direction={isOpen ? 'up' : 'down'}
          onClick={handleToggleOpen}
        />
      </S.Container>
    </>
  );
};

export default MenuAccordion;
