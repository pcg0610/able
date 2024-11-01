import { ComponentType, useState } from 'react';

import * as S from '@features/canvas/ui/sidebar/menu-accordion.style';

import ArrowButton from '@/shared/ui/button/arrow-button';
import MenuBlock from '@features/canvas/ui/sidebar/menu-block';

interface MenuAccordionProps {
  label: string;
  Icon?: ComponentType;
}

const MenuAccordion = ({ label, Icon }: MenuAccordionProps) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggleOpen = () => {
    setIsOpen(!isOpen);
  };

  const capitalizeFirstLetter = (text: string) => {
    return text.charAt(0).toUpperCase() + text.slice(1);
  };

  return (
    <S.Accordion>
      <S.Menu onClick={handleToggleOpen}>
        <S.LabelWrapper>
          {Icon && <Icon />}
          {capitalizeFirstLetter(label)}
        </S.LabelWrapper>
        <ArrowButton direction={isOpen ? 'up' : 'down'} />
      </S.Menu>
      <S.MenuBlockWrapper isOpen={isOpen}>
        <MenuBlock label='Activation' Icon={Icon} />
        <MenuBlock label='Activation' Icon={Icon} />
        <MenuBlock label='Activation' Icon={Icon} />
        <MenuBlock label='Activation' Icon={Icon} />
        <MenuBlock label='Activation' Icon={Icon} />
      </S.MenuBlockWrapper>
    </S.Accordion>
  );
};

export default MenuAccordion;
