import { ComponentType, useState } from 'react';

import * as S from '@features/canvas/ui/sidebar/menu-accordion.style';

import ArrowButton from '@/shared/ui/button/arrow-button';
import MenuBlock from './menu-block';

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
    <>
      <S.Menu>
        <S.LabelWrapper>
          {Icon && <Icon />}
          {capitalizeFirstLetter(label)}
        </S.LabelWrapper>
        <ArrowButton
          direction={isOpen ? 'up' : 'down'}
          onClick={handleToggleOpen}
        />
      </S.Menu>
      {isOpen && <MenuBlock label='Activation' Icon={Icon} />}
    </>
  );
};

export default MenuAccordion;
