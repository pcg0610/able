import { ComponentType, useState } from 'react';

import * as S from '@features/canvas/ui/sidebar/menu-accordion.style';

import ArrowButton from '@/shared/ui/button/arrow-button';

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
      <S.Container>
        <S.LabelWrapper>
          {Icon && <Icon />}
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
