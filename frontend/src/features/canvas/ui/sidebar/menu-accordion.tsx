import { ReactNode, useState } from 'react';

import * as S from '@features/canvas/ui/sidebar/menu-accordion.style';

import ArrowButton from '@/shared/ui/button/arrow-button';

interface MenuAccordionProps {
  title: string;
  children?: ReactNode;
}

const MenuAccordion = ({ title }: MenuAccordionProps) => {
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
        <span>{capitalizeFirstLetter(title)}</span>
        <ArrowButton
          direction={isOpen ? 'up' : 'down'}
          onClick={handleToggleOpen}
        />
      </S.Container>
    </>
  );
};

export default MenuAccordion;
