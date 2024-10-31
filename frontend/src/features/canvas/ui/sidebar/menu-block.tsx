import { ComponentType } from 'react';

import * as S from '@features/canvas/ui/sidebar/menu-block.style';

import MenuIcon from '@assets/icons/menu.svg?react';

interface MenuBlockProps {
  label: string;
  Icon?: ComponentType;
}

const MenuBlock = ({ label, Icon }: MenuBlockProps) => {
  return (
    <S.Container>
      <S.Content>
        <S.LabelWrapper>
          {Icon && <Icon />}
          <S.LabelText>{label}</S.LabelText>
        </S.LabelWrapper>
        <MenuIcon />
      </S.Content>
    </S.Container>
  );
};

export default MenuBlock;
