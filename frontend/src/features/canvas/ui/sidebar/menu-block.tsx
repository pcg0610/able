import { ComponentType } from 'react';
import { useDrag } from 'react-dnd';

import * as S from '@features/canvas/ui/sidebar/menu-block.style';

import MenuIcon from '@assets/icons/menu.svg?react';

interface MenuBlockProps {
  label: string;
  Icon?: ComponentType;
}

const MenuBlock = ({ label, Icon }: MenuBlockProps) => {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: 'BLOCK', // 드래그 타입을 지정
    item: { label }, // 드래그 시 전달할 데이터
    collect: (monitor) => ({
      isDragging: !!monitor.isDragging(),
    }),
  }));

  return (
    <S.Container ref={drag} isDragging={isDragging}>
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
