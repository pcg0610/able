import { ComponentType } from 'react';
import { useDrag } from 'react-dnd';

import * as S from '@features/canvas/ui/sidebar/menu-block.style';

import MenuIcon from '@assets/icons/menu.svg?react';
import Tooltip from '@/shared/ui/tooltip/tooltip';

interface MenuBlockProps {
  label: string;
  Icon?: ComponentType;
}

const MenuBlock = ({ label, Icon }: MenuBlockProps) => {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: 'BLOCK',
    item: { label },
    collect: (monitor) => ({
      isDragging: !!monitor.isDragging(),
    }),
  }));

  return (
    <Tooltip text={label}>
      <S.Container ref={drag} isDragging={isDragging}>
        <S.Content>
          <S.LabelWrapper>
            {Icon && <Icon />}
            <S.LabelText>{label}</S.LabelText>
          </S.LabelWrapper>
          <MenuIcon />
        </S.Content>
      </S.Container>
    </Tooltip>
  );
};

export default MenuBlock;
