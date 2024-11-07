import { ComponentType } from 'react';
import { useDrag } from 'react-dnd';

import * as S from '@features/canvas/ui/sidebar/menu-block.style';
import { blockColors } from '@shared/constants/block';
import { BlockField } from '@features/canvas/types/block.type';
import { capitalizeFirstLetter } from '@shared/utils/formatters.util';

import MenuIcon from '@icons/menu.svg?react';
import Tooltip from '@shared/ui/tooltip/tooltip';

interface MenuBlockProps {
  type: string;
  name: string;
  fields: BlockField[];
  Icon?: ComponentType;
}

const MenuBlock = ({ type, name, fields, Icon }: MenuBlockProps) => {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: 'BLOCK',
    item: { type, name, fields },
    collect: (monitor) => ({
      isDragging: !!monitor.isDragging(),
    }),
  }));

  return (
    <Tooltip text={capitalizeFirstLetter(name)}>
      <S.Container
        ref={drag}
        isDragging={isDragging}
        blockColor={blockColors[type]}
      >
        <S.Content>
          <S.LabelWrapper>
            {Icon && <Icon />}
            <S.LabelText>{capitalizeFirstLetter(name)}</S.LabelText>
          </S.LabelWrapper>
          <MenuIcon />
        </S.Content>
      </S.Container>
    </Tooltip>
  );
};

export default MenuBlock;
