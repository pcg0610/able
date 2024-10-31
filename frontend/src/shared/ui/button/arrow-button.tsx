import * as S from '@shared/ui/button/arrow-button.style';

import ArrowIcon from '@assets/icons/arrow.svg?react';

interface ArrowButtonProps {
  direction?: 'up' | 'down' | 'left';
  color: string;
  onClick: () => void;
}

const ArrowButton = ({
  direction = 'down',
  color,
  onClick,
}: ArrowButtonProps) => {
  console.log(color);
  return (
    <S.StyledButton direction={direction} onClick={onClick}>
      <ArrowIcon fill={color} />
    </S.StyledButton>
  );
};

export default ArrowButton;
