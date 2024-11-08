import { ReactNode } from 'react';

import { StyledButton, StyledIcon } from '@shared/ui/button/basic-button.style';
import Common from '@shared/styles/common';

interface BasicButtonProps {
  color?: string;
  backgroundColor?: string;
  width?: string;
  height?: string;
  text: string;
  icon?: ReactNode;
  onClick?: () => void;
}

const BasicButton = ({
  color = Common.colors.white,
  backgroundColor = Common.colors.primary,
  width,
  height,
  text,
  icon,
  onClick,
}: BasicButtonProps) => {
  return (
    <StyledButton color={color} backgroundColor={backgroundColor} width={width} height={height} onClick={onClick}>
      {icon && <StyledIcon>{icon}</StyledIcon>}
      {text}
    </StyledButton>
  );
};

export default BasicButton;
