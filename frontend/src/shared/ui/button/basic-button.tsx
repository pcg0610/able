import { ReactNode } from 'react';

import { StyledButton } from '@shared/ui/button/basic-button.style';
import Common from '@shared/styles/common';

interface BasicButtonProps {
  color?: string;
  backgroundColor?: string;
  text: string;
  icon?: ReactNode;
  onClick?: () => void;
}

const BasicButton = ({
  color = Common.colors.white,
  backgroundColor = Common.colors.primary,
  text,
  icon,
  onClick,
}: BasicButtonProps) => {
  return (
    <StyledButton
      color={color}
      backgroundColor={backgroundColor}
      onClick={onClick}
    >
      {icon && <span>{icon}</span>}
      {text}
    </StyledButton>
  );
};

export default BasicButton;
