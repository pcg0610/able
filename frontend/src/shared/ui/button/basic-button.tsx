import React from 'react';

import { StyledButton } from '@shared/ui/button/basic-button.style';
import { Common } from '@shared/styles/common';

interface BasicButtonProps {
  color?: string;
  backgroundColor?: string;
  text: string;
  icon?: React.ReactNode;
  onClick?: () => void;
}

const BasicButton: React.FC<BasicButtonProps> = ({
  color = Common.colors.white,
  backgroundColor = Common.colors.primary,
  text,
  icon,
  onClick,
}) => {
  return (
    <StyledButton color={color} backgroundColor={backgroundColor} onClick={onClick}>
      {icon && <span>{icon}</span>}
      {text}
    </StyledButton>
  );
};

export default BasicButton;
