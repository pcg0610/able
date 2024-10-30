import React from 'react';

import { StyledButton } from '@shared/ui/button/basic-button.style';

interface BasicButtonProps {
  color?: string;
  backgroundColor?: string;
  text: string;
  icon?: React.ReactNode;
  onClick?: () => void;
}

const BasicButton: React.FC<BasicButtonProps> = ({
  color = '#0051FF',
  backgroundColor = '#A2C5F9',
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
