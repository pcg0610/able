import styled from '@emotion/styled';

import Common from '@shared/styles/common';

export const StyledButton = styled.button<{
  color: string; backgroundColor: string;
  width?: string; height?: string;
}>`
  background-color: ${(props) => props.backgroundColor};
  color: ${(props) => props.color};
  border: none;
  border-radius: .375rem;
  font-size: ${Common.fontSizes.base};
  font-weight: ${Common.fontWeights.semiBold};
  padding: ${(props) => (props.height ? '0' : '0.75rem 1.875rem')};
  width: ${(props) => props.width || 'auto'};
  height: ${(props) => props.height || 'auto'};
  justify-content: center;
  cursor: pointer;
  display: flex;  
  align-items: center;  
  gap: 0.5rem; 

  transition: background-color 0.3s, filter 0.3s;

  &:hover {
    filter: brightness(0.9) saturate(2);
  }

  &:focus {
    outline: none;
  }
`;

export const StyledIcon = styled.span`
  display: flex;
  align-items: center;
`