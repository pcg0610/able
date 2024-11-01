import styled from '@emotion/styled';

import Common from '@shared/styles/common';

export const StyledButton = styled.button<{ color: string; backgroundColor: string }>`
  background-color: ${(props) => props.backgroundColor};
  color: ${(props) => props.color};
  border: none;
  border-radius: .375rem;
  padding: 0.75rem 1.875rem; 
  font-size: ${Common.fontSizes.base};
  font-weight: ${Common.fontWeights.semiBold};
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