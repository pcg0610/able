import styled from '@emotion/styled';

export const StyledButton = styled.button<{ color: string; backgroundColor: string }>`
  background-color: ${(props) => props.backgroundColor};
  color: ${(props) => props.color};
  border: none;
  border-radius: 6px;
  padding: 14px 36px;
  font-size: 18px;
  font-weight: bold;
  cursor: pointer;

  transition: background-color 0.3s;

  &:hover {
    background-color: #bbdefb;
  }

  &:focus {
    outline: none;
  }
`;