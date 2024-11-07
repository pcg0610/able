import styled from '@emotion/styled';
import Common from '@shared/styles/common';

export const SidebarContainer = styled.div`
  width: 14.5rem;
  height: 100%;
  padding: 1rem;
  background-color: ${Common.colors.white};
  box-shadow: 0.125rem 0 0.3125rem rgba(73, 73, 73, 0.1);
  border-right: 0.0625rem solid rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

export const EpochItem = styled.div<{ isSelected: boolean }>`
  padding: 0.75rem;
  background-color: ${(props) => (props.isSelected ? Common.colors.gray200 : 'transparent')};
  border-radius: 0.375rem;
  font-weight: ${Common.fontWeights.regular};
  font-size: 0.9375rem;
  cursor: pointer;
  transition-property: background-color;
  transition-duration: 0.4s;

  &:hover {
    background-color: ${Common.colors.gray100};
  }
`;
