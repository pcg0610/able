import styled from '@emotion/styled';
import Common from '@shared/styles/common';

export const SidebarContainer = styled.div`
  width: 18rem;
  height: 100%;
  padding: 1rem;
  background-color: ${Common.colors.white};
  box-shadow: .125rem 0 .3125rem rgba(73, 73, 73, 0.1);
  border-right: .0625rem solid rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

export const EpochItem = styled.div`
  padding: 0.75rem;
  background-color: ${Common.colors.gray100};
  border-radius: 0.375rem;
  cursor: pointer;
  &:hover {
    background-color: ${Common.colors.gray200};
  }
`;