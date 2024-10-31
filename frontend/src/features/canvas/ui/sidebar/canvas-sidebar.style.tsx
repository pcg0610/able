import Common from '@/shared/styles/common';
import styled from '@emotion/styled';

export const SidebarContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  width: 16.25rem;
  overflow-y: auto;
  border-right: 1px solid ${Common.colors.gray200};
  padding: 0.9375rem;
`;
