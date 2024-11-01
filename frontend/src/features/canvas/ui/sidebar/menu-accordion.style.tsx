import { keyframes } from '@emotion/react';
import styled from '@emotion/styled';

export const Accordion = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.625rem;
`;

export const Menu = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  user-select: none;
`;

export const LabelWrapper = styled.div`
  display: flex;
  align-items: center;
  gap: 0.375rem;
`;

export const MenuBlockWrapper = styled.div<{ isOpen: boolean }>`
  display: flex;
  flex-direction: column;
  gap: 0.625rem;
  overflow: hidden;
  height: ${({ isOpen }) => (isOpen ? 'auto' : '0')};
  animation: ${({ isOpen }) => (isOpen ? slideDown : slideUp)} 0.5s ease
    forwards;
  opacity: ${({ isOpen }) => (isOpen ? 1 : 0)};
`;

const slideDown = keyframes`
  from {
    max-height: 0;
    opacity: 0;
  }
  to {
    max-height: 200px; /* 적당한 최대 높이 */
    opacity: 1;
  }
`;

const slideUp = keyframes`
  from {
    max-height: 200px; /* 적당한 최대 높이 */
    opacity: 1;
  }
  to {
    max-height: 0;
    opacity: 0;
  }
`;
