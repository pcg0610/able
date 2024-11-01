import { keyframes } from '@emotion/react';
import styled from '@emotion/styled';

export const Accordion = styled.div`
  display: flex;
  flex-direction: column;
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

  overflow: hidden;
  animation: ${({ isOpen }) => (isOpen ? slideDown : slideUp)} 0.4s ease
    forwards;
  opacity: ${({ isOpen }) => (isOpen ? 1 : 1)};
`;

const slideDown = keyframes`
  from {
    max-height: 0;
    opacity: 0;
  }
  to {
    max-height: 200px;
    opacity: 1;
  }
`;

const slideUp = keyframes`
  from {
    max-height: 200px;
    opacity: 1;
  }
  to {
    max-height: 0;
    opacity: 0;
  }
`;
