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

export const MenuBlockWrapper = styled.div<{
  isOpen: boolean;
  contentHeight: number;
}>`
  display: flex;
  flex-direction: column;
  overflow: hidden;
  height: ${({ isOpen, contentHeight }) =>
    isOpen ? `${contentHeight}px` : '0'};
  transition: height 0.3s ease;
`;
