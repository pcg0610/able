import styled from '@emotion/styled';

import Common from '@/shared/styles/common';

export const Container = styled.div<{ isDragging: boolean }>`
  border: 0.1rem solid #34d399;
  border-radius: 0.3125rem;
  background-color: ${Common.colors.white};
  width: 100%;
  position: relative;
  padding: 0.25rem 0;
  opacity: ${({ isDragging }) => (isDragging ? 0.5 : 1)};

  &::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 0.5rem;
    background-color: #34d399;
    border-top-left-radius: 0.15rem;
    border-bottom-left-radius: 0.15rem;
  }
`;

export const Content = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 0.5rem;
`;

export const LabelWrapper = styled.div`
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding-left: 0.5rem;
`;

export const LabelText = styled.span`
  font-size: ${Common.fontSizes.sm};
`;
