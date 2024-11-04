import styled from '@emotion/styled';
import Common from '@/shared/styles/common';

export const blockColors: Record<string, string> = {
  transform: '#FF6347',
  layer: '#FFD700',
  activation: '#34D399',
  loss: '#5393c8',
  operation: '#FF8C00',
  optimizer: '#6A5ACD',
  model: '#8A2BE2',
};

export const Container = styled.div<{ blockColor: string }>`
  width: 15.625rem;
  border-radius: 0.25rem;
  background-color: ${({ blockColor }) => blockColor};
  border: 0.0625rem solid ${({ blockColor }) => blockColor};
  overflow: hidden;
`;

export const Label = styled.div`
  width: 100%;
  padding: 0.625rem 0.75rem;

  font-size: ${Common.fontSizes.sm};
  font-weight: ${Common.fontWeights.semiBold};
  color: ${Common.colors.white};
`;

export const FieldWrapper = styled.div`
  background-color: ${Common.colors.white};
`;

export const InputWrapper = styled.div<{ blockColor: string }>`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.4rem 0.75rem;
  border-bottom: 0.0625rem solid ${({ blockColor }) => blockColor};

  &:last-of-type {
    border-bottom: none; /* 마지막 필드에는 선 없음 */
  }
`;

export const Name = styled.label`
  font-size: ${Common.fontSizes.xs};
`;

export const Input = styled.input<{ required: boolean }>`
  width: 40%;
  height: 1.5rem;
  padding: 0 0.5rem;

  font-size: ${Common.fontSizes.xs};
  color: ${Common.colors.gray500};
  text-align: end;

  border: 1px solid ${Common.colors.gray200};
  border-radius: 0.1rem;
  outline: none;

  ::placeholder {
    color: ${Common.colors.gray200};
  }
`;
