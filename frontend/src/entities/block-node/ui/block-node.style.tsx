import styled from '@emotion/styled';
import Common from '@shared/styles/common';

export const Container = styled.div<{ blockColor: string; isConnected: boolean; isSelected: boolean }>`
  width: 15.625rem;
  border-radius: 0.25rem;
  background-color: ${({ blockColor }) => blockColor};
  border: 0.0625rem solid ${({ blockColor }) => blockColor};
  overflow: hidden;
  opacity: ${({ isConnected, isSelected }) => (isSelected || isConnected ? 1 : 0.3)};
  box-shadow: ${({ isSelected, blockColor }) => (isSelected ? `0 0 0.4rem ${blockColor}` : '')};
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
    border-bottom: none;
  }
`;

export const Name = styled.label`
  font-size: ${Common.fontSizes.xs};
`;

export const Checkbox = styled.input<{ blockColor: string }>`
  appearance: none;
  width: 1rem;
  height: 1rem;
  border: 1px solid ${Common.colors.gray200};
  border-radius: 0.1rem;
  background-color: ${Common.colors.white};
  cursor: pointer;
  position: relative;

  &:checked {
    background-color: ${({ blockColor }) => blockColor};
    border-color: ${({ blockColor }) => blockColor};
  }

  &:checked::after {
    content: '✓';
    color: ${Common.colors.white};
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: ${Common.fontSizes.sm};
  }

  &:hover {
    border-color: ${({ blockColor }) => blockColor};
  }
`;

export const Input = styled.input`
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

export const Image = styled.img`
  width: 100%;
  max-height: 9.375rem;
  object-fit: contain;
`;

export const CustomUploadContainer = styled.div`
  border: none;
  padding: 1.5625rem;
  min-height: 9.375rem;
  text-align: center;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  transition: border-color 0.3s ease;
  color: #666666;

  &:hover {
    border-color: #666666;
  }
`;

export const HiddenInput = styled.input`
  display: none;
`;

export const BarContainer = styled.div`
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding: 10px;
  background-color: #f8f8f8;
  border: 1px solid #ddd;
  border-radius: 8px;
`;

export const BarWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 0 8px;
`;

export const Bar = styled.div<{ height: number; color: string }>`
  width: 40px;
  height: ${(props) => props.height}px;
  background-color: ${(props) => props.color};
  border-radius: 4px 4px 0 0;
  display: flex;
  align-items: flex-end;
  justify-content: center;
`;

export const BarLabel = styled.span`
  margin-top: 8px;
  font-weight: bold;
  font-size: 12px;
  color: #333;
`;

export const BarScore = styled.span`
  font-size: 12px;
  color: #ffffff;
  margin-bottom: 4px;
  font-weight: bold;
`;

export const GraphButton = styled.button`
  position: absolute;
  top: -30px;
  right: 10px; 
  cursor: pointer;
  background: white;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 4px 3px;
`