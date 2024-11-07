import styled from '@emotion/styled';
import { keyframes } from '@emotion/react';

import Common from '@shared/styles/common';

const fadeIn = keyframes`
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
`;

const fadeOut = keyframes`
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
`;

export const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.25);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  animation: ${fadeIn} 0.3s ease forwards;
`;

export const ModalWrapper = styled.div`
  width: 25rem;
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 0.25rem 0.9375rem rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  animation: ${fadeIn} 0.3s ease forwards;
`;

export const ModalHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
`;

export const Title = styled.h2`
  font-size: ${Common.fontSizes.lg};
  font-weight: bold;
`;

export const CloseButton = styled.button`
  background: none;
  border: none;
  padding: 0.3125rem 0.625rem;
  font-size: ${Common.fontSizes['2xl']};
  cursor: pointer;
`;

export const ModalBody = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

export const InputWrapper = styled.div`
  display: flex;
  flex-direction: column;
`;

export const Label = styled.label`
  font-size: ${Common.fontSizes.sm};
  margin-bottom: 0.25rem;
`;

export const Input = styled.input`
  padding: 0.5rem;
  border: 0.0625rem solid #ddd;
  border-radius: 0.25rem;
  font-size: ${Common.fontSizes.sm};
  &.readonly {
    background-color: #f0f0f0; 
    color: #888; 
    cursor: not-allowed;
  }
  &::placeholder {
    color: ${Common.colors.gray300}; 
  }
  &:focus {
    border: 0.125rem solid #85b7d9;
    outline: none; 
  }
`;

export const Select = styled.select`
  padding: 0.5rem;
  border: 0.0625rem solid #ddd;
  border-radius: 0.25rem;
  font-size: ${Common.fontSizes.sm};
`;

export const ModalFooter = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 1rem;
`;

export const CancelButton = styled.button`
  background-color: #f5f5f5;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
  cursor: pointer;
  font-size: ${Common.fontSizes.sm};
`;

export const ConfirmButton = styled.button`
  background-color: #007bff;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
  cursor: pointer;
  font-size: ${Common.fontSizes.sm};
`;
