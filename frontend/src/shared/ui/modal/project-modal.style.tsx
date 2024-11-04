// project-modal.style.ts
import styled from '@emotion/styled';
import { keyframes } from '@emotion/react';

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
  width: 400px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  padding: 24px;
  animation: ${fadeIn} 0.3s ease forwards;
`;

export const ModalHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
`;

export const Title = styled.h2`
  font-size: 18px;
  font-weight: bold;
`;

export const CloseButton = styled.button`
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
`;

export const ModalBody = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

export const InputWrapper = styled.div`
  display: flex;
  flex-direction: column;
`;

export const Label = styled.label`
  font-size: 14px;
  margin-bottom: 4px;
`;

export const Input = styled.input`
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  &.readonly {
    background-color: #f0f0f0; 
    color: #888; 
    cursor: not-allowed;
  }
  &::placeholder {
    color: #aaa; 
  }
`;

export const Select = styled.select`
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
`;

export const ModalFooter = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 16px;
`;

export const CancelButton = styled.button`
  background-color: #f5f5f5;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
`;

export const ConfirmButton = styled.button`
  background-color: #007bff;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
`;
