import styled from '@emotion/styled';

export const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 1rem;
  font-family: Arial, sans-serif;
`;

export const TopSection = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

export const ButtonWrapper = styled.div`
  display: flex;
  gap: 0.5rem;
`;

export const Button = styled.button<{ primary?: boolean; danger?: boolean }>`
  padding: 0.5rem 1rem;
  font-size: 0.9rem;
  border: none;
  border-radius: 4px;
  color: white;
  background-color: ${({ primary, danger }) =>
      primary ? '#007bff' : danger ? '#dc3545' : '#ccc'};
  cursor: ${({ disabled }) => (disabled ? 'not-allowed' : 'pointer')};

  &:disabled {
    opacity: 0.6;
  }
`;