import styled from '@emotion/styled';

export const PaginationWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
`;

export const PageNumber = styled.button<{ isActive: boolean }>`
  background-color: ${({ isActive }) => (isActive ? '#e3f2fd' : 'transparent')};
  color: ${({ isActive }) => (isActive ? '#1a73e8' : '#000')};
  border: none;
  padding: 8px 12px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;

  &:hover {
    background-color: #f1f1f1;
  }
`;

export const ArrowButton = styled.button<{ disabled: boolean }>`
  background: none;
  border: none;
  color: ${({ disabled }) => (disabled ? '#d3d3d3' : '#000')};
  cursor: ${({ disabled }) => (disabled ? 'not-allowed' : 'pointer')};
  font-size: 16px;
  padding: 8px;
  transition: color 0.3s;

  &:hover {
    color: ${({ disabled }) => (disabled ? '#d3d3d3' : '#1a73e8')};
  }
`;
