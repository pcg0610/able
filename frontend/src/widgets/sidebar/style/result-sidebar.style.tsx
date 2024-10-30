import styled from '@emotion/styled';

export const SidebarContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px;
  background-color: #f9f9f9;
  width: 60px;
  height: 100vh;
  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
`;

export const SidebarButton = styled.button<{ active: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  margin: 10px 0;
  border-radius: 8px;
  background-color: ${({ active }) => (active ? '#1e88e5' : '#fff')};
  color: ${({ active }) => (active ? '#fff' : '#333')};
  border: none;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s, color 0.3s;

  &:hover {
    background-color: #1e88e5;
    color: #fff;
  }
`;
