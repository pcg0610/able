import styled from '@emotion/styled';
import { Common } from '@shared/styles/common';

export const SidebarContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 0px;
  background-color: ${Common.colors.white};
  width: 60px;
  height: 100vh;
  box-shadow: 2px 0 5px rgba(73, 73, 73, 0.1);
  border-right: 1px solid rgba(0, 0, 0, 0.1);
`;

export const SidebarButton = styled.button<{ active: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  margin: 10px;
  border-radius: 10px;
  background-color: ${({ active }) => (active ? Common.colors.primary : Common.colors.white)};
  color: ${({ active }) => (active ? Common.colors.white : Common.colors.black)};
  border: none;
  cursor: pointer;
  transition: background-color 0.3s, color 0.3s;
`;
