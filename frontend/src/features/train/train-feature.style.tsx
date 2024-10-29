import styled from '@emotion/styled';

export const Container = styled.div`
  padding: 20px;
`;

export const Header = styled.div`
  display: flex;
  justify-content: end;
  align-items: center;
  margin-bottom: 20px;
`;

export const GridContainer = styled.div`
  display: grid;
  gap: 20px;
  grid-template-rows: auto auto;

  & > .top-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  & > .bottom-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
  }
`;

export const GraphCard = styled.div`
  background: #f9f9f9;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
`;

export const GraphTitle = styled.h3`
  margin-top: 10px;
  font-size: 1rem;
  color: #333;
`;

export const ReleaseButton = styled.button`
  padding: 14px 36px;
  font-size: 18px;
  color: #1e88e5;
  background-color: #e3f2fd;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: bold;
  transition: background-color 0.3s;

  &:hover {
    background-color: #bbdefb;
  }

  &:focus {
    outline: none;
  }
`;