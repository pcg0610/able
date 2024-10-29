import styled from '@emotion/styled';

export const Container = styled.div`
  padding: 20px;
`;

export const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
`;

export const GridContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;

  // 전체 행을 차지할 아이템 스타일
  & > div:nth-of-type(1) {
    grid-column: span 2;
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
