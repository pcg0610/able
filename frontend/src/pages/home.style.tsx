import styled from '@emotion/styled';

export const PageLayout = styled.div`
  display: flex;
  height: calc(100vh - 60px); // 헤더 높이만큼 뺀 값 (필요에 따라 조정)
`;

export const ContentContainer = styled.div`
  flex: 1;
  padding: 20px; // 콘텐츠에 여백 추가
  overflow: auto; // 스크롤이 필요할 경우 자동으로 추가
`;