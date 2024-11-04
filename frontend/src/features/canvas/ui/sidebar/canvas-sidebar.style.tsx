import Common from '@/shared/styles/common';
import styled from '@emotion/styled';

export const SidebarContainer = styled.div`
  width: 16.25rem;
  height: 100%;
  overflow-y: auto;

  display: flex;
  flex-direction: column;
  gap: 1.25rem;

  border-right: 1px solid ${Common.colors.gray200};
  padding: 0.9375rem;

  /* 스크롤바 디자인 숨기기 */
  &::-webkit-scrollbar {
    width: 0; /* 웹킷 브라우저에서 스크롤바 너비를 0으로 설정하여 숨김 */
    background: transparent;
  }

  /* Firefox의 경우 스크롤바 숨기기 */
  scrollbar-width: none; /* Firefox에서 스크롤바 제거 */
  -ms-overflow-style: none; /* IE와 Edge에서 스크롤바 제거 */
`;
