import styled from '@emotion/styled';

import Common from '@shared/styles/common';

export const HistoryListWrapper = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin-top: 16px;
  font-size: 14px;
  color: #333;
  table-layout: fixed;
`;

export const HistoryRow = styled.tr`
  border-top: 1px solid ${Common.colors.gray100};
  cursor: pointer;

  &:last-of-type {
    border-bottom: none;
  }
`;

export const HistoryCell = styled.td`
  padding: 12px;
  text-align: center;
  width: ${({ width }) => width || 'auto'};
`;

export const StatusText = styled.span<{ status: string }>`
  color: ${({ status }) => (status === '완료' ? '#1a73e8' : '#ff5252')};
  font-weight: bold;
`;
