import { useNavigate } from 'react-router-dom';

import { HISTORY_PAGE_LIMIT } from '@features/home/constants/history.constant';
import type { HistoryItem } from '@features/home/types/home.type';
import { useProjectNameStore } from '@entities/project/model/project.model';

import { HistoryListWrapper, HistoryRow, HistoryCell, StatusText } from '@/features/home/ui/content/history-list.style';

interface HistoryListProps {
  trainSummaries: HistoryItem[];
}

const HistoryList = ({ trainSummaries }: HistoryListProps) => {
  const navigate = useNavigate();
  const { setResultName } = useProjectNameStore();

  const handleHistoryClick = (result: string) => {
    setResultName(result);
    navigate('/train');
  };

  return (
    <HistoryListWrapper>
      <thead>
        <tr>
          <HistoryCell as="th" width="10%">
            회차
          </HistoryCell>
          <HistoryCell as="th" width="40%">
            학습일
          </HistoryCell>
          <HistoryCell as="th" width="20%">
            정확도
          </HistoryCell>
          <HistoryCell as="th" width="20%">
            상태
          </HistoryCell>
        </tr>
      </thead>
      <tbody>
        {trainSummaries.length === 0 ? (
          <HistoryRow>
            <HistoryCell colSpan={4} style={{ height: '18.125rem', textAlign: 'center', verticalAlign: 'middle' }}>
              데이터가 없습니다
            </HistoryCell>
          </HistoryRow>
        ) : (
          Array.from({ length: HISTORY_PAGE_LIMIT }).map((_, index) =>
            trainSummaries[index] ? (
              <HistoryRow
                key={trainSummaries[index].index}
                onClick={() => handleHistoryClick(trainSummaries[index].originDirName)}
              >
                <HistoryCell width="10%">{trainSummaries[index].index}</HistoryCell>
                <HistoryCell width="40%">{trainSummaries[index].date}</HistoryCell>
                <HistoryCell width="20%">{trainSummaries[index].accuracy}</HistoryCell>
                <HistoryCell width="20%">
                  <StatusText status={trainSummaries[index].status}>{trainSummaries[index].status}</StatusText>
                </HistoryCell>
              </HistoryRow>
            ) : (
              <HistoryRow key={`empty-${index}`} style={{ height: '3.125rem' }}>
                <HistoryCell width="10%"></HistoryCell>
                <HistoryCell width="40%"></HistoryCell>
                <HistoryCell width="20%"></HistoryCell>
                <HistoryCell width="20%"></HistoryCell>
              </HistoryRow>
            )
          )
        )}
      </tbody>
    </HistoryListWrapper>
  );
};

export default HistoryList;
