import { useNavigate } from 'react-router-dom';

import {
  HistoryListWrapper,
  HistoryRow,
  HistoryCell,
  StatusText,
} from '@/features/home/ui/content/history-list.style';

interface HistoryItem {
  id: number;
  date: string;
  accuracy: string;
  status: string;
}

interface HistoryListProps {
  items: HistoryItem[];
}

const HistoryList = ({ items }: HistoryListProps) => {
  const navigate = useNavigate();

  const handleHistoryClick = () => {
    navigate('/train');
  };

  return (
    <HistoryListWrapper>
      <thead>
        <tr>
          <HistoryCell as='th' width='10%'>
            번호
          </HistoryCell>
          <HistoryCell as='th' width='40%'>
            학습일
          </HistoryCell>
          <HistoryCell as='th' width='20%'>
            정확도
          </HistoryCell>
          <HistoryCell as='th' width='20%'>
            상태
          </HistoryCell>
        </tr>
      </thead>
      <tbody>
        {items.map((item, index) => (
          <HistoryRow key={item.id} onClick={handleHistoryClick}>
            <HistoryCell width='10%'>{item.id}</HistoryCell>
            <HistoryCell width='40%'>{item.date}</HistoryCell>
            <HistoryCell width='20%'>{item.accuracy}</HistoryCell>
            <HistoryCell width='20%'>
              <StatusText status={item.status}>{item.status}</StatusText>
            </HistoryCell>
          </HistoryRow>
        ))}
      </tbody>
    </HistoryListWrapper>
  );
};

export default HistoryList;
