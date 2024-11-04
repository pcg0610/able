import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

import HistoryList from '@features/home/ui/content/historyList';
import Pagination from '@shared/ui/pagination/pagination';
import * as S from '@features/home/ui/content/homeContent.style';
import WritingIcon from '@icons/writing.svg?react';
import ClockIcon from '@icons/clock.svg?react';
import FolderIcon from '@icons/folder.svg?react';

const HomeContent = () => {

   const historyItems = [
      { id: 1, date: '2024.10.23 16:35', accuracy: '73%', status: '완료' },
      { id: 2, date: '2024.10.23 16:40', accuracy: '85%', status: '진행 중' },
      { id: 3, date: '2024.10.23 16:40', accuracy: '85%', status: '진행 중' },
      { id: 4, date: '2024.10.23 16:40', accuracy: '85%', status: '진행 중' },
      { id: 5, date: '2024.10.23 16:40', accuracy: '85%', status: '진행 중' },
   ];

   const navigate = useNavigate();
   const [currentPage, setCurrentPage] = useState(1);

   const handleCanvasClick = () => {
      navigate('/canvas');
   };

   return (
      <S.HomeContentWrapper>
         <S.Title>
            <FolderIcon width={45} height={45} />
            컴퓨터 비전 프로젝트 <span style={{ fontSize: '14px', color: '#888' }}>python 3.9.6</span>
         </S.Title>
         <div>
            <S.SubTitle>
               <WritingIcon width={30} height={30} />
               작업 중인 캔버스
            </S.SubTitle>
            <S.CanvasImage src="src/assets/Frame 83.png" alt="Canvas Image" onClick={handleCanvasClick} />
         </div>
         <div>
            <S.SubTitle>
               <ClockIcon width={30} height={30} />
               학습 기록
            </S.SubTitle>
            <S.HistoryWrapper>
               <HistoryList items={historyItems} />
               <Pagination currentPage={currentPage} totalPages={26} onPageChange={setCurrentPage} />
            </S.HistoryWrapper>
         </div>
      </S.HomeContentWrapper>
   );
};

export default HomeContent;
