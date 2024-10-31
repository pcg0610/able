import * as S from '@/features/train/analyze/analyze.style';

import PlayIcon from '@icons/play.svg?react'
import BasicButton from '@shared/ui/button/basic-button'
import EpochListSidebar from '@widgets/sidebar/epoch-list-sidebar';
import CanvasResult from '@features/train/analyze/canvas-result';


const Analyze = () => {
   return (
      <S.AnalyzeContainer>
         <EpochListSidebar />
         <S.ContentWrapper>
            <CanvasResult />
            <BasicButton
               text="추론하기"
               icon={<PlayIcon width={13} height={15} />}
               onClick={() => {
                  console.log('모델 실행 버튼 클릭됨');
               }}
            />
         </S.ContentWrapper>
      </S.AnalyzeContainer>
   );
};

export default Analyze;