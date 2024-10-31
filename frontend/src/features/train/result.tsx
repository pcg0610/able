import * as S from '@features/train/train-feature.style';
import Common from '@shared/styles/common';

import ConfusionMatrix from '@features/train/result/confusion-matrix';
import EpochGraph from '@features/train/result/epoch-graph';
import F1Score from '@features/train/result/f1-score';
import LossGraph from '@features/train/result/loss-graph';
import PerformanceTable from '@features/train/result/performance-table';
import BasicButton from '@shared/ui/button/basic-button'

const Result = () => {
   return (
      <S.Container>
         <S.Header>
            <BasicButton
               color={Common.colors.primary}
               backgroundColor={Common.colors.secondary}
               text="모델 배포하기"
               onClick={() => {
                  console.log('모델 실행 버튼 클릭됨');
               }}
            />
         </S.Header>
         <S.GridContainer>
            <div className="top-row">
               <S.GraphCard>
                  <S.GraphTitle>Training and validation loss</S.GraphTitle>
                  <LossGraph />
               </S.GraphCard>
               <S.GraphCard>
                  <S.GraphTitle>Epoch Accuracy</S.GraphTitle>
                  <EpochGraph />
               </S.GraphCard>
            </div>
            <div className="bottom-row">
               <S.GraphCard>
                  <S.GraphTitle>Confusion Matrix</S.GraphTitle>
                  <ConfusionMatrix />
               </S.GraphCard>
               <S.GraphCard>
                  <S.GraphTitle>F1-Score</S.GraphTitle>
                  <F1Score />
               </S.GraphCard>
               <S.GraphCard>
                  <S.GraphTitle>Performance Matrices Table</S.GraphTitle>
                  <PerformanceTable />
               </S.GraphCard>
            </div>
         </S.GridContainer>
      </S.Container>
   );
};

export default Result;