import { useEffect, useState } from 'react';

import * as S from '@features/train/ui/result/result.style';
import Common from '@shared/styles/common';
import { useGraphs } from '@features/train/api/use-result.query';
import { useProjectNameStore } from '@entities/project/model/project.model';

import ConfusionMatrix from '@features/train/ui/result/confusion-matrix';
import EpochGraph from '@features/train/ui/result/epoch-graph';
import F1Score from '@features/train/ui/result/f1-score';
import LossGraph from '@features/train/ui/result/loss-graph';
import PerformanceTable from '@features/train/ui/result/performance-table';
import BasicButton from '@shared/ui/button/basic-button';
import { EpochResult } from '@features/train/types/analyze.type';

type LossData = { epoch: number; training: number; validation: number }[];
type AccuracyData = { epoch: number; accuracy: number }[];

const transformLossData = (epochResults: EpochResult[] | null) => {
  if (!epochResults) {
    return [];
  }

  return epochResults.map((item) => ({
    epoch: parseInt(item.epoch, 10),
    training: item.losses.training,
    validation: item.losses.validation,
  }));
};

const transformAccuracyData = (epochResults: EpochResult[] | null) => {
  if (!epochResults) {
    return [];
  }

  return epochResults.map((item) => ({
    epoch: parseInt(item.epoch, 10),
    accuracy: item.accuracies.accuracy,
  }));
};

const Result = () => {
  const { projectName, resultName } = useProjectNameStore();
  const { data: graphs } = useGraphs(projectName, resultName);
  const [lossData, setLossData] = useState<LossData>([]);
  const [accuracyData, setAccuracyData] = useState<AccuracyData>([]);

  useEffect(() => {
    setLossData(transformLossData(graphs?.epochResult || null));
    setAccuracyData(transformAccuracyData(graphs?.epochResult || null));
  }, [graphs]);


  return (
    <S.Container>
      <S.Header>
        <BasicButton
          color={Common.colors.primary}
          backgroundColor={Common.colors.secondary}
          text="모델 배포하기"
          width="10rem"
          onClick={() => {
            console.log('모델 실행 버튼 클릭됨');
          }}
        />
      </S.Header>
      <S.GridContainer>
        <div className="top-row">
          <S.GraphCard>
            <S.GraphTitle>Training and validation loss</S.GraphTitle>
            <LossGraph lossData={lossData} />
          </S.GraphCard>
          <S.GraphCard>
            <S.GraphTitle>Epoch Accuracy</S.GraphTitle>
            <EpochGraph epochData={accuracyData} />
          </S.GraphCard>
        </div>
        <div className="bottom-row">
          <S.GraphCard>
            <S.GraphTitle>Confusion Matrix</S.GraphTitle>
            <ConfusionMatrix />
          </S.GraphCard>
          <S.GraphCard>
            <S.F1ScoreTitle>F1-Score</S.F1ScoreTitle>
            <F1Score f1Score={Number(graphs?.f1Score)} />
          </S.GraphCard>
          <S.GraphCard>
            <S.GraphTitle>Performance Matrices Table</S.GraphTitle>
            <PerformanceTable performanceMetrics={graphs?.performanceMetrics} />
          </S.GraphCard>
        </div>
      </S.GridContainer>
    </S.Container>
  );
};

export default Result;
