import { useEffect, useState } from 'react';

import * as S from '@features/train/ui/result/result.style';
import Common from '@shared/styles/common';

import ConfusionMatrix from '@features/train/ui/result/confusion-matrix';
import EpochGraph from '@features/train/ui/result/epoch-graph';
import F1Score from '@features/train/ui/result/f1-score';
import LossGraph from '@features/train/ui/result/loss-graph';
import PerformanceTable from '@features/train/ui/result/performance-table';
import BasicButton from '@shared/ui/button/basic-button';

const Data = {
  f1_score: '0.4',
  epoch_result: [
    {
      epoch: '0',
      losses: { training: 0.02, validation: 0.12 },
      accuracies: { accuracy: 0.85 },
    },
    {
      epoch: '10',
      losses: { training: 0.07, validation: 0.11 },
      accuracies: { accuracy: 0.88 },
    },
    {
      epoch: '20',
      losses: { training: 0.05, validation: 0.14 },
      accuracies: { accuracy: 0.89 },
    },
    {
      epoch: '30',
      losses: { training: 0.09, validation: 0.13 },
      accuracies: { accuracy: 0.9 },
    },
    {
      epoch: '40',
      losses: { training: 0.1, validation: 0.16 },
      accuracies: { accuracy: 0.91 },
    },
    {
      epoch: '50',
      losses: { training: 0.12, validation: 0.15 },
      accuracies: { accuracy: 0.92 },
    },
    {
      epoch: '60',
      losses: { training: 0.15, validation: 0.14 },
      accuracies: { accuracy: 0.93 },
    },
    {
      epoch: '70',
      losses: { training: 0.18, validation: 0.13 },
      accuracies: { accuracy: 0.94 },
    },
    {
      epoch: '80',
      losses: { training: 0.2, validation: 0.12 },
      accuracies: { accuracy: 0.95 },
    },
    {
      epoch: '90',
      losses: { training: 0.22, validation: 0.07 },
      accuracies: { accuracy: 0.96 },
    },
    {
      epoch: '100',
      losses: { training: 0.24, validation: 0.1 },
      accuracies: { accuracy: 0.97 },
    },
    {
      epoch: '110',
      losses: { training: 0.26, validation: 0.09 },
      accuracies: { accuracy: 0.97 },
    },
    {
      epoch: '120',
      losses: { training: 0.27, validation: 0.08 },
      accuracies: { accuracy: 0.98 },
    },
    {
      epoch: '130',
      losses: { training: 0.28, validation: 0.07 },
      accuracies: { accuracy: 0.98 },
    },
    {
      epoch: '140',
      losses: { training: 0.29, validation: 0.09 },
      accuracies: { accuracy: 0.99 },
    },
    {
      epoch: '150',
      losses: { training: 0.3, validation: 0.05 },
      accuracies: { accuracy: 0.99 },
    },
  ],
};

const transformLossData = (epochResult) => {
  return epochResult.map((item) => ({
    epoch: parseInt(item.epoch, 10),
    training: item.losses.training,
    validation: item.losses.validation,
  }));
};

const transformAccuracyData = (epochResult) => {
  return epochResult.map((item) => ({
    epoch: parseInt(item.epoch, 10),
    accuracy: item.accuracies.accuracy,
  }));
};

const Result = () => {
  const [f1Score, setF1Score] = useState(0.0);
  const [lossData, setLossData] = useState([]);
  const [accuracyData, setAccuracyData] = useState([]);

  useEffect(() => {
    setLossData(transformLossData(Data.epoch_result));
    setAccuracyData(transformAccuracyData(Data.epoch_result));
    setF1Score(parseFloat(Data.f1_score));
  }, []);

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
            <LossGraph data={lossData} />
          </S.GraphCard>
          <S.GraphCard>
            <S.GraphTitle>Epoch Accuracy</S.GraphTitle>
            <EpochGraph data={accuracyData} />
          </S.GraphCard>
        </div>
        <div className="bottom-row">
          <S.GraphCard>
            <S.GraphTitle>Confusion Matrix</S.GraphTitle>
            <ConfusionMatrix />
          </S.GraphCard>
          <S.GraphCard>
            <S.F1ScoreTitle>F1-Score</S.F1ScoreTitle>
            <F1Score f1score={f1Score} />
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
