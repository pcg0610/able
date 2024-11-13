import * as S from '@features/deploy/ui/common/deploy-into.style';

import RocketIcon from '@icons/rocket.svg?react';

interface InfoContainerProps {
  title: string;
}

const InfoContainer = ({ title }: InfoContainerProps) => {
  return (
    <S.InfoWrapper>
      <S.TitleSection>
        <RocketIcon width={43} height={43} />
        <S.Title>{title}</S.Title>
      </S.TitleSection>
      <S.InfoSection>
        <S.InfoText>
          <S.Label>FastAPI</S.Label>
          <S.Value>0.33.1</S.Value>
        </S.InfoText>
        <S.InfoText>
          <S.Label>Port</S.Label>
          <S.Value>8080</S.Value>
        </S.InfoText>
        <S.InfoText>
          <S.Label>Status</S.Label>
          <S.Value>
            <S.Status>running</S.Status>
          </S.Value>
        </S.InfoText>
      </S.InfoSection>
    </S.InfoWrapper>
  );
};

export default InfoContainer;
