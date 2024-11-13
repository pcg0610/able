import * as S from '@features/deploy/ui/common/deploy-into.style';

import RocketIcon from '@icons/rocket.svg?react';
import { useFetchDeployInfo } from '../../api/use-deploy.query';

interface InfoContainerProps {
  title: string;
}

const InfoContainer = ({ title }: InfoContainerProps) => {
  const { data: deployInfo } = useFetchDeployInfo();

  return (
    <S.InfoWrapper>
      <S.TitleSection>
        <RocketIcon width={43} height={43} />
        <S.Title>{title}</S.Title>
      </S.TitleSection>
      <S.InfoSection>
        <S.InfoText>
          <S.Label>FastAPI</S.Label>
          <S.Value>{deployInfo?.apiVersion}</S.Value>
        </S.InfoText>
        <S.InfoText>
          <S.Label>Port</S.Label>
          <S.Value>{deployInfo?.port}</S.Value>
        </S.InfoText>
        <S.InfoText>
          <S.Label>Status</S.Label>
          <S.Value>
            <S.Status>{deployInfo?.status}</S.Status>
          </S.Value>
        </S.InfoText>
      </S.InfoSection>
    </S.InfoWrapper>
  );
};

export default InfoContainer;
