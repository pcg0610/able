import * as S from '@features/deploy/ui/common/deploy-into.style';

import { useFetchDeployInfo } from '@features/deploy/api/use-deploy.query';

import RocketIcon from '@icons/rocket.svg?react';
import Skeleton from '@/shared/ui/loading/skeleton';

interface InfoContainerProps {
  title: string;
}

const InfoContainer = ({ title }: InfoContainerProps) => {
  const { data: deployInfo, isFetching } = useFetchDeployInfo();

  return (
    <S.InfoWrapper>
      <S.TitleSection>
        <RocketIcon width={43} height={43} />
        <S.Title>{title}</S.Title>
      </S.TitleSection>

      <S.InfoSection>
        <S.InfoText>
          <S.Label>FastAPI</S.Label>
          {isFetching ? <Skeleton width={2.9375} height={1.25} /> : <S.Value>{deployInfo?.apiVersion}</S.Value>}
        </S.InfoText>
        <S.InfoText>
          <S.Label>Port</S.Label>
          {isFetching ? <Skeleton width={2.9375} height={1.25} /> : <S.Value>{deployInfo?.port}</S.Value>}
        </S.InfoText>
        <S.InfoText>
          <S.Label>Status</S.Label>
          <S.Value>
            {isFetching ? <Skeleton width={2.9375} height={1.25} /> : <S.Status>{deployInfo?.status}</S.Status>}
          </S.Value>
        </S.InfoText>
      </S.InfoSection>
    </S.InfoWrapper>
  );
};

export default InfoContainer;
