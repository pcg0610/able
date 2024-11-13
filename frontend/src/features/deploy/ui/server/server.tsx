import * as S from '@features/deploy/ui/server/server.style';
import Common from '@shared/styles/common';

import InfoContainer from '@features/deploy/ui/common/deploy-info';
import LogViewer from '@features/deploy/ui/server/log-viewer';
import BasicButton from '@shared/ui/button/basic-button';
import PlayIcon from '@icons/play.svg?react';
import RestartIcon from '@icons/restart.svg?react';
import StopIcon from '@icons/stop.svg?react';

const Server = () => {
  return (
    <S.Container>
      <S.TopSection>
        <InfoContainer title="Server" />
        <S.ButtonWrapper>
          <BasicButton
            backgroundColor={Common.colors.gray200}
            text="START"
            width="8.125rem"
            height="3rem"
            icon={<PlayIcon width={13} height={15} />}
            onClick={() => {
              console.log('시작 버튼 클릭됨');
            }}
          />
          <BasicButton
            backgroundColor={Common.colors.primary}
            text="RESTART"
            width="8.125rem"
            height="3rem"
            icon={<RestartIcon width={24} height={24} />}
            onClick={() => {
              console.log('다시시작 버튼 클릭됨');
            }}
          />
          <BasicButton
            backgroundColor={Common.colors.red}
            text="STOP"
            width="8.125rem"
            height="3rem"
            icon={<StopIcon width={30} height={30} />}
            onClick={() => {
              console.log('멈춤 버튼 클릭됨');
            }}
          />
        </S.ButtonWrapper>
      </S.TopSection>
      <LogViewer />
    </S.Container>
  );
};

export default Server;
