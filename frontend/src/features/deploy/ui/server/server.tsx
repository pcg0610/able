import * as S from '@features/deploy/ui/server/server.style';
import Common from '@shared/styles/common';
import { useRestartServer, useStartServer, useStopServer } from '@features/deploy/api/use-server.mutation';

import InfoContainer from '@features/deploy/ui/common/deploy-info';
import LogViewer from '@features/deploy/ui/server/log-viewer';
import BasicButton from '@shared/ui/button/basic-button';
import PlayIcon from '@icons/play.svg?react';
import RestartIcon from '@icons/restart.svg?react';
import StopIcon from '@icons/stop.svg?react';

const Server = () => {
  const { mutate: startServer } = useStartServer();
  const { mutate: stopServer } = useStopServer();
  const { mutate: restartServer } = useRestartServer();

  const handleStart = () => {
    startServer();
  };

  const handleRestart = () => {
    restartServer();
  };

  const handleStop = () => {
    stopServer();
  };

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
            onClick={handleStart}
          />
          <BasicButton
            backgroundColor={Common.colors.primary}
            text="RESTART"
            width="8.125rem"
            height="3rem"
            icon={<RestartIcon width={24} height={24} />}
            onClick={handleRestart}
          />
          <BasicButton
            backgroundColor={Common.colors.red}
            text="STOP"
            width="8.125rem"
            height="3rem"
            icon={<StopIcon width={30} height={30} />}
            onClick={handleStop}
          />
        </S.ButtonWrapper>
      </S.TopSection>
      <LogViewer />
    </S.Container>
  );
};

export default Server;
