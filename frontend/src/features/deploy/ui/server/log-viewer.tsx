import { LogContainer, LogText } from '@features/deploy/ui/server/log-viewer.style';

const LogViewer = () => {
   return (
      <LogContainer>
         <LogText>
            {/* 여기에 로그 내용을 추가하세요 */}
            2024-10-23 09:49:22 [Note] [Entrypoint]: Entrypoint script for MariaDB Server 1:11.5.2...
            {/* ... */}
         </LogText>
      </LogContainer>
   );
};

export default LogViewer;