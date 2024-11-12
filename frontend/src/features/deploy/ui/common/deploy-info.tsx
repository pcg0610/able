import { InfoWrapper, InfoSection, InfoText, TitleSection, Title, Status, Label, Value } from '@features/deploy/ui/common/deploy-into.style';

import RocketIcon from '@icons/rocket.svg?react';

const InfoContainer = () => {
   return (
      <InfoWrapper>
         <TitleSection>
            <RocketIcon width={43} height={43} />
            <Title>Server</Title>
         </TitleSection>
         <InfoSection>
            <InfoText>
               <Label>FastAPI</Label>
               <Value>0.33.1</Value>
            </InfoText>
            <InfoText>
               <Label>Port</Label>
               <Value>8080</Value>
            </InfoText>
            <InfoText>
               <Label>Status</Label>
               <Value><Status>running</Status></Value>
            </InfoText>
         </InfoSection>
      </InfoWrapper>
   );
};

export default InfoContainer;
