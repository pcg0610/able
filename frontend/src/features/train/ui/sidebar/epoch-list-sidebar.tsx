import { useState } from 'react';
import { SidebarContainer, EpochItem } from '@features/train/ui/sidebar/epoch-list-sidebar.style';

const EpochListSidebar = () => {
  const [selectedEpoch, setSelectedEpoch] = useState('20241222_epoch_10.pth');

  const epochs = ['20241222_epoch_10.pth', '20241222_epoch_20.pth', '20241222_epoch_30.pth', '20241222_epoch_40.pth'];

  const handleClick = (epoch: string) => {
    setSelectedEpoch(epoch);
  };

  return (
    <SidebarContainer>
      {epochs.map((epoch, index) => (
        <EpochItem key={index} isSelected={selectedEpoch === epoch} onClick={() => handleClick(epoch)}>
          {epoch}
        </EpochItem>
      ))}
    </SidebarContainer>
  );
};

export default EpochListSidebar;
