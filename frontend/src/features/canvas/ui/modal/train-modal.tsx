import { useState } from 'react';

import Modal from '@shared/ui/modal/modal';
import Input from '@shared/ui/input/input';
import DeviceSelect from '@features/common/ui/dropdown/device-select';

interface TrainConfig {
  epoch: number | null;
  batchSize: number | null;
  device: { index: number | null; name: string };
}

interface TrainModalProps {
  onClose: () => void;
}

const TrainModal = ({ onClose }: TrainModalProps) => {
  const [trainConfig, setTrainConfig] = useState<TrainConfig>({
    epoch: null,
    batchSize: null,
    device: { index: null, name: '' },
  });

  const handleConfigChange = <T extends keyof TrainConfig>(label: T, value: TrainConfig[T]) => {
    setTrainConfig((prev) => ({
      ...prev,
      [label]: value,
    }));
  };

  return (
    <Modal onClose={onClose} title="학습 정보를 입력하세요" confirmText="학습하기">
      <Input
        label="학습 횟수 (epoch)"
        value={trainConfig.epoch ?? ''}
        onChange={(e) => handleConfigChange('epoch', Number(e.target.value))}
      />
      <Input
        label="배치 크기 (batch size)"
        value={trainConfig.batchSize ?? ''}
        onChange={(e) => handleConfigChange('batchSize', Number(e.target.value))}
      />
      <DeviceSelect onSelect={(device) => handleConfigChange('device', device)} />
    </Modal>
  );
};

export default TrainModal;
